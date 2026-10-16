//! Compile a validated program into host machine code with Cranelift.
//!
//! Every program becomes a function with the same fixed signature, written
//! in C terms as:
//!
//! ```c
//! int32_t f(const int64_t *args, size_t args_len, int64_t *result);
//! ```
//!
//! It returns one of the `STATUS_*` codes:
//!
//! - `STATUS_OK`: `*result` holds the return value.
//! - `STATUS_OVERFLOW`: an `ADD`, `SUBTRACT` or `MULTIPLY` overflowed signed
//!   64-bit arithmetic; `*result` holds the index of that instruction.
//! - `STATUS_BAD_ARITY`: `args_len` is not the compiled argument count.
//!   Nothing is read from `args` and `*result` is not written.
//!
//! Failures are ordinary branches to a block that returns a status code.
//! The generated code never traps, never unwinds and never calls anything.

use std::sync::{Mutex, OnceLock};

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{AbiParam, BlockArg, InstBuilder, MemFlagsData, Value, types};
use cranelift_codegen::isa::OwnedTargetIsa;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module, default_libcall_names};

use crate::program::{self, Op};

pub const STATUS_OK: i32 = 0;
pub const STATUS_OVERFLOW: i32 = 1;
pub const STATUS_BAD_ARITY: i32 = 2;

/// The Rust view of the generated function's signature. Cranelift builds
/// it with the host's default calling convention, which is the C calling
/// convention on every supported platform (System V on Linux and macOS
/// x86-64, Windows x64, and AAPCS64 including Apple's variant).
type EntryPoint = unsafe extern "C" fn(*const i64, usize, *mut i64) -> i32;

/// Machine code for one program, together with the memory that holds it.
///
/// The function pointer is never handed out: the only way to run the code
/// is [`NativeCode::call`], which borrows `self`. The executable memory is
/// freed only when `self` is dropped, so the pointer can never dangle.
pub struct NativeCode {
    entry: EntryPoint,
    arg_count: usize,
    /// Owns the executable memory. It is only touched again in `Drop`. The
    /// `Mutex` is never locked; it exists because `JITModule` is `Send` but
    /// not `Sync`, and Python objects must be both.
    module: Mutex<Option<JITModule>>,
}

impl NativeCode {
    /// Validate `ops` and compile them for exactly `arg_count` arguments.
    pub fn compile(ops: &[Op], arg_count: usize) -> Result<Self, String> {
        program::validate(ops, arg_count)?;

        let isa = host_isa()?;
        let mut module = JITModule::new(JITBuilder::with_isa(isa, default_libcall_names()));

        let pointer = module.target_config().pointer_type();
        let mut signature = module.make_signature();
        signature.params.push(AbiParam::new(pointer)); // args
        signature.params.push(AbiParam::new(pointer)); // args_len
        signature.params.push(AbiParam::new(pointer)); // result
        signature.returns.push(AbiParam::new(types::I32)); // status

        let func_id = module
            .declare_function("speedysnek_i64", Linkage::Local, &signature)
            .map_err(|e| format!("cranelift: {e}"))?;

        let mut ctx = module.make_context();
        ctx.func.signature = signature;
        let mut builder_ctx = FunctionBuilderContext::new();
        let mut b = FunctionBuilder::new(&mut ctx.func, &mut builder_ctx);

        let entry = b.create_block();
        b.append_block_params_for_function_params(entry);
        b.switch_to_block(entry);
        b.seal_block(entry);
        let params = b.block_params(entry).to_vec();
        let (args, args_len, result) = (params[0], params[1], params[2]);

        // Shared failure exits.
        let bad_arity = b.create_block();
        let overflow = b.create_block();
        let overflow_index = b.append_block_param(overflow, types::I64);

        // Check the argument count before reading any argument.
        let body = b.create_block();
        let expected = b.ins().iconst(pointer, arg_count as i64);
        let arity_ok = b.ins().icmp(IntCC::Equal, args_len, expected);
        b.ins().brif(arity_ok, body, &[], bad_arity, &[]);
        b.switch_to_block(body);
        b.seal_block(body);

        // The stack machine exists only at compile time: each stack slot is
        // an SSA value, so the generated code keeps values in registers.
        let mut stack: Vec<Value> = Vec::new();
        for (index, op) in ops.iter().enumerate() {
            match *op {
                Op::LoadArg(arg) => {
                    // Validation guarantees arg < arg_count <= MAX_ARGS, so
                    // the offset fits in i32, and the arity check above
                    // guarantees the load is in bounds.
                    let offset = arg as i32 * 8;
                    stack.push(
                        b.ins()
                            .load(types::I64, MemFlagsData::trusted(), args, offset),
                    );
                }
                Op::LoadConst(value) => stack.push(b.ins().iconst(types::I64, value)),
                Op::Add | Op::Subtract | Op::Multiply => {
                    let right = pop(&mut stack)?;
                    let left = pop(&mut stack)?;
                    // Each instruction yields the (wrapped) value and an
                    // overflow flag. The wrapped value is only used on the
                    // path where the flag is clear, so it is exact there.
                    let (value, overflowed) = match *op {
                        Op::Add => b.ins().sadd_overflow(left, right),
                        Op::Subtract => b.ins().ssub_overflow(left, right),
                        _ => b.ins().smul_overflow(left, right),
                    };
                    let at = b.ins().iconst(types::I64, index as i64);
                    let next = b.create_block();
                    b.ins()
                        .brif(overflowed, overflow, &[BlockArg::Value(at)], next, &[]);
                    b.switch_to_block(next);
                    b.seal_block(next);
                    stack.push(value);
                }
                Op::Return => {
                    let value = pop(&mut stack)?;
                    b.ins().store(MemFlagsData::trusted(), value, result, 0);
                    let status = b.ins().iconst(types::I32, i64::from(STATUS_OK));
                    b.ins().return_(&[status]);
                }
            }
        }

        b.switch_to_block(overflow);
        b.seal_block(overflow);
        b.ins()
            .store(MemFlagsData::trusted(), overflow_index, result, 0);
        let status = b.ins().iconst(types::I32, i64::from(STATUS_OVERFLOW));
        b.ins().return_(&[status]);

        b.switch_to_block(bad_arity);
        b.seal_block(bad_arity);
        let status = b.ins().iconst(types::I32, i64::from(STATUS_BAD_ARITY));
        b.ins().return_(&[status]);

        b.finalize(module.target_config());

        // `define_function` runs Cranelift's IR verifier (on by default).
        module
            .define_function(func_id, &mut ctx)
            .map_err(|e| format!("cranelift: {e:?}"))?;
        module.clear_context(&mut ctx);
        module
            .finalize_definitions()
            .map_err(|e| format!("cranelift: {e}"))?;

        let code = module.get_finalized_function(func_id);
        // SAFETY: `code` is the start of the function just finalized, which
        // Cranelift built with exactly `EntryPoint`'s parameters and return
        // type in the host's default (C) calling convention. It stays
        // executable until `module.free_memory()`, which only `Drop` calls.
        let entry = unsafe { std::mem::transmute::<*const u8, EntryPoint>(code) };

        Ok(NativeCode {
            entry,
            arg_count,
            module: Mutex::new(Some(module)),
        })
    }

    pub fn arg_count(&self) -> usize {
        self.arg_count
    }

    /// Run the code. Returns the status and the value written to `result`
    /// (the return value, the overflowing instruction's index, or 0).
    pub fn call(&self, args: &[i64]) -> (i32, i64) {
        let mut result: i64 = 0;
        // SAFETY:
        // - `self.entry` points to code owned by `self.module`, which is only
        //   freed in `Drop`; borrowing `self` keeps it alive for this call.
        // - The pointer has the `EntryPoint` signature (see `compile`).
        // - `args.as_ptr()` is valid for `args.len()` reads, and the code
        //   reads nothing unless `args.len()` equals the compiled argument
        //   count, and then only indexes below it. It writes one aligned
        //   i64 to `result`, a live local.
        // - The code cannot trap or unwind: loads are in bounds, there is no
        //   division, overflow is a branch, and it calls no other function.
        let status = unsafe { (self.entry)(args.as_ptr(), args.len(), &mut result) };
        (status, result)
    }
}

impl Drop for NativeCode {
    fn drop(&mut self) {
        let module = match self.module.get_mut() {
            Ok(module) => module.take(),
            Err(poisoned) => poisoned.into_inner().take(),
        };
        if let Some(module) = module {
            // SAFETY: `self.entry` is the only pointer into this module and
            // is never exposed. `drop` has exclusive access to `self`, so no
            // `call` is running and none can start afterwards.
            unsafe { module.free_memory() };
        }
    }
}

fn pop(stack: &mut Vec<Value>) -> Result<Value, String> {
    // Unreachable after validation, but an error is better than a panic.
    stack.pop().ok_or_else(|| "stack underflow".to_owned())
}

/// The host target, created once and shared by every compilation.
fn host_isa() -> Result<OwnedTargetIsa, String> {
    static ISA: OnceLock<Result<OwnedTargetIsa, String>> = OnceLock::new();
    ISA.get_or_init(|| {
        let mut flags = settings::builder();
        for (name, value) in [
            ("opt_level", "speed"),
            ("is_pic", "false"),
            ("use_colocated_libcalls", "false"),
        ] {
            flags
                .set(name, value)
                .map_err(|e| format!("cranelift flag {name}: {e}"))?;
        }
        cranelift_native::builder()
            .map_err(|e| format!("unsupported host: {e}"))?
            .finish(settings::Flags::new(flags))
            .map_err(|e| format!("cranelift: {e}"))
    })
    .clone()
}
