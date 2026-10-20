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
//! - `STATUS_BUDGET_EXHAUSTED`: a backward jump found the call's backedge
//!   budget used up; `*result` holds the index of that jump. Only programs
//!   with a loop can return it.
//!
//! Failures are ordinary branches to a block that returns a status code.
//! The generated code never traps, never unwinds and never calls anything.
//!
//! The operand stack exists only at compile time: each entry is an SSA
//! value. Validation guarantees that it is empty at every jump and jump
//! target, so no stack value ever crosses from one Cranelift block to
//! another.
//!
//! Each local slot is a Cranelift `Variable`. A `STORE_LOCAL` defines the
//! variable and a `LOAD_LOCAL` uses it; within a block that is just the
//! most recently stored SSA value, so an assigned expression is computed
//! once and an assignment copies nothing at run time. Where paths join,
//! `cranelift_frontend`'s SSA construction gives the join block a parameter
//! for each slot whose value differs between the incoming paths, and every
//! jump into it passes that path's value: a path that did not store a slot
//! passes the value the slot held before the branch. SSA values never
//! change, so a later store to one slot cannot affect a value already read
//! from it or copied to another. Validation proves that every slot read is
//! stored on every path to the read, so the SSA construction never has to
//! invent an initial value.
//!
//! Every IR instruction that starts a basic block (index 0, a jump target,
//! or the instruction after a `JUMP_IF_FALSE`) gets its own Cranelift
//! block. A block is sealed once all of its predecessors have been
//! emitted: for a block reached only by forward edges that is as soon as
//! code generation reaches it, and for a loop header, which backward jumps
//! also reach, after the whole program. Until then `use_var` gives the
//! header a provisional parameter per local it reads; sealing fills each in
//! from the entry edge and every backedge, which is how the header sees the
//! values of the previous iteration. A comparison becomes a signed `icmp`,
//! whose result feeds only the `brif` of the `JUMP_IF_FALSE` after it. Only
//! the selected branch runs: every `ADD`, `SUBTRACT` and `MULTIPLY` keeps
//! its own overflow branch in the block it belongs to, even when its result
//! is later overwritten or never read, and an operation on a path not taken
//! never runs.
//!
//! A program with a backward jump also gets a backedge counter, another
//! `Variable`, set to the budget on entry. Every backward jump returns
//! `STATUS_BUDGET_EXHAUSTED` if the counter is zero, and otherwise
//! decrements it before jumping. The check is on every backward jump the
//! generator emits, whatever validation decided, and forward edges cannot
//! form a cycle, so no program can run more than `budget` backedges in one
//! call. Programs without one get no counter and no checks.

use std::sync::{Mutex, OnceLock};

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{AbiParam, Block, BlockArg, InstBuilder, MemFlagsData, Value, types};
use cranelift_codegen::isa::OwnedTargetIsa;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module, default_libcall_names};

use crate::program::{self, Comparison, Op};

pub const STATUS_OK: i32 = 0;
pub const STATUS_OVERFLOW: i32 = 1;
pub const STATUS_BAD_ARITY: i32 = 2;
pub const STATUS_BUDGET_EXHAUSTED: i32 = 3;

/// Backward jumps a call may take before it returns
/// `STATUS_BUDGET_EXHAUSTED`. A small loop body takes one or two
/// nanoseconds per iteration, so an exhausted call holds the GIL for a
/// millisecond or two (see `docs/loops.md`).
pub const DEFAULT_BACKEDGE_BUDGET: u64 = 1_000_000;

/// The largest budget [`NativeCode::compile`] accepts. It is the default:
/// a budget can be lowered (tests do), never raised.
pub const MAX_BACKEDGE_BUDGET: u64 = DEFAULT_BACKEDGE_BUDGET;

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
    /// The backedge budget, if the program has a loop.
    backedge_budget: Option<u64>,
    /// Owns the executable memory. It is only touched again in `Drop`. The
    /// `Mutex` is never locked; it exists because `JITModule` is `Send` but
    /// not `Sync`, and Python objects must be both.
    module: Mutex<Option<JITModule>>,
}

impl NativeCode {
    /// Validate `ops` and compile them for exactly `arg_count` arguments.
    /// A loop may take at most `backedge_budget` backward jumps per call.
    pub fn compile(ops: &[Op], arg_count: usize, backedge_budget: u64) -> Result<Self, String> {
        if backedge_budget > MAX_BACKEDGE_BUDGET {
            return Err(format!(
                "backedge budget {backedge_budget} exceeds the maximum of {MAX_BACKEDGE_BUDGET}"
            ));
        }
        let (local_count, _) = program::validate(ops, arg_count)?;

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
        // Only a program with a backward jump can use up its budget.
        let has_loop = ops
            .iter()
            .enumerate()
            .any(|(index, op)| matches!(*op, Op::Jump(target) if target as usize <= index));
        let exhausted = has_loop.then(|| {
            let block = b.create_block();
            let index = b.append_block_param(block, types::I64);
            (block, index)
        });

        // Check the argument count before reading any argument.
        let body = b.create_block();
        let expected = b.ins().iconst(pointer, arg_count as i64);
        let arity_ok = b.ins().icmp(IntCC::Equal, args_len, expected);
        b.ins().brif(arity_ok, body, &[], bad_arity, &[]);
        b.switch_to_block(body);
        b.seal_block(body);

        // A block for every instruction that starts one, other than the
        // first, which is `body`. Validation guarantees every target is in
        // range, and that only an unconditional jump goes backward.
        let mut blocks: Vec<Option<Block>> = vec![None; ops.len()];
        // Blocks that a backward jump reaches, sealed only at the end.
        let mut backedge_target = vec![false; ops.len()];
        for (index, op) in ops.iter().enumerate() {
            if let Op::Jump(target) | Op::JumpIfFalse(target) = *op {
                let target = target as usize;
                if target >= ops.len() || (target <= index && matches!(op, Op::JumpIfFalse(_))) {
                    return Err(format!("invalid jump target {target} at index {index}"));
                }
                blocks[target].get_or_insert_with(|| b.create_block());
                backedge_target[target] |= target <= index;
                if let Op::JumpIfFalse(_) = op {
                    blocks[index + 1].get_or_insert_with(|| b.create_block());
                }
            }
        }

        // One variable per local slot. Validation proves each is defined on
        // every path before it is used.
        let locals: Vec<Variable> = (0..local_count)
            .map(|_| b.declare_var(types::I64))
            .collect();

        // The backedge counter, only for a program with a backward jump.
        let budget = has_loop.then(|| {
            let budget = b.declare_var(types::I64);
            let initial = b.ins().iconst(types::I64, backedge_budget as i64);
            b.def_var(budget, initial);
            budget
        });

        // The stack machine exists only at compile time: each stack slot is
        // an SSA value, so the generated code keeps values in registers.
        let mut stack: Vec<StackValue> = Vec::new();
        // Whether the current block already ends with a jump or return.
        let mut terminated = false;
        for (index, op) in ops.iter().enumerate() {
            if let Some(block) = blocks[index] {
                if !terminated {
                    // Fall through. Validation guarantees the stack is empty.
                    b.ins().jump(block, &[]);
                }
                b.switch_to_block(block);
                if !backedge_target[index] {
                    // Every predecessor is at a lower index and already
                    // emitted.
                    b.seal_block(block);
                }
                terminated = false;
            } else if terminated {
                // Unreachable after validation.
                return Err(format!("instruction {index} is unreachable"));
            }
            match *op {
                Op::LoadArg(arg) => {
                    // Validation guarantees arg < arg_count <= MAX_ARGS, so
                    // the offset fits in i32, and the arity check above
                    // guarantees the load is in bounds.
                    let offset = arg as i32 * 8;
                    stack.push(StackValue::Int(b.ins().load(
                        types::I64,
                        MemFlagsData::trusted(),
                        args,
                        offset,
                    )));
                }
                Op::LoadConst(value) => {
                    stack.push(StackValue::Int(b.ins().iconst(types::I64, value)))
                }
                Op::LoadLocal(slot) => {
                    let var = local(&locals, slot)?;
                    stack.push(StackValue::Int(b.use_var(var)));
                }
                Op::StoreLocal(slot) => {
                    let value = pop_int(&mut stack)?;
                    b.def_var(local(&locals, slot)?, value);
                }
                Op::Add | Op::Subtract | Op::Multiply => {
                    let right = pop_int(&mut stack)?;
                    let left = pop_int(&mut stack)?;
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
                    stack.push(StackValue::Int(value));
                }
                Op::Compare(comparison) => {
                    let right = pop_int(&mut stack)?;
                    let left = pop_int(&mut stack)?;
                    // A direct signed comparison: never a subtraction, which
                    // could overflow.
                    let cc = match comparison {
                        Comparison::LessThan => IntCC::SignedLessThan,
                        Comparison::LessEqual => IntCC::SignedLessThanOrEqual,
                        Comparison::GreaterThan => IntCC::SignedGreaterThan,
                        Comparison::GreaterEqual => IntCC::SignedGreaterThanOrEqual,
                        Comparison::Equal => IntCC::Equal,
                        Comparison::NotEqual => IntCC::NotEqual,
                    };
                    stack.push(StackValue::Condition(b.ins().icmp(cc, left, right)));
                }
                Op::Jump(target) => {
                    let block = block_at(&blocks, target)?;
                    if target as usize <= index {
                        // A backedge: spend one unit of the budget, or stop.
                        let (budget, (exhausted, _)) = budget
                            .zip(exhausted)
                            .ok_or("backward jump without a budget")?;
                        let remaining = b.use_var(budget);
                        let used_up = b.ins().icmp_imm_s(IntCC::Equal, remaining, 0);
                        let at = b.ins().iconst(types::I64, index as i64);
                        let next = b.create_block();
                        b.ins()
                            .brif(used_up, exhausted, &[BlockArg::Value(at)], next, &[]);
                        b.switch_to_block(next);
                        b.seal_block(next);
                        let remaining = b.ins().iadd_imm_s(remaining, -1);
                        b.def_var(budget, remaining);
                    }
                    b.ins().jump(block, &[]);
                    terminated = true;
                }
                Op::JumpIfFalse(target) => {
                    let condition = match stack.pop() {
                        Some(StackValue::Condition(value)) => value,
                        _ => return Err(format!("no condition for the jump at {index}")),
                    };
                    let taken = block_at(&blocks, target)?;
                    let next = block_at(&blocks, index as u32 + 1)?;
                    if taken == next {
                        // Both outcomes continue at the same instruction.
                        b.ins().jump(next, &[]);
                    } else {
                        b.ins().brif(condition, next, &[], taken, &[]);
                    }
                    terminated = true;
                }
                Op::Return => {
                    let value = pop_int(&mut stack)?;
                    b.ins().store(MemFlagsData::trusted(), value, result, 0);
                    let status = b.ins().iconst(types::I32, i64::from(STATUS_OK));
                    b.ins().return_(&[status]);
                    terminated = true;
                }
            }
        }
        if !terminated {
            // Unreachable after validation.
            return Err("program falls off its end".to_owned());
        }
        // Every backedge has been emitted now.
        for (index, block) in blocks.iter().enumerate() {
            if let (Some(block), true) = (block, backedge_target[index]) {
                b.seal_block(*block);
            }
        }

        b.switch_to_block(overflow);
        b.seal_block(overflow);
        b.ins()
            .store(MemFlagsData::trusted(), overflow_index, result, 0);
        let status = b.ins().iconst(types::I32, i64::from(STATUS_OVERFLOW));
        b.ins().return_(&[status]);

        if let Some((exhausted, exhausted_index)) = exhausted {
            b.switch_to_block(exhausted);
            b.seal_block(exhausted);
            b.ins()
                .store(MemFlagsData::trusted(), exhausted_index, result, 0);
            let status = b
                .ins()
                .iconst(types::I32, i64::from(STATUS_BUDGET_EXHAUSTED));
            b.ins().return_(&[status]);
        }

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
            backedge_budget: budget.map(|_| backedge_budget),
            module: Mutex::new(Some(module)),
        })
    }

    pub fn arg_count(&self) -> usize {
        self.arg_count
    }

    /// The backedge budget of a program with a loop, else `None`.
    pub fn backedge_budget(&self) -> Option<u64> {
        self.backedge_budget
    }

    /// Run the code. Returns the status and the value written to `result`
    /// (the return value, the index of the overflowing instruction or of the
    /// backward jump that found the budget used up, or 0).
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
        // - It terminates: every cycle passes a backward jump, and every
        //   backward jump spends the finite backedge budget.
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

/// A value on the compile-time operand stack.
#[derive(Clone, Copy)]
enum StackValue {
    /// An i64.
    Int(Value),
    /// The i8 result of an `icmp`, consumed only by the following branch.
    Condition(Value),
}

fn pop_int(stack: &mut Vec<StackValue>) -> Result<Value, String> {
    // Unreachable after validation, but an error is better than a panic.
    match stack.pop() {
        Some(StackValue::Int(value)) => Ok(value),
        Some(StackValue::Condition(_)) => Err("expected an integer, found a condition".to_owned()),
        None => Err("stack underflow".to_owned()),
    }
}

fn local(locals: &[Variable], slot: u32) -> Result<Variable, String> {
    // Unreachable after validation.
    locals
        .get(slot as usize)
        .copied()
        .ok_or_else(|| format!("local {slot} out of range"))
}

fn block_at(blocks: &[Option<Block>], index: u32) -> Result<Block, String> {
    // Unreachable after validation.
    blocks
        .get(index as usize)
        .copied()
        .flatten()
        .ok_or_else(|| format!("no block starts at instruction {index}"))
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
