//! The serialized program format shared with `speedysnek._compiler`.
//!
//! Python sends a program as a tuple of small tuples of plain integers:
//! `(opcode,)` for operations without an operand and `(opcode, operand)`
//! for `LOAD_ARG` and `LOAD_CONST`. The opcode numbers below are the whole
//! contract; Rust never looks at Python's `Instruction` or `Op` types.
//!
//! Python validates programs before sending them, but this module checks
//! everything again: a malformed program must become a Python exception,
//! never a panic or bad machine code.

/// Push argument `operand`.
pub const OP_LOAD_ARG: i64 = 1;
/// Push the constant `operand`.
pub const OP_LOAD_CONST: i64 = 2;
/// Pop `right`, pop `left`, push `left + right`.
pub const OP_ADD: i64 = 3;
/// Pop `right`, pop `left`, push `left - right`.
pub const OP_SUBTRACT: i64 = 4;
/// Pop `right`, pop `left`, push `left * right`.
pub const OP_MULTIPLY: i64 = 5;
/// Pop the single remaining value and return it.
pub const OP_RETURN: i64 = 6;

/// The largest supported number of arguments. It keeps every argument's
/// byte offset (`index * 8`) well inside the `i32` offsets Cranelift uses.
pub const MAX_ARGS: usize = 1 << 16;

/// One decoded instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    LoadArg(u32),
    LoadConst(i64),
    Add,
    Subtract,
    Multiply,
    Return,
}

impl Op {
    /// Decode one `(opcode, operand?)` pair. `index` is only used in errors.
    pub fn decode(index: usize, opcode: i64, operand: Option<i64>) -> Result<Op, String> {
        let op = match (opcode, operand) {
            (OP_LOAD_ARG, Some(arg)) => {
                let arg = u32::try_from(arg).map_err(|_| {
                    format!("LOAD_ARG at index {index} has invalid argument index {arg}")
                })?;
                Op::LoadArg(arg)
            }
            (OP_LOAD_CONST, Some(value)) => Op::LoadConst(value),
            (OP_ADD, None) => Op::Add,
            (OP_SUBTRACT, None) => Op::Subtract,
            (OP_MULTIPLY, None) => Op::Multiply,
            (OP_RETURN, None) => Op::Return,
            (OP_LOAD_ARG | OP_LOAD_CONST, None) => {
                return Err(format!(
                    "opcode {opcode} at index {index} requires an operand"
                ));
            }
            (OP_ADD | OP_SUBTRACT | OP_MULTIPLY | OP_RETURN, Some(_)) => {
                return Err(format!("opcode {opcode} at index {index} takes no operand"));
            }
            _ => return Err(format!("unknown opcode {opcode} at index {index}")),
        };
        Ok(op)
    }
}

/// Check that `ops` is a well-formed program for `arg_count` arguments.
///
/// The rules match `speedysnek._interpreter.validate`: the program is not
/// empty, never pops from an empty stack and ends with its only `RETURN`,
/// which finds exactly one value. In addition every `LOAD_ARG` index must
/// be below `arg_count`, so generated code never reads past the arguments.
pub fn validate(ops: &[Op], arg_count: usize) -> Result<(), String> {
    if arg_count > MAX_ARGS {
        return Err(format!(
            "arg_count {arg_count} exceeds the maximum of {MAX_ARGS}"
        ));
    }
    if ops.is_empty() {
        return Err("program is empty".to_owned());
    }

    let last = ops.len() - 1;
    let mut depth: usize = 0;
    for (index, op) in ops.iter().enumerate() {
        let (pops, pushes) = match op {
            Op::LoadArg(arg) => {
                if *arg as usize >= arg_count {
                    return Err(format!(
                        "LOAD_ARG at index {index} reads argument {arg}, \
                         but there are only {arg_count}"
                    ));
                }
                (0, 1)
            }
            Op::LoadConst(_) => (0, 1),
            Op::Add | Op::Subtract | Op::Multiply => (2, 1),
            Op::Return => (1, 0),
        };
        if depth < pops {
            return Err(format!(
                "stack underflow at index {index}: needs {pops} value(s), stack has {depth}"
            ));
        }
        if *op == Op::Return {
            if index != last {
                return Err(format!(
                    "RETURN at index {index} is not the final instruction"
                ));
            }
            if depth != 1 {
                return Err(format!(
                    "RETURN at index {index} finds {depth} values on the stack, expected 1"
                ));
            }
        }
        depth = depth - pops + pushes;
    }

    if ops[last] != Op::Return {
        return Err("program does not end with RETURN".to_owned());
    }
    Ok(())
}
