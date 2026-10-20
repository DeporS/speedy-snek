//! The serialized program format shared with `speedysnek._compiler`.
//!
//! Python sends a program as a tuple of small tuples of plain integers:
//! `(opcode,)` for operations without an operand and `(opcode, operand)`
//! for `LOAD_ARG`, `LOAD_CONST`, `LOAD_LOCAL`, `STORE_LOCAL`, `JUMP` and
//! `JUMP_IF_FALSE`. The opcode numbers below are the whole contract; Rust
//! never looks at Python's `Instruction` or `Op` types.
//!
//! Local slots are separate from arguments and start out unassigned. A
//! program never says how many slots it uses: validation proves that every
//! `LOAD_LOCAL` reads a slot that a `STORE_LOCAL` wrote on every path to
//! it, and counts the slots from the indexes it sees.
//!
//! A jump's operand is the index of the instruction it continues at, which
//! must lie ahead of the jump: control flow is a directed acyclic graph
//! whose edges all point forward. Comparisons push a *condition*, which
//! only the `JUMP_IF_FALSE` right after them may pop. The operand stack is
//! empty at every jump and jump target, so paths join only through local
//! slots, and every path ends with its own `RETURN`.
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
/// Push the value last stored in local slot `operand`.
pub const OP_LOAD_LOCAL: i64 = 7;
/// Pop a value and store it in local slot `operand`.
pub const OP_STORE_LOCAL: i64 = 8;
/// Continue at instruction index `operand`.
pub const OP_JUMP: i64 = 9;
/// Pop a condition; if it is false, continue at instruction index `operand`.
pub const OP_JUMP_IF_FALSE: i64 = 10;
/// Pop `right`, pop `left`, push the condition `left < right` (signed).
pub const OP_LESS_THAN: i64 = 11;
/// Pop `right`, pop `left`, push the condition `left <= right` (signed).
pub const OP_LESS_EQUAL: i64 = 12;
/// Pop `right`, pop `left`, push the condition `left > right` (signed).
pub const OP_GREATER_THAN: i64 = 13;
/// Pop `right`, pop `left`, push the condition `left >= right` (signed).
pub const OP_GREATER_EQUAL: i64 = 14;
/// Pop `right`, pop `left`, push the condition `left == right`.
pub const OP_EQUAL: i64 = 15;
/// Pop `right`, pop `left`, push the condition `left != right`.
pub const OP_NOT_EQUAL: i64 = 16;

/// The largest supported number of arguments. It keeps every argument's
/// byte offset (`index * 8`) well inside the `i32` offsets Cranelift uses.
pub const MAX_ARGS: usize = 1 << 16;

/// The largest supported number of local slots. It bounds the compiler's
/// per-slot bookkeeping; slots never live in memory.
pub const MAX_LOCALS: usize = 1 << 16;

/// The largest supported number of `JUMP` and `JUMP_IF_FALSE` instructions.
/// Validation keeps one set of assigned locals per pending jump target, so
/// this bounds its memory to `MAX_JUMPS * MAX_LOCALS` bits (32 MiB).
pub const MAX_JUMPS: usize = 1 << 12;

/// A signed comparison of two integers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Comparison {
    LessThan,
    LessEqual,
    GreaterThan,
    GreaterEqual,
    Equal,
    NotEqual,
}

/// One decoded instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    LoadArg(u32),
    LoadConst(i64),
    LoadLocal(u32),
    StoreLocal(u32),
    Add,
    Subtract,
    Multiply,
    Return,
    Compare(Comparison),
    Jump(u32),
    JumpIfFalse(u32),
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
            (OP_LOAD_LOCAL | OP_STORE_LOCAL, Some(slot)) => {
                let slot = u32::try_from(slot).map_err(|_| {
                    format!("opcode {opcode} at index {index} has invalid local index {slot}")
                })?;
                if opcode == OP_LOAD_LOCAL {
                    Op::LoadLocal(slot)
                } else {
                    Op::StoreLocal(slot)
                }
            }
            (OP_JUMP | OP_JUMP_IF_FALSE, Some(target)) => {
                let target = u32::try_from(target).map_err(|_| {
                    format!("opcode {opcode} at index {index} has invalid target {target}")
                })?;
                if opcode == OP_JUMP {
                    Op::Jump(target)
                } else {
                    Op::JumpIfFalse(target)
                }
            }
            (OP_LESS_THAN, None) => Op::Compare(Comparison::LessThan),
            (OP_LESS_EQUAL, None) => Op::Compare(Comparison::LessEqual),
            (OP_GREATER_THAN, None) => Op::Compare(Comparison::GreaterThan),
            (OP_GREATER_EQUAL, None) => Op::Compare(Comparison::GreaterEqual),
            (OP_EQUAL, None) => Op::Compare(Comparison::Equal),
            (OP_NOT_EQUAL, None) => Op::Compare(Comparison::NotEqual),
            (OP_ADD, None) => Op::Add,
            (OP_SUBTRACT, None) => Op::Subtract,
            (OP_MULTIPLY, None) => Op::Multiply,
            (OP_RETURN, None) => Op::Return,
            (
                OP_LOAD_ARG | OP_LOAD_CONST | OP_LOAD_LOCAL | OP_STORE_LOCAL | OP_JUMP
                | OP_JUMP_IF_FALSE,
                None,
            ) => {
                return Err(format!(
                    "opcode {opcode} at index {index} requires an operand"
                ));
            }
            (
                OP_ADD | OP_SUBTRACT | OP_MULTIPLY | OP_RETURN | OP_LESS_THAN | OP_LESS_EQUAL
                | OP_GREATER_THAN | OP_GREATER_EQUAL | OP_EQUAL | OP_NOT_EQUAL,
                Some(_),
            ) => {
                return Err(format!("opcode {opcode} at index {index} takes no operand"));
            }
            _ => return Err(format!("unknown opcode {opcode} at index {index}")),
        };
        Ok(op)
    }
}

/// The kind of a value on the operand stack.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Int,
    Condition,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Int => "an integer",
            Kind::Condition => "a condition",
        }
    }
}

/// A set of local slots, one bit each.
#[derive(Clone, Debug, PartialEq, Eq)]
struct SlotSet(Vec<u64>);

impl SlotSet {
    fn empty(slots: usize) -> SlotSet {
        SlotSet(vec![0; slots.div_ceil(64)])
    }

    fn contains(&self, slot: usize) -> bool {
        self.0[slot / 64] & (1 << (slot % 64)) != 0
    }

    fn insert(&mut self, slot: usize) {
        self.0[slot / 64] |= 1 << (slot % 64);
    }

    fn intersect(&mut self, other: &SlotSet) {
        for (word, other) in self.0.iter_mut().zip(&other.0) {
            *word &= other;
        }
    }
}

/// Check that `ops` is a well-formed program for `arg_count` arguments.
///
/// The rules match `speedysnek._interpreter.validate`, checked here
/// independently:
///
/// - the program is not empty and contains a `RETURN`;
/// - every jump targets an instruction index ahead of itself, inside the
///   program, so there are no cycles;
/// - every instruction is reachable from the first;
/// - nothing pops from an empty stack, and every operation finds the kind
///   of value it needs: only `JUMP_IF_FALSE` pops a condition;
/// - every comparison is immediately followed by `JUMP_IF_FALSE`;
/// - the stack is empty after every jump and on every path into a jump
///   target;
/// - every `LOAD_LOCAL` reads a slot stored on every path to it;
/// - every path ends with a `RETURN` that finds exactly one value.
///
/// In addition every `LOAD_ARG` index must be below `arg_count`, so
/// generated code never reads past the arguments, every local index below
/// `MAX_LOCALS`, and there are at most `MAX_JUMPS` jumps.
///
/// Returns the number of local slots: one more than the largest index
/// used, or 0.
pub fn validate(ops: &[Op], arg_count: usize) -> Result<usize, String> {
    if arg_count > MAX_ARGS {
        return Err(format!(
            "arg_count {arg_count} exceeds the maximum of {MAX_ARGS}"
        ));
    }
    if ops.is_empty() {
        return Err("program is empty".to_owned());
    }
    if !ops.contains(&Op::Return) {
        return Err("program has no RETURN".to_owned());
    }

    // Static checks first, so that the slot sets below have a known size
    // and every jump target is known to be in range.
    let mut local_count = 0;
    let mut jumps = 0;
    for (index, op) in ops.iter().enumerate() {
        match *op {
            Op::LoadArg(arg) => {
                if arg as usize >= arg_count {
                    return Err(format!(
                        "LOAD_ARG at index {index} reads argument {arg}, \
                         but there are only {arg_count}"
                    ));
                }
            }
            Op::LoadLocal(slot) => {
                check_local(index, "LOAD_LOCAL", slot)?;
            }
            Op::StoreLocal(slot) => {
                local_count = local_count.max(check_local(index, "STORE_LOCAL", slot)? + 1);
            }
            Op::Jump(target) | Op::JumpIfFalse(target) => {
                jumps += 1;
                if jumps > MAX_JUMPS {
                    return Err(format!(
                        "program has more than the maximum of {MAX_JUMPS} jumps"
                    ));
                }
                let target = target as usize;
                if target <= index {
                    return Err(format!(
                        "jump at index {index} goes back to index {target}; \
                         only forward jumps are allowed"
                    ));
                }
                if target >= ops.len() {
                    return Err(format!(
                        "jump at index {index} goes to index {target}, \
                         past the end of the program"
                    ));
                }
            }
            _ => {}
        }
    }

    // One pass in index order sees every predecessor of an instruction
    // before the instruction itself, because every edge points forward.
    // `assigned` is `None` where the previous instruction does not fall
    // through. `joins[target]` holds the slots stored on every jump into a
    // target not reached yet; the stack is always empty there.
    let mut stack: Vec<Kind> = Vec::new();
    let mut assigned: Option<SlotSet> = Some(SlotSet::empty(local_count));
    let mut joins: Vec<Option<SlotSet>> = vec![None; ops.len()];
    for (index, op) in ops.iter().enumerate() {
        if let Some(jumped) = joins[index].take() {
            match &mut assigned {
                None => assigned = Some(jumped),
                Some(current) => {
                    if !stack.is_empty() {
                        return Err(format!(
                            "jump target at index {index} is reached from index {} \
                             with {} value(s) on the stack; the stack must be empty \
                             where paths join",
                            index - 1,
                            stack.len()
                        ));
                    }
                    current.intersect(&jumped);
                }
            }
        }
        let Some(current) = &mut assigned else {
            return Err(format!("instruction at index {index} is unreachable"));
        };

        let (pops, push): (&[Kind], Option<Kind>) = match op {
            Op::LoadArg(_) | Op::LoadConst(_) | Op::LoadLocal(_) => (&[], Some(Kind::Int)),
            Op::StoreLocal(_) | Op::Return => (&[Kind::Int], None),
            Op::Add | Op::Subtract | Op::Multiply => (&[Kind::Int, Kind::Int], Some(Kind::Int)),
            Op::Compare(_) => (&[Kind::Int, Kind::Int], Some(Kind::Condition)),
            Op::Jump(_) => (&[], None),
            Op::JumpIfFalse(_) => (&[Kind::Condition], None),
        };
        if stack.len() < pops.len() {
            return Err(format!(
                "stack underflow at index {index}: needs {} value(s), stack has {}",
                pops.len(),
                stack.len()
            ));
        }
        let operands = stack.split_off(stack.len() - pops.len());
        for (needed, found) in pops.iter().zip(&operands) {
            if needed != found {
                return Err(format!(
                    "instruction at index {index} needs {} but found {}",
                    needed.name(),
                    found.name()
                ));
            }
        }
        stack.extend(push);

        match *op {
            Op::LoadLocal(slot) => {
                let slot = slot as usize;
                if slot >= local_count || !current.contains(slot) {
                    return Err(format!(
                        "LOAD_LOCAL at index {index} reads local {slot} before it is \
                         assigned on every path"
                    ));
                }
            }
            Op::StoreLocal(slot) => current.insert(slot as usize),
            Op::Compare(_) => {
                if !matches!(ops.get(index + 1), Some(Op::JumpIfFalse(_))) {
                    return Err(format!(
                        "comparison at index {index} must be immediately followed \
                         by JUMP_IF_FALSE"
                    ));
                }
            }
            Op::Return | Op::Jump(_) | Op::JumpIfFalse(_) => {
                if !stack.is_empty() {
                    return Err(format!(
                        "instruction at index {index} leaves {} value(s) on the stack, \
                         expected none",
                        stack.len()
                    ));
                }
                if let Op::Jump(target) | Op::JumpIfFalse(target) = *op {
                    let join = &mut joins[target as usize];
                    match join {
                        None => *join = Some(current.clone()),
                        Some(join) => join.intersect(current),
                    }
                }
                if !matches!(op, Op::JumpIfFalse(_)) {
                    assigned = None;
                }
            }
            _ => {}
        }
    }

    if assigned.is_some() {
        return Err(format!(
            "the instruction at index {} falls off the end of the program; \
             every path must end with RETURN",
            ops.len() - 1
        ));
    }
    Ok(local_count)
}

fn check_local(index: usize, name: &str, slot: u32) -> Result<usize, String> {
    let slot = slot as usize;
    if slot >= MAX_LOCALS {
        return Err(format!(
            "{name} at index {index} uses local {slot}, \
             but the maximum is {}",
            MAX_LOCALS - 1
        ));
    }
    Ok(slot)
}
