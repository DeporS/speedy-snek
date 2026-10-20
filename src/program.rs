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
//! A jump's operand is the index of the instruction it continues at. Every
//! edge points forward except the backedges of at most one `while` loop:
//! unconditional `JUMP`s back to the loop's header, under the shape rules
//! of [`validate`] (see `docs/loops.md`). Comparisons push a *condition*,
//! which only the `JUMP_IF_FALSE` right after them may pop. The operand
//! stack is empty at every jump and jump target, so paths join only
//! through local slots, and every path ends with its own `RETURN`.
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

    fn is_subset(&self, other: &SlotSet) -> bool {
        self.0
            .iter()
            .zip(&other.0)
            .all(|(word, other)| word & !other == 0)
    }
}

/// The one loop a program may contain (see [`validate`]).
///
/// The loop occupies the indexes `header..exit`. Its test runs from
/// `header` to the `JUMP_IF_FALSE` at `test`, which jumps to `exit`; the
/// body runs from `test + 1` to `exit - 1` and jumps back to `header`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Loop {
    pub header: usize,
    pub test: usize,
    pub exit: usize,
}

/// Find the program's loop and check its shape.
///
/// Every jump target must already be known to be inside the program.
/// Returns `None` for a program without backward jumps.
fn find_loop(ops: &[Op]) -> Result<Option<Loop>, String> {
    // Every backward jump is an unconditional JUMP, and all of them go to
    // the same header.
    let mut header: Option<(usize, usize)> = None; // (header, first backedge)
    let mut last_backedge = 0;
    for (index, op) in ops.iter().enumerate() {
        match *op {
            Op::JumpIfFalse(target) if target as usize <= index => {
                return Err(format!(
                    "JUMP_IF_FALSE at index {index} goes back to index {target}; \
                     only an unconditional JUMP can be a loop's backward jump"
                ));
            }
            Op::Jump(target) if target as usize <= index => {
                let target = target as usize;
                match header {
                    None => header = Some((target, index)),
                    Some((first_target, first)) if first_target != target => {
                        return Err(format!(
                            "the backward jumps at index {first} and {index} go to \
                             different targets ({first_target} and {target}); at most \
                             one loop is supported"
                        ));
                    }
                    Some(_) => {}
                }
                last_backedge = index;
            }
            _ => {}
        }
    }
    let Some((header, _)) = header else {
        return Ok(None);
    };

    // The header is a straight-line test that ends with the loop's exit.
    let (test, exit) = match ops[header..]
        .iter()
        .enumerate()
        .find(|(_, op)| matches!(op, Op::Jump(_) | Op::JumpIfFalse(_) | Op::Return))
    {
        Some((offset, Op::JumpIfFalse(target))) => (header + offset, *target as usize),
        found => {
            let at = found.map_or(header, |(offset, _)| header + offset);
            return Err(format!(
                "the loop at index {header} must start with its condition: the \
                 first jump or RETURN from there, at index {at}, must be a \
                 JUMP_IF_FALSE that leaves the loop"
            ));
        }
    };
    if exit <= last_backedge {
        return Err(format!(
            "the loop condition's JUMP_IF_FALSE at index {test} goes to index {exit}, \
             which is not after the loop's last backward jump at index {last_backedge}"
        ));
    }

    // Entered only at the header, left only by the test.
    let inside = |index: usize| header <= index && index < exit;
    for (index, op) in ops.iter().enumerate() {
        match *op {
            Op::Return if inside(index) => {
                return Err(format!(
                    "RETURN at index {index} is inside the loop at indexes \
                     {header}..{exit}; a loop can only be left by its condition"
                ));
            }
            Op::Jump(target) | Op::JumpIfFalse(target) => {
                let target = target as usize;
                if index == test || (target == header && index > test && index < exit) {
                    continue;
                }
                if inside(index) && !(test < target && target < exit) {
                    return Err(format!(
                        "the jump at index {index} leaves the loop body at indexes \
                         {}..{exit} for index {target}; a loop can only be left by \
                         its condition",
                        test + 1
                    ));
                }
                if !inside(index) && header < target && target < exit {
                    return Err(format!(
                        "the jump at index {index} enters the loop at index {target}; \
                         a loop can only be entered at its header, index {header}"
                    ));
                }
            }
            _ => {}
        }
    }
    if !matches!(ops[exit - 1], Op::Jump(target) if target as usize == header) {
        return Err(format!(
            "the loop body falls through from index {} into the loop's exit at \
             index {exit}; it must end with a backward JUMP",
            exit - 1
        ));
    }
    Ok(Some(Loop { header, test, exit }))
}

/// Check that `ops` is a well-formed program for `arg_count` arguments.
///
/// The rules match `speedysnek._interpreter.validate`, checked here
/// independently:
///
/// - the program is not empty and contains a `RETURN`;
/// - every jump targets an index inside the program, and every jump goes
///   forward, except the backward `JUMP`s of at most one loop (below);
/// - every instruction is reachable from the first;
/// - nothing pops from an empty stack, and every operation finds the kind
///   of value it needs: only `JUMP_IF_FALSE` pops a condition;
/// - every comparison is immediately followed by `JUMP_IF_FALSE`;
/// - the stack is empty after every jump and on every path into a jump
///   target, the loop header included;
/// - every `LOAD_LOCAL` reads a slot stored on every path to it;
/// - every path ends with a `RETURN` that finds exactly one value.
///
/// A loop has a header at index `H` and an exit at index `X`:
///
/// - every backward jump is an unconditional `JUMP` to `H`;
/// - the first jump or `RETURN` at or after `H` is a `JUMP_IF_FALSE` to
///   `X`, which lies after every backward jump;
/// - no jump from outside `H..X` lands strictly inside it, and nothing
///   inside it leaves for `X` or beyond: no jump, no `RETURN`, and the
///   instruction at `X - 1` is a backward jump.
///
/// In addition every `LOAD_ARG` index must be below `arg_count`, so
/// generated code never reads past the arguments, every local index below
/// `MAX_LOCALS`, and there are at most `MAX_JUMPS` jumps.
///
/// Returns the number of local slots (one more than the largest index
/// used, or 0) and the loop, if there is one.
pub fn validate(ops: &[Op], arg_count: usize) -> Result<(usize, Option<Loop>), String> {
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
    let found = find_loop(ops)?;
    let header = found.map(|found| found.header);

    // One pass in index order sees every forward predecessor of an
    // instruction before the instruction itself. The only other edges are
    // the loop's backedges into its header: the header's state comes from
    // its forward predecessors alone (the first iteration), and every
    // backedge is checked to bring back a compatible state (an empty stack
    // and at least the same assigned slots), so joining it in would change
    // nothing and the pass has reached a fixed point.
    //
    // `assigned` is `None` where the previous instruction does not fall
    // through. `joins[target]` holds the slots stored on every forward jump
    // into a target not reached yet; the stack is always empty there.
    let mut stack: Vec<Kind> = Vec::new();
    let mut assigned: Option<SlotSet> = Some(SlotSet::empty(local_count));
    let mut joins: Vec<Option<SlotSet>> = vec![None; ops.len()];
    let mut at_header: Option<SlotSet> = None;
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
        if header == Some(index) {
            if !stack.is_empty() {
                return Err(format!(
                    "the loop header at index {index} is reached from index {} \
                     with {} value(s) on the stack; the stack must be empty \
                     where paths join",
                    index - 1,
                    stack.len()
                ));
            }
            at_header = Some(current.clone());
        }

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
                    let target = target as usize;
                    if target <= index {
                        // A backedge: `find_loop` proved it goes to the
                        // header, which this pass has already visited.
                        let entry = at_header.as_ref().ok_or_else(|| {
                            format!("backward jump at index {index} has no header")
                        })?;
                        if !entry.is_subset(current) {
                            return Err(format!(
                                "the backward jump at index {index} returns to the loop \
                                 header with fewer assigned locals than it started with"
                            ));
                        }
                    } else {
                        let join = &mut joins[target];
                        match join {
                            None => *join = Some(current.clone()),
                            Some(join) => join.intersect(current),
                        }
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
    Ok((local_count, found))
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
