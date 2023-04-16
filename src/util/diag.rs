use std::slice::Iter;
use crate::util::file::Span;

pub enum MsgKind {
    IntegerConstantOverflow,
    InvalidEscapeSequence,
    UnexpectedCharacter
}

struct DiagMsg {
    kind: MsgKind,
    span: Span
}

/// An engine for collecting diagnostic messages.
///
/// Each implementor of the [Diag] trait shall hold exactly one instance of this struct and collect
/// all its diagnostics errors there.
pub struct DiagEngine {
    messages: Vec<DiagMsg>
}

impl DiagEngine {
    pub fn new() -> Self {
        Self {
            messages: vec![],
        }
    }

    pub fn diag(&mut self, kind: MsgKind, span: Span) {
        self.messages.push(DiagMsg {
            kind,
            span,
        })
    }

    pub fn messages(&mut self) -> Iter<DiagMsg> {
        self.messages.iter()
    }
}


/// Trait for objects using the diagnostics interface.
///
/// A large proportion of Lark subsystems are not well-suited for using the builtin Rust error
/// handling scheme. This is due to a need to recover from errors, sometimes multiple ones. Using
/// Rust's Error/Result types would require complex handling on all levels, in contrast to this
/// diagnostic scheme. The diagnostics system is meant to be used primarily for user-facing errors,
/// which do not alter the overall program flow in a major way.
///
/// Implementations of this trait provide a way for the parent to track errors encountered throughout
/// the implementor's lifetime by providing access to the internal [DiagEngine] instance. The trait
/// also includes convenience methods for accessing the DiagEngine API.
pub trait Diag {
    fn diag_engine(&self) -> &DiagEngine;
}