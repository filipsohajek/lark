use std::fmt::{Display, Formatter};
use crate::util::file::Span;
use std::slice::Iter;
use crate::lex::Token;

#[derive(Debug, Clone, Copy)]
pub enum MsgKind {
    IntegerConstantOverflow,
    InvalidEscapeSequence,
    UnexpectedCharacter,
    ExpectedNameAfterFn,
    ExpectedParenAfterFnName,
    ExpectedFnReturnType,
    ExpectedColonAfterFnArg,
    ExpectedCommaAfterFnArg,
    UnexpectedTokenInFnArgs,
    ExpectedValueInExpression,
    ExpectedNameAfterLet,
    MissingSemicolon,
    InvalidType,
    UnclosedParen,
    UnclosedBracket,
    UnexpectedToken,
}

enum Severity {
    Info,
    Warn,
    Error
}

impl Severity {
    fn as_str(&self) -> &'static str {
        match self {
            Severity::Info => "INFO",
            Severity::Warn => "WARN",
            Severity::Error => "ERROR"
        }
    }
}

impl MsgKind {
    fn severity(&self) -> Severity {
        match self {
            MsgKind::IntegerConstantOverflow => Severity::Error,
            MsgKind::InvalidEscapeSequence => Severity::Error,
            MsgKind::UnexpectedCharacter => Severity::Error,
            MsgKind::ExpectedNameAfterFn => Severity::Error,
            MsgKind::ExpectedParenAfterFnName => Severity::Error,
            MsgKind::ExpectedFnReturnType => Severity::Error,
            MsgKind::ExpectedColonAfterFnArg => Severity::Error,
            MsgKind::ExpectedCommaAfterFnArg => Severity::Error,
            MsgKind::UnexpectedTokenInFnArgs => Severity::Error,
            MsgKind::ExpectedValueInExpression => Severity::Error,
            MsgKind::ExpectedNameAfterLet => Severity::Error,
            MsgKind::MissingSemicolon => Severity::Error,
            MsgKind::InvalidType => Severity::Error,
            MsgKind::UnclosedParen => Severity::Error,
            MsgKind::UnclosedBracket => Severity::Error,
            MsgKind::UnexpectedToken => Severity::Error,
        }
    }

    fn message(&self) -> String {
        match self {
            MsgKind::IntegerConstantOverflow => format!("integer constant overflow"),
            MsgKind::InvalidEscapeSequence => format!("invalid escape sequence"),
            MsgKind::UnexpectedCharacter => format!("unexpected character"),
            MsgKind::ExpectedNameAfterFn => format!("expected function name after \"fn\""),
            MsgKind::ExpectedParenAfterFnName => format!("expected \"(\" after function name"),
            MsgKind::ExpectedFnReturnType => format!("expected return type after \"->\""),
            MsgKind::ExpectedColonAfterFnArg => format!("expected \":\" after function argument"),
            MsgKind::ExpectedCommaAfterFnArg => format!("expected \",\" after function argument"),
            MsgKind::UnexpectedTokenInFnArgs => format!("unexpected token in function arguments"),
            MsgKind::ExpectedValueInExpression => format!("expected value"),
            MsgKind::ExpectedNameAfterLet => format!("expected variable name after \"let\""),
            MsgKind::MissingSemicolon => format!("missing semicolon"),
            MsgKind::InvalidType => format!("invalid type specifier"),
            MsgKind::UnclosedParen => format!("unclosed \"(\""),
            MsgKind::UnclosedBracket => format!("unclosed \"[\""),
            MsgKind::UnexpectedToken => format!("unexpected token")
        }
    }
}


#[derive(Debug, Clone, Copy)]
pub struct DiagMsg {
    pub kind: MsgKind,
    pub span: Span,
}

impl Display for DiagMsg {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "[{}]: {}", self.kind.severity().as_str(), self.kind.message());
        writeln!(f, "{}", self.span.line_span().0);
        Ok(())
    }
}

/// An engine for collecting diagnostic messages.
///
/// Each implementor of the [Diag] trait shall hold exactly one instance of this struct and collect
/// all its diagnostics errors there.
pub struct DiagEngine {
    messages: Vec<DiagMsg>,
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