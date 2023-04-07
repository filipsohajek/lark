use crate::lex::LexerError;
use crate::util::file::Span;
use thiserror::Error;

/// Base error type for the Lark compiler.
#[derive(Error, Debug, Copy, PartialEq, Clone)]
pub enum LarkError {
    #[error("Lexer error")]
    Lex(LexerError, Span)
}

pub type LarkResult<T> = Result<T, LarkError>;