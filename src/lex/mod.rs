pub mod lexer;

use crate::util::file::Span;
use crate::util::InternedStr;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum KwKind {
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    Char,
    Str,
    Return,
    While,
    If,
    Fn,
    Else,
    Let,
}

impl TryFrom<&str> for KwKind {
    type Error = ();

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Ok(match value {
            "u8" => Self::U8,
            "u16" => Self::U16,
            "u32" => Self::U32,
            "u64" => Self::U64,
            "i8" => Self::I8,
            "i16" => Self::I16,
            "i32" => Self::I32,
            "i64" => Self::I64,
            "char" => Self::Char,
            "str" => Self::Str,
            "return" => Self::Return,
            "while" => Self::While,
            "if" => Self::If,
            "fn" => Self::Fn,
            "else" => Self::Else,
            "let" => Self::Let,
            _ => return Err(()),
        })
    }
}

/// Determines the type of a Token. This type should not own any heap data to keep it Copy.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum TokKind {
    Identifier(InternedStr),
    Keyword(KwKind),
    IntLiteral(u64),
    StrLiteral(InternedStr), // TODO: Separate interner?
    LSqBracket,              // "["
    RSqBracket,              // "]"
    Star,                    // "*"
    LParen,                  // "("
    RParen,                  // ")"
    Semicolon,               // ";"
    LCurlyBracket,           // "{"
    RCurlyBracket,           // "}"
    Colon,                   // ":"
    Comma,                   // ","
    EqSign,                  // "="
    DoubleVBar,              // "||"
    DoubleAnd,               // "&&"
    VBar,                    // "|"
    And,                     // "&"
    Caret,                   // "^"
    DoubleEq,                // "=="
    ExclEq,                  // "!="
    LAngleEq,                // "<="
    RAngleEq,                // ">="
    LAngleBracket,           // "<"
    RAngleBracket,           // ">"
    Plus,                    // "+"
    Minus,                   // "-"
    Slash,                   // "/"
    Percent,                 // "%"
    ExclMark,                // "!"
    Arrow,                   // "->"
    Eof,
}

/// Represents a single token in the input source.
///
/// The information in this struct should be enough to reconstruct the source form of the token. The
/// type itself doesn't own any data and is Copy, so that it's easier to pass Tokens around the program.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Token {
    pub kind: TokKind,
    pub span: Span,
}
