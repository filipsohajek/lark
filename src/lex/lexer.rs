use crate::lex::{KwKind, TokKind, Token};
use crate::util::diag::{Diag, DiagEngine, MsgKind};
use crate::util::file::Span;
use crate::util::InternedStr;
use std::iter::Peekable;
use std::str::Chars;

/// Check if the supplied character is valid to be inside of an identifier. \[a-zA-Z0-9_\]
fn is_identifier_char(c: char) -> bool {
    ((c >= 'a') && (c <= 'z'))
        || ((c >= 'A') && (c <= 'Z'))
        || ((c >= '0') && (c <= '9'))
        || (c == '_')
}

/// Lark language lexer. Lexes Tokens from an arbitrary Span backed by the thread-local SourceMap.
/// Parses integer constants and string literals into internal representations.
///
/// The lexer works in the forward direction only and at all times holds an iterator over the not
/// yet scanned characters and a Span covering all these characters. When lexing a Token, the Lexer
/// keeps track of the number of consumed bytes and after a token is finished, truncates this Span,
/// thus ensuring a consistent internal state after each token.
///
/// The lexer does not try to recover, therefore any lexing error, unless explicitly stated in its
/// documentation, is considered to be fatal and all subsequent output should be considered incorrect.
pub struct Lexer<'buf> {
    buffer: Peekable<Chars<'buf>>,
    unscanned_span: Span,
    tok_length: usize,
    diag_engine: DiagEngine,
}

impl<'buf> Lexer<'buf> {
    /// Create a new Lexer instance lexing a specified [Span]. The Span must be a valid one, backed
    /// by the thread-local [SourceMap].
    pub fn new_from_span(base_span: Span) -> Self {
        Self {
            buffer: base_span.buffer().chars().peekable(),
            unscanned_span: base_span,
            tok_length: 0,
            diag_engine: DiagEngine::new(),
        }
    }

    /// Peek at the next input character. Does not advance.
    fn peek_char(&mut self) -> Option<char> {
        self.buffer.peek().map(|&c| c)
    }

    /// Take the next input character and advance the token length.
    fn next_char(&mut self) -> Option<char> {
        let next_c = self.buffer.next();
        if next_c.is_some() {
            self.tok_length += next_c.unwrap().len_utf8();
        }

        next_c
    }

    /// Take the next input character if and only if it matches the specified one. Returns true
    /// if the character matched, false otherwise.
    fn accept_char(&mut self, accepted_c: char) -> bool {
        // Use the peek_char/next_char methods to ensure token length consistency
        match self.peek_char() {
            Some(c) if c == accepted_c => {
                self.next_char();
                true
            }
            _ => false
        }
    }

    /// Truncate the yet-unscanned [Span] and reset the token length.
    fn finish_token(&mut self) {
        self.unscanned_span = self.unscanned_span.truncate_head(self.tok_length);
        self.tok_length = 0;
    }

    /// Return the [Span] of the currently lexed token.
    fn token_span(&mut self) -> Span {
        self.unscanned_span.substr(0, self.tok_length)
    }

    fn diag_last(&mut self, msg_kind: MsgKind) {
        let span = self.token_span();
        self.diag_engine.diag(msg_kind, span);
    }

    /// Lex an input character sequence into either an identifier or a keyword. Returns either
    /// [TokKind::Identifier] or [TokKind::Keyword].
    ///
    /// This method is called from the base [Lexer::lex] method after a identifier/keyword start character
    /// has been encountered. It accepts all identifier characters (cf. [is_identifier_char]). The
    /// resulting buffer is then matched against known keywords; if this fails, the buffer is interned
    /// and an identifier token kind is returned.
    fn lex_ident_or_kw(&mut self) -> TokKind {
        while let Some(c) = self.peek_char() {
            if !is_identifier_char(c) {
                break;
            }
            self.next_char();
        }

        let ident_buf = self.token_span().buffer();
        KwKind::try_from(ident_buf)
            .map_or(TokKind::Identifier(InternedStr::new_from_str(ident_buf)),
                    |kw_kind| TokKind::Keyword(kw_kind))
    }

    /// Lex an integer constant. Returns the lexed value on success.
    ///
    /// The method starts after the first character of the constant has been lexed by the [Lexer::lex]
    /// method and takes this first character as an argument to determine the proper radix to use or
    /// to use it as an initial value.
    ///
    /// Integer constants are internally represented by the u64 type. If an overflow occurs, this
    /// method returns an error. It does not attempt to recover by advancing beyond the constant
    /// and the lexer may be in an inconsistent state afterwards (TODO).
    fn lex_int_constant(&mut self, first_digit: char) -> u64 {
        // The first digit may indicate the radix (0x), or it may be a value
        let (radix, mut val) = match first_digit {
            '0' if self.accept_char('x') => (16u64, 0u64),
            _ => (10u64, first_digit.to_digit(10).unwrap() as u64)
        };

        while let Some(c) = self.peek_char() {
            if !c.is_digit(radix as u32) {
                break;
            }
            self.next_char();

            let c_digit = c.to_digit(radix as u32).unwrap() as u64;

            // Checked-append the digit to the value, bailing out upon overflow
            val = val.checked_mul(radix)
                .and_then(|mult_val| mult_val.checked_add(c_digit))
                .map_or_else(|| {
                    self.diag_last(MsgKind::IntegerConstantOverflow);
                    val
                }, |val| val);
        }

        val
    }

    /// Lex an escape sequence. Called by [Lexer::lex_str_literal] after encountering a backslash
    /// ("\\") character. On success, it returns a single character corresponding to the lexed
    /// escape sequence.
    fn lex_escape_sequence(&mut self) -> Option<char> {
        Some(match self.next_char() {
            Some('n') => '\n',
            Some('t') => '\t',
            Some('r') => '\r',
            Some('"') => '"',
            Some('x') => {
                // A hexadecimal escape sequence has at most two digits, so we parse them explicitly.
                let digit_1 = match self.next_char() {
                    Some(c) => c.to_digit(16).unwrap(),
                    None => {
                        self.diag_last(MsgKind::InvalidEscapeSequence);
                        return None;
                    }
                };
                let digit_2 =
                    self.next_char()
                        .and_then(|c| c.to_digit(16));

                let char_code = match digit_2 {
                    Some(d) => digit_1 * 16 + d,
                    None => digit_1
                };

                match char::from_u32(char_code) {
                    Some(c) => c,
                    None => {
                        self.diag_last(MsgKind::InvalidEscapeSequence);
                        return None;
                    }
                }
            }
            _ => {
                self.diag_last(MsgKind::InvalidEscapeSequence);
                return None;
            }
        })
    }

    /// Lex a string literal. Called by [Lexer::lex] after a " character has been encountered.
    /// Accepts all characters until the matching ". Escape sequences in the string are parsed
    /// (by [Lexer::lex_escape_sequence]), the resulting string is interned, and returned.
    fn lex_str_literal(&mut self) -> InternedStr {
        let mut lexed_string = String::new();
        while let Some(c) = self.peek_char() {
            if c == '"' {
                self.next_char();
                break;
            }
            if c == '\\' {
                match self.lex_escape_sequence() {
                    Some(esc_c) => lexed_string.push(esc_c),
                    None => {}
                }
                continue;
            }
            lexed_string.push(c);
            self.next_char();
        }

        InternedStr::new_from_str(lexed_string.as_str())
    }

    /// Skip all whitespace in the input character stream and reset the lexer state so that this
    /// whitespace is ignored.
    fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek_char() {
            if !c.is_whitespace() {
                break;
            }
            self.next_char();
        }
        self.finish_token();
    }

    /// Lex a single token from the input character stream.
    ///
    /// This method is only meant to be used by the [Iterator] implementation, direct use requires
    /// properly managing the lexer state between calls, since this method does not reset the state
    /// after returning.
    fn lex(&mut self) -> TokKind {
        match self.next_char() {
            Some(c) => match c {
                '[' => TokKind::LSqBracket,
                ']' => TokKind::RSqBracket,
                '*' => TokKind::Star,
                '(' => TokKind::LParen,
                ')' => TokKind::RParen,
                ';' => TokKind::Semicolon,
                '{' => TokKind::LCurlyBracket,
                '}' => TokKind::RCurlyBracket,
                ':' => TokKind::Colon,
                ',' => TokKind::Comma,
                '=' if self.accept_char('=') => TokKind::DoubleEq,
                '=' => TokKind::EqSign,
                '<' if self.accept_char('=') => TokKind::LAngleEq,
                '<' => TokKind::LAngleBracket,
                '>' if self.accept_char('=') => TokKind::RAngleEq,
                '>' => TokKind::RAngleBracket,
                '+' => TokKind::Plus,
                '-' => TokKind::Minus,
                '/' => TokKind::Slash,
                '%' => TokKind::Percent,
                '|' if self.accept_char('|') => TokKind::DoubleVBar,
                '|' => TokKind::VBar,
                '&' if self.accept_char('&') => TokKind::DoubleAnd,
                '&' => TokKind::And,
                '^' => TokKind::Caret,
                '!' if self.accept_char('=') => TokKind::ExclEq,
                '!' => TokKind::ExclMark,
                c if is_identifier_char(c) => self.lex_ident_or_kw(),
                '_' => self.lex_ident_or_kw(),
                '"' => TokKind::StrLiteral(self.lex_str_literal()),
                c if c.is_digit(10) => TokKind::IntLiteral(self.lex_int_constant(c)),
                c if c.is_whitespace() => {
                    self.skip_whitespace();
                    return self.lex(); // tail call
                }
                _ => {
                    self.diag_last(MsgKind::UnexpectedCharacter);
                    return self.lex(); // tail call
                }
            },
            None => TokKind::Eof,
        }
    }
}

impl<'buf> Diag for Lexer<'buf> {
    fn diag_engine(&self) -> &DiagEngine {
        &self.diag_engine
    }
}

impl<'buf> Iterator for Lexer<'buf> {
    type Item = Token;

    fn next(&mut self) -> Option<Self::Item> {
        match self.lex() {
            TokKind::Eof => None,
            kind => {
                let tok = Token {
                    kind,
                    span: self.token_span(),
                };
                self.finish_token();
                Some(tok)
            }
        }
    }
}
