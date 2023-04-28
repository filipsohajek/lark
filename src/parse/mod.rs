pub mod ast;

use std::fmt::Debug;
use std::iter;

use crate::lex::lexer::Lexer;
use crate::lex::{KwKind, TokKind, Token};
use crate::parse::ast::{
    BaseType, BinaryOp, Expr, ExprKind, FnArg, PrimitiveType, Stmt, StmtBlock, Type
};
use crate::util::diag::{DiagEngine, MsgKind};
use crate::util::file::Span;
use crate::util::InternedStr;

/// Iterator adapter allowing arbitrary fixed-size lookahead
///
/// The adapter keeps the current lookahead in a ring buffer, which is populated
/// every time it advances. Therefore there is no guarantee on the
/// [Iterator::next] calls to the underlying iterator. The ring buffer is
/// populated upon creation, which allows the [PeekN::peek] and [PeekN::peek_n]
/// methods to be pure.
#[derive(Clone, Debug)]
struct PeekN<I: Iterator, const N: usize>
where
    I::Item: Sized + Copy + Debug
{
    buffer: [Option<I::Item>; N],
    cur_index: usize,
    parent: I
}

impl<I, const N: usize> PeekN<I, N>
where
    I::Item: Sized + Copy + Debug,
    I: Iterator
{
    /// Take exactly N items from the provided iterator as [Option], padding the
    /// result with [Option::None] if the iterator has finished before.
    fn take_n_exact(iter: &mut I) -> [Option<I::Item>; N] {
        iter.by_ref()
            .map(|x| Some(x))
            .chain(iter::repeat(None))
            .take(N)
            .collect::<Vec<Option<I::Item>>>()
            .try_into()
            .unwrap()
    }

    fn new(mut parent: I) -> Self {
        Self {
            buffer: Self::take_n_exact(&mut parent),
            cur_index: 0,
            parent
        }
    }

    /// Peek at the next token in the iterator. Does not mutate state.
    fn peek(&self) -> &Option<I::Item> {
        &self.buffer[self.cur_index]
    }

    /// Peek at the n-th following token in the iterator. Does not mutate state.
    fn peek_n(&self, n: usize) -> &Option<I::Item> {
        assert!(n <= N);
        self.buffer[self.cur_index..]
            .iter()
            .chain(self.buffer[..self.cur_index].iter())
            .nth(n - 1)
            .unwrap()
    }
}

impl<I, const N: usize> Iterator for PeekN<I, N>
where
    I::Item: Sized + Copy + Debug,
    I: Iterator
{
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        let next_it = self.buffer[self.cur_index];
        self.buffer[self.cur_index] = self.parent.next();
        if self.cur_index == (N - 1) {
            self.cur_index = 0
        } else {
            self.cur_index += 1;
        }

        next_it
    }
}

/// Wrapper for peeking and advancing lexer state in the parser. Peeks at the
/// next token and matches it against the provided patterns, advancing the lexer
/// if the patterns match.
macro_rules! accept_token {
    ($parser:expr, $( $kind:pat => $accept_block:expr,)+ $else_block: block) => (
        match $parser.peek_token() {
            $(&Some(Token {kind: $kind, ..}) => {
                $parser.next_token();
                $accept_block
            },)+
            _ => {
                $else_block
            }
        }
    );

    ($parser:expr, $( $kind:pat => $accept_block:block,)+ @$tok:pat => $else_block: block) => (
        match $parser.peek_token() {
            $(&Some(Token {kind: $kind, ..}) => {
                $parser.next_token();
                $accept_block
            },)+
            &$tok => {
                $else_block
            }
        }
    )
}

/// Lark parser. Uses a recursive descent parser to create an AST structure from
/// the tokens of the provided [Lexer]. The parser attempts to error-recover as
/// much as possible and will always produce a valid AST. Any errors during the
/// parsing are captured using the [Diag] interface.
pub struct Parser<'lex> {
    lexer: PeekN<Lexer<'lex>, 1>,
    diag_engine: DiagEngine,
    last_span: Span
}

impl<'lex> Parser<'lex> {
    pub fn new(lexer: Lexer<'lex>) -> Self {
        Self {
            lexer: PeekN::new(lexer),
            diag_engine: DiagEngine::new(),
            last_span: Span::default()
        }
    }

    /// Advance the lexer to the next token and update the last-seen Span.
    fn next_token(&mut self) -> Option<Token> {
        let tok = self.lexer.next();
        if let Some(Token { span, .. }) = tok {
            self.last_span = span;
        }
        tok
    }

    fn peek_token(&mut self) -> &Option<Token> {
        self.lexer.peek()
    }

    fn diag_last(&mut self, kind: MsgKind) {
        self.diag_engine.diag(kind, self.last_span)
    }

    fn diag_tok_or_last(&mut self, kind: MsgKind, tok_opt: Option<Token>) {
        self.diag_engine
            .diag(kind, tok_opt.map(|t| t.span).unwrap_or(self.last_span))
    }

    /// Parse a function definition argument list (FnArgList). Does not attempt
    /// to recover missing commas nor type specifiers.
    ///
    /// FnHeader:
    ///     fn Ident ( FnArgList ) -> Type
    ///
    /// FnArgList:
    ///     FnArg
    ///     FnArg , FnArgList
    ///
    /// FnArg:
    ///     Ident : Type
    fn parse_fn_arg_list(&mut self) -> Option<Vec<FnArg>> {
        let mut args = Vec::new();
        // match empty argument list
        accept_token!(self,
            TokKind::RParen => { return Some(args); },
            {}
        );

        'args: loop {
            let arg_name = accept_token!(self,
            TokKind::Identifier(arg_name) => arg_name, {
                self.diag_last(MsgKind::UnexpectedTokenInFnArgs);
                return None;
            });

            accept_token!(self, TokKind::Colon => {}, {
                self.diag_last(MsgKind::ExpectedColonAfterFnArg);
                return None;
            });
            let Some(arg_type) = self.parse_type() else {
                return None;
            };

            accept_token!(self,
                TokKind::Comma => {},  // argument separator ","
                TokKind::RParen => break 'args,  // argument list end ")"
                {
                    self.diag_last(MsgKind::ExpectedCommaAfterFnArg);
                    return None;
                }
            );
            args.push(FnArg { name: arg_name, arg_type });
        }
        Some(args)
    }

    /// Parse a function call arguments (ArgList). Does not attempt to recover
    /// any errors.
    ///
    /// ArgList:
    ///     AssignExpr
    ///     AssignExpr , ArgList
    fn parse_fn_callargs(&mut self) -> Option<Vec<Box<Expr>>> {
        let mut args = Vec::new();
        // match empty argument list
        accept_token!(self, TokKind::RParen => {return Some(args);}, {});
        'args: loop {
            let Some(expr) = self.parse_expr() else {
                    return None;
                };
            args.push(expr);

            accept_token!(self,
                TokKind::Comma => {},  // argument separator ","
                TokKind::RParen => break 'args,  // argument list end ")"
                {
                    self.diag_last(MsgKind::ExpectedCommaAfterFnArg);
                    return None;
                }
            )
        }
        Some(args)
    }

    /// Parse the postfix expression suffix of an expression (PostfixExpr).
    /// Attempts to recover from:
    /// - unclosed array subscript brackets (assumed closing after expression)
    /// - invalid function call arguments (assumed empty)
    ///
    /// PostfixExpr:
    ///     BaseExpr
    ///     PostfixExpr \[ Expr \]
    ///     PostfixExpr ( ArgList )
    fn parse_postfix_suffix(&mut self, mut val: Box<Expr>) -> Option<Box<Expr>> {
        loop {
            let span_start = self.last_span; // used for computing expression spans
            val = accept_token!(self,
            TokKind::LParen => {  // function call opening "("
                let args = self.parse_fn_callargs()
                                .unwrap_or_else(|| vec![]);  // recover from invalid fn calls
                Box::new(Expr {
                    kind: ExprKind::FnCall {callee: val, args},
                    span: span_start.between(self.last_span)
                })
            },
            TokKind::LSqBracket => {  // array subscript opening "["
                let Some(expr) = self.parse_expr() else {
                    return None;
                };
                accept_token!(self, TokKind::RSqBracket => {}, {
                    self.diag_last(MsgKind::UnclosedParen)
                    // assume a closing bracket after the expression
                });
                Box::new(Expr {
                    kind: ExprKind::Subscript {val, expr},
                    span: span_start.between(self.last_span)
                })
            },
            {
                return Some(val);
            })
        }
    }

    /// Parse an expression value with an optional postfix part (PostfixExpr).
    /// Attempts to recover from unclosed parentheses by assuming a closing
    /// after the inner expression.
    ///
    /// PostfixExpr:
    ///     BaseExpr
    ///     PostfixExpr [ Expr ]
    ///     PostfixExpr ( ArgList )
    ///
    /// BaseExpr:
    ///     ( Expr )
    ///     Identifier
    ///     IntegerLiteral
    ///     StringLiteral
    fn parse_postfix_expr(&mut self) -> Option<Box<Expr>> {
        let val = Box::new(match self.next_token() {
            Some(Token { kind: TokKind::IntLiteral(val), span }) => {
                Expr { kind: ExprKind::IntLiteral(val), span }
            },
            Some(Token { kind: TokKind::StrLiteral(val), span }) => {
                Expr { kind: ExprKind::StrLiteral(val), span }
            },
            Some(Token { kind: TokKind::Identifier(val), span }) => {
                Expr { kind: ExprKind::Var(val), span }
            },
            Some(Token { kind: TokKind::LParen, .. }) => {
                // parenthesized expression
                let Some(parsed_expr) = self.parse_expr() else {
                    return None;
                };
                accept_token!(self, TokKind::RParen => {}, {
                    self.diag_last(MsgKind::UnclosedParen);
                    // assume a closing one after the expression
                });

                return Some(parsed_expr);
            },
            tok => {
                self.diag_tok_or_last(MsgKind::ExpectedValueInExpression, tok);
                return None;
            }
        });

        self.parse_postfix_suffix(val)
    }

    /// Parse an expression (Expr).
    ///
    /// Expr:
    ///     AssignExpr
    ///
    /// AssignExpr:
    ///     OrOrExpr
    ///     OrOrExpr = AssignExpr
    ///
    /// ConstantExpr:
    ///     OrOrExpr
    ///
    /// OrOrExpr:
    ///     AndAndExpr
    ///     OrOrExpr || AndAndExpr
    ///
    /// AndAndExpr:
    ///     OrExpr
    ///     AndAndExpr && OrExpr
    ///
    /// OrExpr:
    ///     XorExpr
    ///     OrExpr | XorExpr
    ///
    /// XorExpr:
    ///     AndExpr
    ///     XorExpr ^ AndExpr
    ///
    /// AndExpr:
    ///     CmpExpr
    ///     AndExpr & CmpExpr
    ///
    /// CmpExpr:
    ///     AddExpr
    ///     AddExpr == AddExpr
    ///     AddExpr != AddExpr
    ///     AddExpr <= AddExpr
    ///     AddExpr >= AddExpr
    ///     AddExpr < AddExpr
    ///     AddExpr > AddExpr
    ///
    /// AddExpr:
    ///     MultExpr
    ///     AddExpr + MultExpr
    ///     AddExpr - MultExpr
    ///
    /// MultExpr:
    ///     UnaryExpr
    ///     MultExpr * UnaryExpr
    ///     MultExpr / UnaryExpr
    ///     MultExpr % UnaryExpr
    ///
    /// UnaryExpr:
    ///     BaseExpr
    ///     - UnaryExpr
    ///     + UnaryExpr
    ///     & UnaryExpr
    ///     * UnaryExpr
    ///     ! UnaryExpr
    ///
    /// PostfixExpr:
    ///     BaseExpr
    ///     PostfixExpr [ Expr ]
    ///     PostfixExpr ( ArgList )
    ///
    /// BaseExpr:
    ///     ( Expr )
    ///     Identifier
    ///     IntegerLiteral
    ///     StringLiteral
    fn parse_expr(&mut self) -> Option<Box<Expr>> {
        self.parse_postfix_expr()
            .and_then(|lhs| self.parse_expr_int(lhs, 0))
    }

    /// Get the precedence of a binary operation corresponding to the given
    /// token.
    ///
    /// The method takes a reference to the operation token in an expression
    /// (i.e. "+", "-", ...) and returns an optional precedence if the token
    /// corresponds to a binary operation. Higher values indicate greater
    /// precedence. If the given token doesn't correspond to a binary op, None
    /// is returned.
    ///
    /// The precedence table is as follows (from least to most precedent):
    /// - Assignment ("="),
    /// - Logical OR ("||"),
    /// - Logical AND ("&&")
    /// - Bitwise OR ("|")
    /// - Bitwise XOR ("^")
    /// - Bitwise AND ("&")
    /// - Equality comparison operations ("==", "!=")
    /// - Inequality comparison operations ("<", "<=", ">", ">=")
    /// - Addition/subtraction ("+", "-")
    /// - Multiplication/division/modulo ("*", "/", "%")
    fn token_binop_pred(&mut self, token_opt: &Option<Token>) -> Option<u8> {
        let token = match token_opt {
            Some(tok) => tok,
            None => return None
        };
        Some(match token.kind {
            TokKind::EqSign => 1,
            TokKind::DoubleVBar => 2,
            TokKind::DoubleAnd => 3,
            TokKind::VBar => 4,
            TokKind::Caret => 5,
            TokKind::And => 6,
            TokKind::DoubleEq | TokKind::ExclEq => 7,
            TokKind::LAngleBracket
            | TokKind::LAngleEq
            | TokKind::RAngleBracket
            | TokKind::RAngleEq => 8,
            TokKind::Plus | TokKind::Minus => 9,
            TokKind::Star | TokKind::Slash | TokKind::Percent => 10,
            _ => return None
        })
    }

    /// Apply a binary operation to two operands.
    ///
    /// Takes in a token kind corresponding to a binary operation, and a LHS/RHS
    /// pair. Returns the expression resulting from applying the
    /// corresponding binary operation to the LHS/RHS. Panics if the
    /// supplied token kind does not match any binary operation.
    fn token_binop_apply(
        &mut self,
        token_kind: TokKind,
        lhs: Box<Expr>,
        rhs: Box<Expr>
    ) -> Box<Expr> {
        let span = lhs.span.between(rhs.span);

        let kind = match token_kind {
            TokKind::EqSign => ExprKind::Assign { lval: lhs, rval: rhs },
            TokKind::DoubleVBar => ExprKind::Binary { op: BinaryOp::Or, lhs, rhs },
            TokKind::DoubleAnd => ExprKind::Binary { op: BinaryOp::And, lhs, rhs },
            TokKind::VBar => ExprKind::Binary { op: BinaryOp::BitOr, lhs, rhs },
            TokKind::Caret => ExprKind::Binary { op: BinaryOp::BitXor, lhs, rhs },
            TokKind::And => ExprKind::Binary { op: BinaryOp::BitAnd, lhs, rhs },
            TokKind::DoubleEq => ExprKind::Binary { op: BinaryOp::Eq, lhs, rhs },
            TokKind::ExclEq => ExprKind::Binary { op: BinaryOp::Neq, lhs, rhs },
            TokKind::LAngleBracket => ExprKind::Binary { op: BinaryOp::Lt, lhs, rhs },
            TokKind::LAngleEq => ExprKind::Binary { op: BinaryOp::Leq, lhs, rhs },
            TokKind::RAngleBracket => ExprKind::Binary { op: BinaryOp::Gt, lhs, rhs },
            TokKind::RAngleEq => ExprKind::Binary { op: BinaryOp::Geq, lhs, rhs },
            TokKind::Plus => ExprKind::Binary { op: BinaryOp::Add, lhs, rhs },
            TokKind::Minus => ExprKind::Binary { op: BinaryOp::Sub, lhs, rhs },
            TokKind::Star => ExprKind::Binary { op: BinaryOp::Mult, lhs, rhs },
            TokKind::Slash => ExprKind::Binary { op: BinaryOp::Div, lhs, rhs },
            TokKind::Percent => ExprKind::Binary { op: BinaryOp::Mod, lhs, rhs },
            _ => unreachable!()
        };

        Box::new(Expr { kind, span })
    }

    /// Parse an expression using a recursive operator-precedence algorithm.
    ///
    /// The method parses operations until it encounters one with precedence
    /// lower than supplied. On each operation, it parses the LHS (using the
    /// [parse_postfix_expr] method) and then folds operations on RHS while
    /// they are of higher precedence.
    fn parse_expr_int(&mut self, mut lhs: Box<Expr>, min_pred: u8) -> Option<Box<Expr>> {
        let mut lookahead = *self.peek_token();
        while let Some(op_pred) = self.token_binop_pred(&lookahead) {
            if op_pred < min_pred {
                // accept only operations with greater-than-minimal precedence
                break;
            }
            let op = lookahead;
            self.next_token();

            let Some(mut rhs) = self.parse_postfix_expr() else {
                // this happens only on unparseable values, cannot recover
                return None;
            };
            lookahead = *self.peek_token();

            // fold greater-precedence operations on the RHS
            while let Some(lookahead_pred) = self.token_binop_pred(&lookahead) {
                if lookahead_pred <= op_pred {
                    break;
                }
                rhs = match self.parse_expr_int(rhs, op_pred + 1) {
                    Some(new_rhs) => new_rhs,
                    None => return None
                };
                lookahead = *self.peek_token();
            }

            lhs = self.token_binop_apply(op.unwrap().kind, lhs, rhs);
        }

        Some(lhs)
    }

    /// Parse a type specifier (Type), consisting of a base type and optional
    /// suffices (pointer, array). Tries to recover from unclosed array
    /// bracket (assumes closing after expression).
    ///
    /// Type:
    ///     BaseType
    ///     BaseType PointerSuffix
    ///     BaseType PointerSuffix ArraySuffix*
    ///
    /// BaseType:
    ///     PrimitiveType
    ///
    /// PrimitiveType:
    ///     u8
    ///     u16
    ///     u32
    ///     u64
    ///     i8
    ///     i16
    ///     i32
    ///     i64
    ///     char
    ///     str
    fn parse_type(&mut self) -> Option<Box<Type>> {
        // determine the base type by matching keywords to primitive types
        let mut cur_type = accept_token!(self,
        TokKind::Keyword(type_kw @ (KwKind::U8 | KwKind::U16 | KwKind::U32 | KwKind::U64
                         | KwKind::I8 | KwKind::I16 | KwKind::I32 | KwKind::I64
                         | KwKind::Char | KwKind::Str)) => {
            Type::Base(BaseType::Primitive(match type_kw {
                KwKind::U8 => PrimitiveType::Unsigned8,
                KwKind::U16 => PrimitiveType::Unsigned16,
                KwKind::U32 => PrimitiveType::Unsigned32,
                KwKind::U64 => PrimitiveType::Unsigned64    ,
                KwKind::I8 => PrimitiveType::Signed8,
                KwKind::I16 => PrimitiveType::Signed16,
                KwKind::I32 => PrimitiveType::Signed32,
                KwKind::I64 => PrimitiveType::Signed64,
                KwKind::Char => PrimitiveType::Char,
                KwKind::Str => PrimitiveType::Str,
                _ => unreachable!()
            }))
        },
        @token => {
            self.diag_tok_or_last(MsgKind::InvalidType, token);
            return None;
        });

        cur_type = accept_token!(
            self,
            TokKind::Star => Type::Pointer(Box::new(cur_type)),
            { cur_type }
        );

        // parse array suffices (there may be multiple)
        loop {
            accept_token!(self, TokKind::LSqBracket => {}, {break});
            let expr = accept_token!(self, TokKind::RSqBracket => None, {
                let Some(parsed_expr) = self.parse_expr() else {
                    return None;
                };
                accept_token!(self, TokKind::RSqBracket => Some(parsed_expr), {None})
            });
            cur_type = Type::Array(Box::new(cur_type), expr)
        }

        Some(Box::new(cur_type))
    }

    /// Parse a function definition header (FnHeader). Attempts to recover from
    /// an invalid argument list by assuming no arguments.
    ///
    /// FnHeader:
    ///     fn Ident ( FnArgList ) -> Type
    ///
    /// FnArgList:
    ///     FnArg
    ///     FnArg , FnArgList
    ///
    /// FnArg:
    ///     Ident : Type
    fn parse_fn_hdr(&mut self) -> Option<(InternedStr, Box<Type>, Vec<FnArg>)> {
        let fn_name = accept_token!(self, TokKind::Identifier(fn_name) => fn_name, {
            self.diag_last(MsgKind::ExpectedNameAfterFn);
            return None;
        });
        accept_token!(self, TokKind::LParen => {}, {
            self.diag_last(MsgKind::ExpectedParenAfterFnName);
            return None;
        });

        let arg_list = self.parse_fn_arg_list().unwrap_or_else(|| vec![]);
        let ret_type = accept_token!(self,
            TokKind::Arrow => {
                match self.parse_type() {
                    Some(typ) => typ,
                    None => {
                        return None;
                    }
                }
            },
            {Box::new(Type::Void)}
        );

        return Some((fn_name, ret_type, arg_list));
    }

    /// Parse a function declaration (FnDecl). Tries to recover from a missing
    /// semicolon in externed function declarations by assuming one at the
    /// end of the header.
    ///
    /// FnDecl:
    ///     FnHeader ;
    ///     FnHeader { StmtList }
    ///
    /// FnHeader:
    ///     fn Ident ( FnArgList ) -> Type
    ///
    /// FnArgList:
    ///     FnArg
    ///     FnArg , FnArgList
    ///
    /// FnArg:
    ///     Ident : Type
    fn parse_fn_decl(&mut self) -> Option<Stmt> {
        let Some((name, return_type, args)) = self.parse_fn_hdr() else {
                return None;
            };

        // assumes parse_block returns None on no opening brace
        let body = self.parse_block();

        if body.is_none() {
            accept_token!(self, TokKind::Semicolon => {}, {
                self.diag_last(MsgKind::MissingSemicolon);
            });
        }

        return Some(Stmt::FnDecl { name, return_type, args, body });
    }

    /// Parse a statement block (StmtBlock). Attempts to recover from invalid
    /// statements by ignoring them.
    ///
    /// StmtBlock:
    ///     { StmtList }
    ///
    /// StmtList:
    ///     Stmt
    ///     Stmt StmtList
    fn parse_block(&mut self) -> Option<StmtBlock> {
        let start_span = match self.peek_token() {
            &Some(Token { kind: TokKind::LCurlyBracket, span }) => {
                self.next_token();
                span
            },
            _ => return None
        };

        let mut stmts = Vec::new();
        let mut end_span = start_span;

        while let &Some(Token { kind, span }) = self.peek_token() {
            if kind == TokKind::RCurlyBracket {
                self.next_token();
                end_span = span;
                break;
            }
            if let Some(stmt) = self.parse_stmt() {
                stmts.push(stmt);
            };
        }

        Some(StmtBlock { stmts, span: start_span.between(end_span) })
    }

    /// Parse a variable declaration (VarDecl). Attempts to recover from a
    /// missing semicolon by assuming one after the declaration.
    ///
    /// VarDecl:
    ///     let Ident : Type ;
    ///     let Ident = Expr ;
    ///     let Ident : Type = Expr ;
    fn parse_var_decl(&mut self) -> Option<Stmt> {
        let name = accept_token!(self, TokKind::Identifier(var_name) => var_name, {
            self.diag_last(MsgKind::ExpectedNameAfterLet);
            return None;
        });

        let type_hint = accept_token!(self,
            TokKind::Colon => {
                match self.parse_type() {
                    Some(typ) => Some(typ),
                    None => return None
                }
            },
            {None}
        );

        let value = accept_token!(self,
            TokKind::EqSign => {
                match self.parse_expr() {
                    Some(expr) => Some(expr),
                    None => return None
                }
            },
            {None}
        );

        accept_token!(self, TokKind::Semicolon => {}, {
            self.diag_last(MsgKind::MissingSemicolon)
        });

        Some(Stmt::VarDecl { name, type_hint, value })
    }

    /// Resynchronize to the nearest statement in the current block.
    ///
    /// Skips tokens up until a semicolon or a valid statement start is
    /// encountered. Keeps track of the current block scope and ignores
    /// resynchronization tokens inside nested blocks. If the end of the
    /// current block scope is encountered during resynchronization, the
    /// function returns before this end.
    fn resync_stmt(&mut self) {
        let mut block_scope = 0;
        while let Some(Token { kind, .. }) = self.peek_token() {
            match kind {
                TokKind::Semicolon
                | TokKind::Keyword(KwKind::If | KwKind::Return | KwKind::Let | KwKind::While)
                    if block_scope == 0 =>
                {
                    return
                },
                TokKind::LCurlyBracket => block_scope += 1,
                TokKind::RCurlyBracket if block_scope == 0 => return,
                TokKind::RCurlyBracket => block_scope -= 1,
                _ => {}
            };
            self.next_token();
        }
    }

    /// Parse a while statement (WhileStmt).
    ///
    /// WhileStmt:
    ///     while Expr StmtBlock
    fn parse_while(&mut self) -> Option<Stmt> {
        let Some(condition) = self.parse_expr() else {
            return None;
        };
        let Some(body) = self.parse_block() else {
            return None;
        };

        Some(Stmt::While { condition, body: Box::new(body) })
    }

    /// Parse a return statement (ReturnStmt). Attempts to recover from a
    /// missing semicolon by assuming one.
    ///
    /// ReturnStmt:
    ///     return Expr ;
    fn parse_return(&mut self) -> Option<Stmt> {
        let retval = accept_token!(self,
            TokKind::Semicolon => None,
            {
                let Some(expr) = self.parse_expr() else {
                    return None;
                };
                accept_token!(self, TokKind::Semicolon => {}, {
                    self.diag_last(MsgKind::MissingSemicolon);
                });

                Some(expr)
            }
        );

        Some(Stmt::Return { retval })
    }

    /// Parse an expression statement (ExprStmt). Does not attempt to recover
    /// from a missing semicolon.
    ///
    /// ExprStmt:
    ///     Expr ;
    fn parse_expr_stmt(&mut self) -> Option<Stmt> {
        let Some(expr) = self.parse_expr() else {
            return None;
        };

        accept_token!(self, TokKind::Semicolon => {}, {
            self.diag_last(MsgKind::MissingSemicolon);
            return None;
        });

        Some(Stmt::Expr(expr))
    }

    /// Parse a statement (Stmt). Resynchronizes to the next nearest statement
    /// upon failure (see [resync_stmt]).
    ///
    /// Stmt:
    ///     WhileStmt
    ///     ReturnStmt
    ///     DeclStmt
    ///     ExprStmt
    ///     IfStmt
    fn parse_stmt(&mut self) -> Option<Stmt> {
        let Some(stmt) = accept_token!(self,
            TokKind::Keyword(KwKind::Fn) => self.parse_fn_decl(),
            TokKind::Keyword(KwKind::Let) => self.parse_var_decl(),
            TokKind::Keyword(KwKind::While) => self.parse_while(),
            TokKind::Keyword(KwKind::Return) => self.parse_return(),
            {self.parse_expr_stmt()}
        ) else {
            self.resync_stmt();
            return None;
        };

        Some(stmt)
    }

    /// Parse the lexer tokens into an abstract syntax tree (AST).
    ///
    /// Always produces a valid AST, errors are captured into the [DiagEngine]
    /// and available through the [Diag] interface. If an error is present
    /// in the [DiagEngine] after parsing, the AST should be presumed to be
    /// incomplete/logically incorrect.
    pub fn parse(&mut self) -> Vec<Stmt> {
        let mut stmts = Vec::new();
        while let &Some(Token { kind, span }) = self.peek_token() {
            match kind {
                TokKind::Keyword(KwKind::Fn | KwKind::Let | KwKind::While | KwKind::Return) => {
                    let stmt = self.parse_stmt();
                    if stmt.is_some() {
                        stmts.push(stmt.unwrap())
                    }
                },
                _kind => {
                    self.diag_engine.diag(MsgKind::UnexpectedToken, span);
                    self.next_token();
                }
            }
        }

        stmts
    }
}

#[cfg(test)]
mod test {
    use crate::parse::ast::Stmt;
    use crate::util::diag::DiagMsg;
    use crate::util::file::Span;

    fn parse_buf(buf: &str) -> (Vec<Stmt>, Vec<DiagMsg>) {
        let buf_span = Span::new_from_buf(buf).unwrap();
        let mut parser = buf_span.parser();

        (
            parser.parse(),
            parser.diag_engine.messages().cloned().collect()
        )
    }
    #[test]
    fn test_parser() {
        let buf = "fn print_int(val: u32);

fn factorial(x: u32) -> u32 {
    let res: u32 = 0;
    let i: u32 = 1;
    let s = \"ABCD\";

    while i <= x {
        res = res * i;
        i = i + 1;
    }

    return res;
}

fn main(args: str[]) -> u32 {
    print_int(factorial(parse_u32(args[0])));
}";
        let (ast, msgs) = parse_buf(buf);
        println!("{:#?}", ast);
        for msg in msgs {
            println!("{}", msg);
        }
    }
}
