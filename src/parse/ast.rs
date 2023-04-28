use crate::util::file::Span;
use crate::util::InternedStr;

#[derive(Debug)]
pub enum BinaryOp {
    Or,
    And,
    BitOr,
    BitXor,
    BitAnd,
    Eq,
    Neq,
    Leq,
    Geq,
    Lt,
    Gt,
    Add,
    Sub,
    Mult,
    Div,
    Mod
}

#[derive(Debug)]
pub enum UnaryOp {
    Minus,
    Plus,
    Ref,
    Deref,
    Negate
}

#[derive(Debug)]
pub enum ExprKind {
    Binary {
        op: BinaryOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>
    },
    Unary {
        op: UnaryOp,
        rhs: Box<Expr>
    },
    FnCall {
        callee: Box<Expr>,
        args: Vec<Box<Expr>>
    },
    Subscript {
        val: Box<Expr>,
        expr: Box<Expr>
    },
    Assign {
        lval: Box<Expr>,
        rval: Box<Expr>
    },
    IntLiteral(u64),
    StrLiteral(InternedStr),
    Var(InternedStr)
}

#[derive(Debug)]
pub struct Expr {
    pub(crate) kind: ExprKind,
    pub(crate) span: Span
}

#[derive(Debug)]
pub enum PrimitiveType {
    Unsigned8,
    Unsigned16,
    Unsigned32,
    Unsigned64,
    Signed8,
    Signed16,
    Signed32,
    Signed64,
    Char,
    Str
}

#[derive(Debug)]
pub enum BaseType {
    Primitive(PrimitiveType)
}

#[derive(Debug)]
pub enum Type {
    Base(BaseType),
    Array(Box<Type>, Option<Box<Expr>>),
    Pointer(Box<Type>),
    Void
}

#[derive(Debug)]
pub struct FnArg {
    pub(crate) name: InternedStr,
    pub(crate) arg_type: Box<Type>
}

#[derive(Debug)]
pub enum Stmt {
    Return {
        retval: Option<Box<Expr>>
    },
    If {
        condition: Expr,
        body: Box<StmtBlock>,
        else_body: Option<Box<StmtBlock>>
    },
    While {
        condition: Box<Expr>,
        body: Box<StmtBlock>
    },
    VarDecl {
        name: InternedStr,
        type_hint: Option<Box<Type>>,
        value: Option<Box<Expr>>
    },
    FnDecl {
        name: InternedStr,
        return_type: Box<Type>,
        args: Vec<FnArg>,
        body: Option<StmtBlock>
    },
    Expr(Box<Expr>)
}

#[derive(Debug)]
pub struct StmtBlock {
    pub stmts: Vec<Stmt>,
    pub span: Span
}
