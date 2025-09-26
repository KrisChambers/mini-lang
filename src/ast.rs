

/// Expression AST
#[derive(Eq, PartialEq, PartialOrd, Debug, Clone)]
pub enum Expr {
    // Integer Value Ex: Int(1)
    Int(i32),
    // Boolean Value Ex: Bool(true)
    Bool(bool),
    // Variable Name Ex: Add(Var("a"), Int(1))
    Var(String),
    //// Let bindings : Assign an expression to a name
    Let(String, Box<Expr>),
    // Conditional
    If(Box<Expr>, Box<Expr>, Box<Expr>),

    // Lambda creation
    Lambda(Vec<Expr>, Box<Expr>),

    // Function application
    Apply(Box<Expr>, Vec<Expr>),

    /// Int Operators
    Add(Box<Expr>, Box<Expr>),
    Subtract(Box<Expr>, Box<Expr>),
    Mult(Box<Expr>, Box<Expr>),
    Divide(Box<Expr>, Box<Expr>),
}

