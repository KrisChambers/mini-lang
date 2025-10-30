use stlc::parser::{Expr, Literal, Op, TypeAnn};
use stlc::type_checker::{type_check, TypeError};

#[test]
fn type_check_valid_binop() {
    let expr = Expr::BinOp(
        Op::Add,
        Box::new(Expr::Lit(Literal::Int(5))),
        Box::new(Expr::Lit(Literal::Int(3))),
    );

    let result = type_check(expr);
    assert!(result.is_none());
}

#[test]
fn type_check_invalid_binop() {
    let expr = Expr::BinOp(
        Op::Add,
        Box::new(Expr::Lit(Literal::Int(5))),
        Box::new(Expr::Lit(Literal::Bool(true))),
    );

    let result = type_check(expr);
    assert!(matches!(result, Some(TypeError::MisMatch(_))));
}

#[test]
fn type_check_simple_lambda() {
    // \(x: Int) -> x + 5
    let expr = Expr::Lambda(
        "x".to_string(),
        Some(TypeAnn::Int),
        Box::new(Expr::BinOp(
            Op::Add,
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Lit(Literal::Int(5))),
        )),
    );

    let result = type_check(expr);
    assert!(result.is_none());
}

#[test]
fn type_check_invalid_simple_lambda() {
    // \(x: Int) -> x && True
    let expr = Expr::Lambda(
        "x".to_string(),
        Some(TypeAnn::Int),
        Box::new(Expr::BinOp(
            Op::And,
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Lit(Literal::Bool(true))),
        )),
    );

    let result = type_check(expr);
    assert!(matches!(result, Some(TypeError::MisMatch(_))));
}

#[test]
fn type_check_nested_lambda() {
    // \(x: Int) -> \(y: Int) -> x + y
    let expr = Expr::Lambda(
        "x".to_string(),
        Some(TypeAnn::Int),
        Box::new(Expr::Lambda(
            "y".to_string(),
            Some(TypeAnn::Int),
            Box::new(Expr::BinOp(
                Op::Add,
                Box::new(Expr::Var("x".to_string())),
                Box::new(Expr::Var("y".to_string())),
            )),
        )),
    );

    let result = type_check(expr);
    assert!(result.is_none());
}