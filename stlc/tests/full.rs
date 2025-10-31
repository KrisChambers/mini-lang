use stlc::parser::{parse, Expr, Literal, Op};
use stlc::type_inference::{infer_type, Type};

#[test]
fn simple_program() {
    let input = r"
let x = 3 in
    let add2 = \y -> y + 2
    in
        add2 x

".trim();

    let expected = Expr::Let(
        "x".to_string(),
        Box::new(Expr::Lit(Literal::Int(3))),
        Box::new(Expr::Let(
            "add2".to_string(),
            Box::new(Expr::Lambda(
                "y".to_string(),
                None,
                Box::new(Expr::BinOp(
                    Op::Add,
                    Box::new(Expr::Var("y".to_string())),
                    Box::new(Expr::Lit(Literal::Int(2))),
                )),
            )),
            Box::new(Expr::App(
                Box::new(Expr::Var("add2".to_string())),
                Box::new(Expr::Var("x".to_string())),
            )),
        )),
    );

    let ast = parse(input);
    assert!(ast.is_ok());
    let (_, exprs) = ast.unwrap();
    assert_eq!(exprs.len(), 1);

    let expr = exprs.first().unwrap();
    let r  = infer_type(expr);
    assert!(r.is_ok());

    let t = r.unwrap();
    assert_eq!(t, Type::Int);
}
