use stlc::parser::{self as parser, Expr, Literal, Op, TypeAnn};

#[test]
fn parsing_complex_arrow() {
    let input = "(Int -> Int) -> Int";
    let (_, actual) = stlc::parser::parse_type(input).unwrap();

    assert_eq!(
        actual,
        TypeAnn::Arrow(
            Box::new(TypeAnn::Arrow(
                Box::new(TypeAnn::Int),
                Box::new(TypeAnn::Int)
            )),
            Box::new(TypeAnn::Int)
        )
    );
}

#[test]
fn parse_simple_lambda() {
    let input = "\\(x: Int) -> x + 5";
    let expected = Expr::Lambda(
        "x".to_string(),
        Some(TypeAnn::Int),
        Box::new(Expr::BinOp(
            Op::Add,
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Lit(Literal::Int(5))),
        )),
    );
    let (_, actual) = stlc::parser::parse_lambda(input).unwrap();

    assert_eq!(actual, expected)
}

#[test]
fn parsing_lambda_without_type_annotation() {
    let input = r"\x -> x + 5";
    let expected = Expr::Lambda(
        "x".to_string(),
        None,
        Box::new(Expr::BinOp(
            Op::Add,
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Lit(Literal::Int(5))),
        )),
    );

    let (_, actual) = stlc::parser::parse_lambda(input).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn parse_simple_if() {
    let input = "if True then False else 1";
    let expected = Expr::If(
        Box::new(Expr::Lit(Literal::Bool(true))),
        Box::new(Expr::Lit(Literal::Bool(false))),
        Box::new(Expr::Lit(Literal::Int(1))),
    );

    let (_, actual) = stlc::parser::parse_if_expr(input).unwrap();

    assert_eq!(actual, expected)
}

#[test]
fn parse_lambda_definition_application() {
    let input = r"(\(x: Int) -> x + 2) 5";
    let expected = Expr::App(
        Box::new(Expr::Lambda(
            "x".to_string(),
            Some(TypeAnn::Int),
            Box::new(Expr::BinOp(
                Op::Add,
                Box::new(Expr::Var("x".to_string())),
                Box::new(Expr::Lit(Literal::Int(2))),
            )),
        )),
        Box::new(Expr::Lit(Literal::Int(5))),
    );

    let (_, actual) = stlc::parser::parse_application_expr(input).unwrap();

    assert_eq!(actual, expected);
}

#[test]
fn parse_multi_lambda() {
    let input = r"\(f: Int -> Int) -> \(x: Int) -> f x";
    let expected = Expr::Lambda(
        "f".to_string(),
        Some(TypeAnn::Arrow(
            Box::new(TypeAnn::Int),
            Box::new(TypeAnn::Int),
        )),
        Box::new(Expr::Lambda(
            "x".to_string(),
            Some(TypeAnn::Int),
            Box::new(Expr::App(
                Box::new(Expr::Var("f".to_string())),
                Box::new(Expr::Var("x".to_string())),
            )),
        )),
    );

    let (_, actual) = stlc::parser::parse_expr(input).unwrap();

    assert_eq!(actual, expected);
}

#[test]
fn parse_lambda_var_application() {
    let input = "add2 5";
    let expected = Expr::App(
        Box::new(Expr::Var("add2".to_string())),
        Box::new(Expr::Lit(Literal::Int(5))),
    );

    let (_, actual) = stlc::parser::parse_application_expr(input).unwrap();

    assert_eq!(actual, expected);
}

#[test]
fn parse_simple_addition() {
    let input = "5 + 3";
    let expected = Expr::BinOp(
        Op::Add,
        Box::new(Expr::Lit(Literal::Int(5))),
        Box::new(Expr::Lit(Literal::Int(3))),
    );
    let (_, actual) = stlc::parser::parse_binary_expr(input).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn parse_boolean_and() {
    let input = "True && False";
    let expected = Expr::BinOp(
        Op::And,
        Box::new(Expr::Lit(Literal::Bool(true))),
        Box::new(Expr::Lit(Literal::Bool(false))),
    );
    let (_, actual) = stlc::parser::parse_binary_expr(input).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn parse_boolean_or() {
    let input = "False || True";
    let expected = Expr::BinOp(
        Op::Or,
        Box::new(Expr::Lit(Literal::Bool(false))),
        Box::new(Expr::Lit(Literal::Bool(true))),
    );
    let (_, actual) = stlc::parser::parse_binary_expr(input).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn parse_nested_binary_ops() {
    let input = "(5 + 3) + 2";
    let expected = Expr::BinOp(
        Op::Add,
        Box::new(Expr::BinOp(
            Op::Add,
            Box::new(Expr::Lit(Literal::Int(5))),
            Box::new(Expr::Lit(Literal::Int(3))),
        )),
        Box::new(Expr::Lit(Literal::Int(2))),
    );
    let (_, actual) = stlc::parser::parse_binary_expr(input).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn parse_right_associative_arrow_types() {
    let input = "Int -> Int -> Int";
    let expected = TypeAnn::Arrow(
        Box::new(TypeAnn::Int),
        Box::new(TypeAnn::Arrow(
            Box::new(TypeAnn::Int),
            Box::new(TypeAnn::Int),
        )),
    );
    let (_, actual) = stlc::parser::parse_type(input).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn parse_bool_to_bool_arrow() {
    let input = "Bool -> Bool";
    let expected = TypeAnn::Arrow(Box::new(TypeAnn::Bool), Box::new(TypeAnn::Bool));
    let (_, actual) = stlc::parser::parse_type(input).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn parse_nested_if_expr() {
    let input = "if True then if False then 1 else 2 else 3";
    let expected = Expr::If(
        Box::new(Expr::Lit(Literal::Bool(true))),
        Box::new(Expr::If(
            Box::new(Expr::Lit(Literal::Bool(false))),
            Box::new(Expr::Lit(Literal::Int(1))),
            Box::new(Expr::Lit(Literal::Int(2))),
        )),
        Box::new(Expr::Lit(Literal::Int(3))),
    );
    let (_, actual) = stlc::parser::parse_if_expr(input).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn parse_if_with_binary_condition() {
    let input = "if 5 + 3 then True else False";
    let expected = Expr::If(
        Box::new(Expr::BinOp(
            Op::Add,
            Box::new(Expr::Lit(Literal::Int(5))),
            Box::new(Expr::Lit(Literal::Int(3))),
        )),
        Box::new(Expr::Lit(Literal::Bool(true))),
        Box::new(Expr::Lit(Literal::Bool(false))),
    );
    let (_, actual) = stlc::parser::parse_if_expr(input).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn parse_application_with_literal_arg() {
    let input = "myFunc True";
    let expected = Expr::App(
        Box::new(Expr::Var("myFunc".to_string())),
        Box::new(Expr::Lit(Literal::Bool(true))),
    );
    let (_, actual) = stlc::parser::parse_application_expr(input).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn parse_lambda_with_bool_param() {
    let input = r"\(b: Bool) -> b";
    let expected = Expr::Lambda(
        "b".to_string(),
        Some(TypeAnn::Bool),
        Box::new(Expr::Var("b".to_string())),
    );
    let (_, actual) = stlc::parser::parse_lambda(input).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn parse_lambda_with_function_type() {
    let input = r"\(f: Int -> Bool) -> f";
    let expected = Expr::Lambda(
        "f".to_string(),
        Some(TypeAnn::Arrow(
            Box::new(TypeAnn::Int),
            Box::new(TypeAnn::Bool),
        )),
        Box::new(Expr::Var("f".to_string())),
    );
    let (_, actual) = stlc::parser::parse_lambda(input).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn parse_lambda_returning_if() {
    let input = r"\(x: Int) -> if True then x else 0";
    let expected = Expr::Lambda(
        "x".to_string(),
        Some(TypeAnn::Int),
        Box::new(Expr::If(
            Box::new(Expr::Lit(Literal::Bool(true))),
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Lit(Literal::Int(0))),
        )),
    );
    let (_, actual) = stlc::parser::parse_lambda(input).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn parse_if_with_lambda_in_branch() {
    let input = r"if True then \(x: Int) -> x else \(y: Int) -> y + 1";
    let expected = Expr::If(
        Box::new(Expr::Lit(Literal::Bool(true))),
        Box::new(Expr::Lambda(
            "x".to_string(),
            Some(TypeAnn::Int),
            Box::new(Expr::Var("x".to_string())),
        )),
        Box::new(Expr::Lambda(
            "y".to_string(),
            Some(TypeAnn::Int),
            Box::new(Expr::BinOp(
                Op::Add,
                Box::new(Expr::Var("y".to_string())),
                Box::new(Expr::Lit(Literal::Int(1))),
            )),
        )),
    );
    let (_, actual) = stlc::parser::parse_if_expr(input).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn parse_subtraction() {
    let input = "10 - 5";
    let expected = Expr::BinOp(
        Op::Subtract,
        Box::new(Expr::Lit(Literal::Int(10))),
        Box::new(Expr::Lit(Literal::Int(5))),
    );
    let (_, actual) = stlc::parser::parse_binary_expr(input).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn parse_variable_in_binary_op() {
    let input = "x + 5";
    let expected = Expr::BinOp(
        Op::Add,
        Box::new(Expr::Var("x".to_string())),
        Box::new(Expr::Lit(Literal::Int(5))),
    );
    let (_, actual) = stlc::parser::parse_binary_expr(input).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn parse_simple_let_assignment() {
    let input = "x = 5";
    let expected = ("x".to_string(), Expr::Lit(Literal::Int(5)));

    let (_, actual) = stlc::parser::parse_let_assignment(input).unwrap();

    assert_eq!(actual, expected);
}

#[test]
fn parse_fn_let_assignemnt() {
    let input = r"x = \y -> y + 2";
    let lambda = Expr::Lambda(
        "y".to_string(),
        None,
        Box::new(Expr::BinOp(
            Op::Add,
            Box::new(Expr::Var("y".to_string())),
            Box::new(Expr::Lit(Literal::Int(2))),
        )),
    );
    let expected = ("x".to_string(), lambda);
    let (_, actual) = stlc::parser::parse_let_assignment(input).unwrap();

    assert_eq!(actual, expected);
}

#[test]
fn parse_let_fn() {
    let input = r"let x = \y -> y + 2 in x 3";

    let expected = Expr::Let(
        "x".to_string(),
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
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Lit(Literal::Int(3))),
        )),
    );
    let (_, actual) = stlc::parser::parse_let_expr(input).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn parse_simple_let_expr() {
    let input = "let x = 5 in x + 2";
    let expected = Expr::Let(
        "x".to_string(),
        Box::new(Expr::Lit(Literal::Int(5))),
        Box::new(Expr::BinOp(
            Op::Add,
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Lit(Literal::Int(2))),
        )),
    );
    let (_, actual) = stlc::parser::parse_let_expr(input).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn parse_nested_let_expr() {
    let input = "let x = 5 in let y = 2 in x + y";
    let expected = Expr::Let(
        "x".to_string(),
        Box::new(Expr::Lit(Literal::Int(5))),
        Box::new(Expr::Let(
            "y".to_string(),
            Box::new(Expr::Lit(Literal::Int(2))),
            Box::new(Expr::BinOp(
                Op::Add,
                Box::new(Expr::Var("x".to_string())),
                Box::new(Expr::Var("y".to_string())),
            )),
        )),
    );
    let (_, actual) = stlc::parser::parse_let_expr(input).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn parse_multi_app() {
    let input = "eval f 3".trim();

    let expected = Expr::App(
        Box::new(
            Expr::App(
                Box::new(Expr::Var("eval".to_string())),
                Box::new(Expr::Var("f".to_string()))
            )
        ),
        Box::new(Expr::Lit(Literal::Int(3)))
    );

    let (_, result) = parser::parse_application_expr(input).unwrap();
    assert_eq!(result, expected);

}

#[test]
fn parse_nested_let_expr_2() {
    let input = r"
let doStuff =
    let hour = 2 in
    let min = 3 in
    let add = \x -> \y -> x + y in add hour min
    in
    doStuff
"
    .trim();

    let expected = Expr::Let(
        "doStuff".to_string(),
        Box::new(Expr::Let(
            "hour".to_string(),
            Box::new(Expr::Lit(Literal::Int(2))),
            Box::new(Expr::Let(
                "min".to_string(),
                Box::new(Expr::Lit(Literal::Int(3))),
                Box::new(Expr::Let(
                    "add".to_string(),
                    Box::new(Expr::Lambda(
                        "x".to_string(),
                        None,
                        Box::new(Expr::Lambda(
                            "y".to_string(),
                            None,
                            Box::new(Expr::BinOp(
                                Op::Add,
                                Box::new(Expr::Var("x".to_string())),
                                Box::new(Expr::Var("y".to_string())),
                            )),
                        )),
                    )),
                    Box::new(Expr::App(
                        Box::new(Expr::App(
                            Box::new(Expr::Var("add".to_string())),
                            Box::new(Expr::Var("hour".to_string()))
                        )),
                        Box::new(Expr::Var("min".to_string()))
                    )),
                )),
            ))
        )),
        Box::new(Expr::Var("doStuff".to_string())),
    );

    let (_, actual) = parser::parse_let_expr(input).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn parse_let_with_lambda_and_application() {
    let input = r"let id = \x -> x in id 5";
    let expected = Expr::Let(
        "id".to_string(),
        Box::new(Expr::Lambda(
            "x".to_string(),
            None,
            Box::new(Expr::Var("x".to_string())),
        )),
        Box::new(Expr::App(
            Box::new(Expr::Var("id".to_string())),
            Box::new(Expr::Lit(Literal::Int(5))),
        )),
    );
    let (_, actual) = stlc::parser::parse_let_expr(input).unwrap();
    assert_eq!(actual, expected);
}
