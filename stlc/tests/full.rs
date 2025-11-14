use stlc::parser::{Expr, Literal, Op, parse, parse_program};
use stlc::type_inference::{Type, infer_type};

#[test]
fn binary_desugar() {
    let input = r"
let x = 3 + 2 + 4
".trim();


    let (_, result) = parse_program(input).unwrap();
    let t = infer_type(&result).unwrap();

    assert_eq!(t, Type::Int);
}

#[test]
fn simple_program() {
    let input = r"
let x = 3 in
    let add2 = \y -> y + 2
    in
        add2 x

"
    .trim();

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
    let r = infer_type(expr);
    assert!(r.is_ok());

    let t = r.unwrap();
    assert_eq!(t, Type::Int);
}

#[test]
fn simple_top_level_let_sugar() {
    let input = r"
let doSomething =
    let
        x = 2,
        y = 3
    in
        x + y
"
    .trim();

    let sub_let = "
let
    x = 2,
    y = 3
in
    x + y
"
    .trim();
    let sub_let = {
        let (_, parsed) = parse(sub_let).unwrap();
        assert_eq!(parsed.len(), 1);

        parsed[0].clone()
    };

    let expected = Expr::Let(
        "doSomething".to_string(),
        Box::new(sub_let),
        Box::new(Expr::Var("doSomething".to_string())),
    );

    let (_, result) = parse_program(input).unwrap();

    assert_eq!(expected, result);

    let t = infer_type(&result).expect("Should be a type");

    assert_eq!(t, Type::Int);
}

#[test]
fn identity_fn_type() {
    let input = r"
let id = \x -> x;
    "
    .trim();

    let expected = Expr::Let(
        "id".to_string(),
        Box::new(Expr::Lambda(
            "x".to_string(),
            None,
            Box::new(Expr::Var("x".to_string())),
        )),
        Box::new(Expr::Var("id".to_string())),
    );

    let (_, result) = parse_program(input).unwrap();

    assert_eq!(result, expected);

    let t = infer_type(&result).expect("Expect Type");
    let t_var = Type::Var("v1".to_string());

    assert_eq!(
        t,
        Type::Scheme(
            ["v1".to_string()].into(),
            Box::new(Type::Arrow(
                Box::new(t_var.clone()),
                Box::new(t_var.clone())
            ))
        )
    );
}

fn identity_application() {
    let input = r"
let v = let id = \x -> x in id 5
    "
    .trim();

    let (_, expr) = parse_program(input).unwrap();

    let t = infer_type(&expr).expect("Expect Type");

    assert_eq!(t, Type::Int);
}

#[test]
fn multiple_identity_uses() {
    let input = r"
let id = \x -> x;
let i = id 5;
let b = id True
    "
    .trim();

    /*
        let id = /x -> x in
            let i = id 5 in
                let b = id True in ...

    */

    let (_, expr) = parse_program(input).unwrap();

    // The whole expression should type-check to Bool (the type of the final variable b)
    let t = infer_type(&expr).expect("Type");
    assert_eq!(t, Type::Bool);
}

// This is almost done. Just some small kinks
#[test]
fn polymorphic_composition() {
    use Type::*;
    let input = r"
let compose = \f -> \g -> \x -> f (g x)

let inc = \n -> n + 1

let not = \b -> if b then False else True

let f = compose inc inc

let g = compose not not
    "
    .trim();

    // Now we check the types

    let get_type = |definitions: &str, var_name: &str| {
        let (_, expr) = parse_program(
            &(definitions.to_string() + "\n\n" + format!("let result = {var_name}").as_str()),
        )
        .unwrap();

        infer_type(&expr).expect("Typing Failure")
    };

    let compose_type = Scheme(
        ["v4", "v5", "v3"]
            .iter()
            .map(|x| x.to_string())
            .collect(),
        Box::new(Scheme(
            ["v4", "v5", "v3"].iter().map(|x| x.to_string()).collect(),
            Box::new(Arrow(
                Box::new(Arrow(
                    Box::new(Var("v4".to_string())),
                    Box::new(Var("v5".to_string())),
                )),
                Box::new(Arrow(
                    Box::new(Arrow(
                        Box::new(Var("v3".to_string())),
                        Box::new(Var("v4".to_string())),
                    )),
                    Box::new(Arrow(
                        Box::new(Var("v3".to_string())),
                        Box::new(Var("v5".to_string())),
                    )),
                )),
            )),
        )),
    );

    assert_eq!(
        get_type(input, "f"),
        Type::Arrow(Box::new(Type::Int), Box::new(Type::Int))
    );
    assert_eq!(
        get_type(input, "g"),
        Type::Arrow(Box::new(Type::Bool), Box::new(Type::Bool))
    );
    assert_eq!(
        get_type(input, "not"),
        Type::Arrow(Box::new(Type::Bool), Box::new(Type::Bool))
    );
    assert_eq!(
        get_type(input, "inc"),
        Type::Arrow(Box::new(Type::Int), Box::new(Type::Int))
    );
}
