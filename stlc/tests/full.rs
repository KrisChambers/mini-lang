use stlc::parser::{Expr, parse, parse_program};
use stlc::type_inference::{Type, infer_type};

#[test]
fn binary_desugar() {
    let input = r"let x = 3 + 2 + 4 ".trim();

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
    let t_vars = match t.clone() {
        Type::Scheme(vars, _) => vars.clone(),
        _ => panic!("Expected a Type Scheme"),
    };

    let v = Type::Var(t_vars.clone().into_iter().take(1).collect::<Vec<String>>()[0].clone());

    let expected = Type::Scheme(
        t_vars,
        Box::new(Type::Arrow(Box::new(v.clone()), Box::new(v.clone()))),
    );

    assert_eq!(t, expected);
}

#[test]
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

#[test]
fn polymorphic_composition() {
    let input = r"
let compose = \f -> \g -> \x -> f (g x)

let inc = \n -> n + 1

let not = \b -> if b then False else True

let f = compose inc inc

let g = compose not not
    "
    .trim();

    let get_type = |definitions: &str, var_name: &str| {
        let (_, expr) = parse_program(
            &(definitions.to_string() + "\n\n" + format!("let result = {var_name}").as_str()),
        )
        .unwrap();

        infer_type(&expr).expect("Typing Failure")
    };

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
