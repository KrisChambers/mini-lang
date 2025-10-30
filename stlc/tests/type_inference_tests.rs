use stlc::parser::{Expr, Literal, TypeAnn};
use stlc::type_inference::{Type, w, TypeEnv};

#[test]
fn infer_identity_function() {
    // \x -> x should infer a polymorphic type v0 -> v0
    let expr = Expr::Lambda("x".to_string(), None, Box::new(Expr::Var("x".to_string())));
    let env = TypeEnv::new();
    let result = w(expr, env);

    assert!(result.is_ok());
    let (sub, inferred_type) = result.unwrap();
    // Should be v0 -> v0 (arrow from fresh var to same var)
    match inferred_type {
        Type::Arrow(param, ret) => assert_eq!(param, ret),
        _ => panic!("Expected arrow type"),
    }
}

#[test]
fn infer_typed_lambda_application() {
    // (\(x: Int) -> x) 5 should infer Int
    let expr = Expr::App(
        Box::new(Expr::Lambda(
            "x".to_string(),
            Some(TypeAnn::Int),
            Box::new(Expr::Var("x".to_string())),
        )),
        Box::new(Expr::Lit(Literal::Int(5))),
    );
    let env = TypeEnv::new();
    let result = w(expr, env);

    assert!(result.is_ok());
    let (_, inferred_type) = result.unwrap();
    assert_eq!(inferred_type, Type::Int);
}

#[test]
fn infer_nested_lambda() {
    // \x -> \y -> y should infer v0 -> v1 -> v1
    let expr = Expr::Lambda(
        "x".to_string(),
        None,
        Box::new(Expr::Lambda(
            "y".to_string(),
            None,
            Box::new(Expr::Var("y".to_string())),
        )),
    );
    let env = TypeEnv::new();
    let result = w(expr, env);

    assert!(result.is_ok());
    let (_, inferred_type) = result.unwrap();
    // Should be v0 -> (v1 -> v1)
    match inferred_type {
        Type::Arrow(_, inner) => match *inner {
            Type::Arrow(p, r) => assert_eq!(p, r),
            _ => panic!("Expected nested arrow"),
        },
        _ => panic!("Expected arrow type"),
    }
}

#[test]
fn test_infer_simple_if_with_int_branches() {
    // if True then 1 else 2 should infer Int
    let expr = Expr::If(
        Box::new(Expr::Lit(Literal::Bool(true))),
        Box::new(Expr::Lit(Literal::Int(1))),
        Box::new(Expr::Lit(Literal::Int(2))),
    );
    let env = TypeEnv::new();
    let result = w(expr, env);

    assert!(result.is_ok());
    let (_, inferred_type) = result.unwrap();
    assert_eq!(inferred_type, Type::Int);
}

#[test]
fn test_infer_if_with_bool_branches() {
    // if True then True else False should infer Bool
    let expr = Expr::If(
        Box::new(Expr::Lit(Literal::Bool(true))),
        Box::new(Expr::Lit(Literal::Bool(true))),
        Box::new(Expr::Lit(Literal::Bool(false))),
    );
    let env = TypeEnv::new();
    let result = w(expr, env);

    assert!(result.is_ok());
    let (_, inferred_type) = result.unwrap();
    assert_eq!(inferred_type, Type::Bool);
}

#[test]
fn test_infer_if_with_type_mismatch() {
    // if True then 1 else False should fail (cannot unify Int and Bool)
    let expr = Expr::If(
        Box::new(Expr::Lit(Literal::Bool(true))),
        Box::new(Expr::Lit(Literal::Int(1))),
        Box::new(Expr::Lit(Literal::Bool(false))),
    );
    let env = TypeEnv::new();
    let result = w(expr, env);

    assert!(result.is_err());
}

#[test]
fn test_infer_if_with_polymorphic_branches() {
    // if True then (\x -> x) else (\y -> y) should infer v0 -> v0
    let expr = Expr::If(
        Box::new(Expr::Lit(Literal::Bool(true))),
        Box::new(Expr::Lambda(
            "x".to_string(),
            None,
            Box::new(Expr::Var("x".to_string())),
        )),
        Box::new(Expr::Lambda(
            "y".to_string(),
            None,
            Box::new(Expr::Var("y".to_string())),
        )),
    );
    let env = TypeEnv::new();
    let result = w(expr, env);

    assert!(result.is_ok());
    let (_, inferred_type) = result.unwrap();
    // Should be polymorphic: v0 -> v0 (or similar)
    match inferred_type {
        Type::Arrow(param, ret) => {
            // Both parameter and return type should be the same type variable
            assert_eq!(param, ret);
        }
        _ => panic!("Expected arrow type"),
    }
}

#[test]
fn test_infer_simple_let_binding_with_int() {
    // let x = 5 in x should infer Int
    let expr = Expr::Let(
        "x".to_string(),
        Box::new(Expr::Lit(Literal::Int(5))),
        Box::new(Expr::Var("x".to_string())),
    );
    let env = TypeEnv::new();
    let result = w(expr, env);

    assert!(result.is_ok());
    let (_, inferred_type) = result.unwrap();
    assert_eq!(inferred_type, Type::Int);
}

#[test]
fn test_infer_simple_let_binding_with_bool() {
    // let b = True in b should infer Bool
    let expr = Expr::Let(
        "b".to_string(),
        Box::new(Expr::Lit(Literal::Bool(true))),
        Box::new(Expr::Var("b".to_string())),
    );
    let env = TypeEnv::new();
    let result = w(expr, env);

    assert!(result.is_ok());
    let (_, inferred_type) = result.unwrap();
    assert_eq!(inferred_type, Type::Bool);
}

#[test]
fn test_infer_let_binding_with_lambda() {
    // let f = \x -> x in f 5 should infer Int
    let expr = Expr::Let(
        "f".to_string(),
        Box::new(Expr::Lambda(
            "x".to_string(),
            Some(TypeAnn::Int),
            Box::new(Expr::Var("x".to_string())),
        )),
        Box::new(Expr::App(
            Box::new(Expr::Var("f".to_string())),
            Box::new(Expr::Lit(Literal::Int(5))),
        )),
    );
    let env = TypeEnv::new();
    let result = w(expr, env);

    assert!(result.is_ok());
    let (_, inferred_type) = result.unwrap();
    assert_eq!(inferred_type, Type::Int);
}

#[test]
fn test_infer_nested_let_bindings() {
    // let x = 5 in let y = 10 in y should infer Int
    let expr = Expr::Let(
        "x".to_string(),
        Box::new(Expr::Lit(Literal::Int(5))),
        Box::new(Expr::Let(
            "y".to_string(),
            Box::new(Expr::Lit(Literal::Int(10))),
            Box::new(Expr::Var("y".to_string())),
        )),
    );
    let env = TypeEnv::new();
    let result = w(expr, env);

    assert!(result.is_ok());
    let (_, inferred_type) = result.unwrap();
    assert_eq!(inferred_type, Type::Int);
}

#[test]
fn test_infer_let_binding_with_shadowing() {
    // let x = 5 in let x = True in x should infer Bool
    let expr = Expr::Let(
        "x".to_string(),
        Box::new(Expr::Lit(Literal::Int(5))),
        Box::new(Expr::Let(
            "x".to_string(),
            Box::new(Expr::Lit(Literal::Bool(true))),
            Box::new(Expr::Var("x".to_string())),
        )),
    );
    let env = TypeEnv::new();
    let result = w(expr, env);

    assert!(result.is_ok());
    let (_, inferred_type) = result.unwrap();
    assert_eq!(inferred_type, Type::Bool);
}