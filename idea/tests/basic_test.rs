use mini_lang::type_check;
use mini_lang::types::{Type, TypeError};

fn basic_test(input: &str, expected: Vec<Result<Type, TypeError>>) {
    let actual = type_check(input);

    assert_eq!(actual, expected);
}

#[test]
fn basic_int() {
    basic_test("let x = 1;", vec![Ok(Type::Int)]);
}

#[test]
fn basic_binary_int_var() {
    basic_test("let a = 1 + 2;", vec![Ok(Type::Int)]);
}

#[test]
fn basic_if_statement() {
    basic_test("let x = if true then 1 else 2;", vec![Ok(Type::Int)]);
}

#[test]
fn basic_if_statement_fail() {
    basic_test(
        "let x = if true then true else 2;",
        vec![Err(TypeError::Expected("If", Type::Bool, Type::Int))],
    );
}

#[test]
fn basic_lambda_constant() {
    basic_test(
        "let const2 = fn(x) { 2 };",
        vec![Ok(Type::Lambda(vec![Type::Unknown], Box::new(Type::Int)))],
    );
}

#[test]
fn basic_lambda_variable_inference() {
    basic_test(
        "let add2 = fn(x) { x + 2 };",
        vec![Ok(Type::Lambda(vec![Type::Int], Box::new(Type::Int)))],
    );
}

#[test]
fn basic_function_application() {
    basic_test(
        "
let add2 = fn(x) { x + 2 };
let result = add2(3);
"
        .trim(),
        vec![
            Ok(Type::Lambda(vec![Type::Int], Box::new(Type::Int))),
            Ok(Type::Int),
        ],
    );
}
