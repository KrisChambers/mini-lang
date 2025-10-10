mod utils;
use mini_lang::types::{Type, TypeError};

use utils::basic_test;


#[test]
fn non_bool_conditional() {
    // Error
    basic_test("
let x = if 1 then 2 else 10;
".trim(), vec![Err(TypeError::Expected("If", Type::Bool, Type::Int))]);

    // Correction
    basic_test("
let x = if 1 == 1 then 2 else 10;
".trim(), vec![Ok(Type::Int)]);
}


#[test]
fn invalid_lambda_parameter() {
    // Error
    basic_test("
let add2 = fn(x) { x + 2 };
let five = add2(true);
".trim(), vec![Ok(Type::Lambda(vec![Type::Int], Box::new(Type::Int))), Err(TypeError::Expected("Apply", Type::Int, Type::Bool))]);

    // Correction
    basic_test("
let add2 = fn(x) { x + 2 };
let five = add2(3);
".trim(), vec![Ok(Type::Lambda(vec![Type::Int], Box::new(Type::Int))), Ok(Type::Int)]);
}
