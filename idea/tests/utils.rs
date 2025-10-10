use mini_lang::{type_check, types::{Type, TypeError}};

/// Test type checking on the provided input.
pub fn basic_test(input: &str, expected: Vec<Result<Type, TypeError>>) {
    let actual = type_check(input);

    assert_eq!(actual, expected);
}

