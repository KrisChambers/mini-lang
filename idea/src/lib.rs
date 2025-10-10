mod ast;
mod eval;
pub mod parser;
pub mod types;

use types::{NaiveTypeChecker, TypeError, Type};
use parser::parse;

pub fn type_check(input: &str) -> Vec<Result<Type, TypeError>> {
    let ast = parse(input);
    NaiveTypeChecker::type_check_program(ast)
}
