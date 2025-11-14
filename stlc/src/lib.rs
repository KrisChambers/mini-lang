use crate::{parser::parse_program, type_inference::{infer_type, Type}};

pub mod parser;
pub mod type_checker;
pub mod type_inference;

pub fn main(input: &str) -> Result<Type, String> {
    let (_, expr) = parse_program(input).map_err(|x| x.to_string())?;

    infer_type(&expr).map_err(|x| format!("{x:?}").to_string())
}
