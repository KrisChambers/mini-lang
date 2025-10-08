use std::collections::HashMap;

use crate::ast::{BinaryOperator, Expr};

#[derive(PartialEq, Eq, Debug, Clone)]
pub enum Type {
    Int,
    Bool,
    Lambda(Vec<Type>, Box<Type>),
    Unknown,
}

#[derive(PartialEq, Eq, Debug, Clone)]
pub enum TypeError {
    InvalidType(String),
    UnknownVariableType(String),
    Expected(Type),
}

struct TypeBindings {
    bindings: HashMap<String, Type>,
}

impl TypeBindings {
    fn new() -> Self {
        Self {
            bindings: HashMap::new(),
        }
    }

    fn get(&self, name: &str) -> Type {
        if let Some(t) = self.bindings.get(name).cloned() {
            t
        } else {
            Type::Unknown
        }
    }

    fn insert(&mut self, name: &str, typ: Type) {
        if self.bindings.insert(name.to_string(), typ).is_some() {
            panic!("Overwriting exisiting type information for {}", name);
        }
    }
}

pub struct NaiveTypeChecker {}

impl NaiveTypeChecker {
    pub fn type_check(expr: Expr) -> Result<Type, TypeError> {
        let mut bindings = TypeBindings::new();

        inner_type_check(expr, None, &mut bindings)
    }

    pub fn type_check_program(expr: Vec<Expr>) -> Vec<Result<Type, TypeError>> {
        let mut bindings = TypeBindings::new();

        expr.iter()
            .map(|x| inner_type_check(x.clone(), None, &mut bindings))
            .collect()
    }
}

fn expect(a: Result<Type, TypeError>, typ: Type) -> Result<Type, TypeError> {
    a.and_then(|x| {
        if x == typ {
            Ok(x)
        } else {
            Err(TypeError::InvalidType(String::new()))
        }
    })
}

fn inner_type_check(
    expr: Expr,
    expected: Option<&Type>,
    bindings: &mut TypeBindings,
) -> Result<Type, TypeError> {
    match expr {
        Expr::Int(_) => Ok(Type::Int),
        Expr::Bool(_) => Ok(Type::Bool),
        Expr::Var(name) => {
            let actual = bindings.get(&name);

            if let Some(expected_type) = expected {
                let actual = if actual != Type::Unknown {
                    actual
                } else {
                    bindings.insert(&name, expected_type.clone());
                    expected_type.clone()
                };

                if actual == *expected_type {
                    Ok(expected_type.clone())
                } else {
                    Err(TypeError::Expected(expected_type.clone()))
                }
            } else {
                Ok(bindings.get(&name))
            }
        }
        Expr::Let(name, expr) => {
            let expr_type = inner_type_check(*expr, None, bindings)?;
            bindings.insert(&name, expr_type.clone());
            Ok(expr_type)
        }
        Expr::If(expr, expr1, expr2) => {
            let cond_type = inner_type_check(*expr, None, bindings)?;
            let true_branch_type = inner_type_check(*expr1, None, bindings)?;
            let false_branch_type = inner_type_check(*expr2, None, bindings)?;

            if cond_type != Type::Bool {
                Err(TypeError::Expected(Type::Bool))
            } else if true_branch_type != false_branch_type {
                Err(TypeError::Expected(true_branch_type))
            } else {
                Ok(true_branch_type)
            }
        }
        Expr::Lambda(exprs, expr) => {
            let body_type = inner_type_check(*expr, None, bindings)?;
            let param_types = exprs
                .iter()
                .map(|x| inner_type_check(x.clone(), None, bindings))
                .try_fold(vec![], |mut acc, e| match e {
                    Ok(t) => {
                        acc.push(t.clone());
                        Ok(acc)
                    }
                    Err(t) => Err(t),
                })?;

            Ok(Type::Lambda(param_types, Box::new(body_type)))
        }
        Expr::Apply(expr, exprs) => {
            let lambda_type = inner_type_check(*expr, None, bindings)?;
            println!("lambda_type: {:?}", lambda_type);
            let arg_types = exprs
                .iter()
                .map(|x| inner_type_check(x.clone(), None, bindings))
                .try_fold(vec![], |mut acc, e| match e {
                    Ok(t) => {
                        acc.push(t.clone());
                        Ok(acc)
                    }
                    Err(t) => Err(t),
                })?;

            if let Type::Lambda(param_types, body_type) = lambda_type {
                let param_types = param_types.iter();
                let arg_types = arg_types.iter();

                for (param, arg) in param_types.zip(arg_types) {
                    if param != arg {
                        return Err(TypeError::Expected(param.clone()));
                    }
                }

                Ok(*body_type.clone())
            } else {
                Err(TypeError::InvalidType("Lambda type error".to_string()))
            }
        }
        Expr::Binary(binary_operator, left, right) => {
            let expected = match binary_operator {
                Add | Subtract | Multiply | Divide => Type::Int,
                And | Or => Type::Bool,
                Equals => Type::Unknown,
            };

            let tleft = inner_type_check(*left, Some(&expected), bindings);
            let tright = inner_type_check(*right, Some(&expected), bindings);

            use BinaryOperator::*;
            match binary_operator {
                Add | Subtract | Multiply | Divide => match tleft.and(tright) {
                    Ok(x) => {
                        if x == Type::Int {
                            Ok(x)
                        } else {
                            Err(TypeError::InvalidType(String::from("")))
                        }
                    }
                    x => x,
                },
                And | Or => match tleft.and(tright) {
                    Ok(x) => {
                        if x == Type::Int {
                            Ok(x)
                        } else {
                            Err(TypeError::InvalidType(String::from("")))
                        }
                    }
                    x => x,
                },
                Equals => Ok(Type::Bool),
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::parser::parse;

    #[test]
    fn basic_check_int() {
        let actual = NaiveTypeChecker::type_check(Expr::Int(1));
        let expected = Ok(Type::Int);

        assert_eq!(actual, expected)
    }

    #[test]
    fn basic_check_bool() {
        let actual = NaiveTypeChecker::type_check(Expr::Bool(true));
        let expected = Ok(Type::Bool);

        assert_eq!(actual, expected)
    }

    #[test]
    fn basic_bool_var() {
        let leta = Expr::Let("a".to_string(), Box::new(Expr::Bool(true)));
        let actual = NaiveTypeChecker::type_check(leta);
        let expected = Ok(Type::Bool);

        assert_eq!(actual, expected)
    }

    #[test]
    fn basic_int_var() {
        let leta = Expr::Let("a".to_string(), Box::new(Expr::Int(0)));
        let actual = NaiveTypeChecker::type_check(leta);
        let expected = Ok(Type::Int);

        assert_eq!(actual, expected)
    }

    #[test]
    fn basic_binary_int_var() {
        let input = "let a = 1 + (2 + 10);";
        let leta = parse(input)[0].clone();
        let actual = NaiveTypeChecker::type_check(leta);
        let expected = Ok(Type::Int);

        assert_eq!(actual, expected)
    }

    #[test]
    fn basic_if_statement() {
        let input = "let x = if true then 1 else 2;";
        let leta = parse(input)[0].clone();
        let actual = NaiveTypeChecker::type_check(leta);
        let expected = Ok(Type::Int);

        assert_eq!(actual, expected)
    }

    #[test]
    fn basic_if_statement_fail() {
        let input = "let x = if true then true else 2;";
        let leta = parse(input)[0].clone();
        let actual = NaiveTypeChecker::type_check(leta);
        let expected = Err(TypeError::Expected(Type::Bool));

        assert_eq!(actual, expected)
    }

    #[test]
    fn basic_lambda_constant() {
        let input = "let const2 = fn(x) { 2 };";
        let parsed = parse(input)[0].clone();
        let actual = NaiveTypeChecker::type_check(parsed);
        let expected = Ok(Type::Lambda(vec![Type::Unknown], Box::new(Type::Int)));

        assert_eq!(actual, expected)
    }

    #[test]
    fn basic_lambda_variable_inference() {
        let input = "let add2 = fn(x) { x + 2 };";
        let parsed = parse(input)[0].clone();
        let actual = NaiveTypeChecker::type_check(parsed);
        let expected = Ok(Type::Lambda(vec![Type::Int], Box::new(Type::Int)));

        assert_eq!(actual, expected)
    }

    #[test]
    fn basic_function_application() {
        let input = "
let add2 = fn(x) { x + 2 };
let result = add2(3);
"
        .trim();
        let parsed = parse(input);
        let results = NaiveTypeChecker::type_check_program(parsed);

        let lambda_expect = Ok(Type::Lambda(vec![Type::Int], Box::new(Type::Int)));
        assert_eq!(results[0], lambda_expect);
        assert_eq!(results[1], Ok(Type::Int))
    }
}
