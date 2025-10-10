use std::collections::HashMap;

use crate::ast::{BinaryOperator, Expr};

#[derive(PartialEq, Eq, Debug, Clone)]
pub enum Type {
    Int,
    Bool,
    Lambda(Vec<Type>, Box<Type>),
    Unknown,
    Var(usize)
}

#[derive(PartialEq, Eq, Debug, Clone)]
pub enum TypeError {
    /// InvalidType Context BadType
    InvalidType(String, Type),
    /// Expected a different Type: Context, ExpectedType, FoundType
    Expected(&'static str, Type, Type),
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
                    Err(TypeError::Expected("Variable", expected_type.clone(), actual.clone()))
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
                Err(TypeError::Expected("If", Type::Bool, cond_type.clone()))
            } else if true_branch_type != false_branch_type {
                Err(TypeError::Expected("If", true_branch_type, false_branch_type.clone()))
            } else {
                Ok(true_branch_type)
            }
        }
        Expr::Lambda(exprs, expr) => {
            println!("EXPR: {:?}", expr);
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
            println!("APPLY name: {:?}", expr.clone());
            println!("APPLY args: {:?}", exprs.clone());

            let lambda_type = inner_type_check(*expr, None, bindings)?;
            println!("APPLY lambda_type: {:?}", lambda_type);

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

            println!("APPLY args_type: {:?}", arg_types);

            if let Type::Lambda(param_types, body_type) = lambda_type {
                let param_types = param_types.iter();
                let arg_types = arg_types.iter();

                let something: Vec<_> = param_types.zip(arg_types)
                    .map( |(p, a)| match (p.clone(), a.clone()) {
                        (Type::Unknown, Type::Unknown) => Ok(Type::Unknown),
                        (Type::Unknown, _) => {
                            println!("arg: {:?}", a); Ok(a.clone())
                        },
                        (_, _) => if a != p { Err(TypeError::Expected("Apply", p.clone(), a.clone())) } else { Ok(p.clone()) }
                    }).filter_map(|x| x.err()).collect();

                if !something.is_empty() {
                    return Err(something[0].clone());
                };

                Ok(*body_type.clone())
            } else {
                Ok(Type::Unknown)
                //Err(TypeError::InvalidType("Lambda type error".to_string(), lambda_type.clone()))
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
                            Err(TypeError::InvalidType("Must be an Int".to_string(), x.clone()))
                        }
                    }
                    x => x,
                },
                And | Or => match tleft.and(tright) {
                    Ok(x) => {
                        if x == Type::Int {
                            Ok(x)
                        } else {
                            Err(TypeError::InvalidType("Must be a Bool".to_string(), x.clone()))
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
}
