use std::collections::HashMap;

use crate::parser::{Expr, Literal, Op, TypeAnn};

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone)]
pub enum Type {
    Int,
    Bool,
    Arrow(Box<Type>, Box<Type>),
}

impl From<TypeAnn> for Type {
    fn from(value: TypeAnn) -> Self {
        match value {
            TypeAnn::Int => Type::Int,
            TypeAnn::Bool => Type::Bool,
            TypeAnn::Arrow(type_ann, type_ann1) => {
                Type::Arrow(Box::new((*type_ann).into()), Box::new((*type_ann1).into()))
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TypeError {
    MisMatch(String),
    AlreadyDefined(String),
    MissingTypeInfoForVariable(String),
}

struct TypeEnvironment(HashMap<String, Type>);

impl TypeEnvironment {
    pub fn new() -> Self {
        TypeEnvironment(HashMap::new())
    }

    /// If the type exists then the return is the existing type
    pub fn add(&mut self, var_name: &str, var_type: &Type) -> Option<Type> {
        self.0.insert(var_name.to_string(), var_type.clone())
    }

    pub fn get(&self, var_name: &str) -> Option<&Type> {
        self.0.get(var_name)
    }

    pub fn remove(&mut self, var_name: &str) -> Option<Type> {
        self.0.remove(var_name)
    }
}

impl Default for TypeEnvironment {
    fn default() -> Self {
        Self::new()
    }
}

pub fn type_check(expr: Expr) -> Option<TypeError> {
    let mut env = TypeEnvironment::new();

    inner_type_check(expr, &mut env)
}

fn inner_type_check(expr: Expr, env: &mut TypeEnvironment) -> Option<TypeError> {
    match expr {
        Expr::Var(var_name) => match env.get(&var_name) {
            Some(_) => None,
            None => Some(TypeError::MissingTypeInfoForVariable(var_name)),
        },
        Expr::Lambda(var_name, type_ann, expr) => env
            .add(&var_name, &type_ann.into())
            .map(|_| TypeError::AlreadyDefined(var_name.clone()))
            // If the variable is fresh then we can add this to the context and check the
            // expression
            .or_else(|| inner_type_check(*expr, env))
            // If there was no type error then we just remove it from the context
            .or_else(|| {
                env.remove(&var_name);
                None
            }),
        Expr::App(f, arg) => {
            let f_type = match get_type(*f, env) {
                Some(t) => t,
                None => return Some(TypeError::MisMatch("Could not find function type".to_string()))
            };
            let arg_type = match get_type(*arg, env) {
                Some(t) => t,
                None => return Some(TypeError::MisMatch("Could not find argument type".to_string()))
            };

            match f_type {
                Type::Int => Some(TypeError::MisMatch("Int is not a valid function type".to_string())),
                Type::Bool => Some(TypeError::MisMatch("Bool is not a valid function type".to_string())),
                Type::Arrow(_source, target) => if *target != arg_type {
                    Some(TypeError::MisMatch("Function Type and argument type mismatch".to_string()))
                } else {
                    None
                }
            }
        },
        Expr::If(expr, true_expr, false_expr) => get_type(*expr, env)
            .and_then(|x| {
                if x != Type::Bool {
                    Some(TypeError::MisMatch("Expected type Bool".to_string()))
                } else {
                    None
                }
            })
            .or_else(|| {
                let true_type = match get_type(*true_expr, env) {
                    Some(t) => t,
                    None => {
                        return Some(TypeError::MisMatch(
                            "Could not determine type for true branch".to_string(),
                        ));
                    }
                };

                let false_type = match get_type(*false_expr, env) {
                    Some(t) => t,
                    None => {
                        return Some(TypeError::MisMatch(
                            "Could not determine type for false branch".to_string(),
                        ));
                    }
                };

                if true_type != false_type {
                    Some(TypeError::MisMatch(
                        "true and false branches must evaluate to the same type".to_string(),
                    ))
                } else {
                    None
                }
            }),
        Expr::Lit(_) => None, //
        Expr::BinOp(op, left, right) => {
            let expected = match op {
                Op::Add | Op::Subtract => Type::Int,
                Op::And | Op::Or => Type::Bool,
            };
            let left_type = get_type(*left, env)?;
            let right_type = get_type(*right, env)?;

            if left_type != expected || right_type != expected {
                Some(TypeError::MisMatch(
                    "Type mismatch in binary expression".to_string(),
                ))
            } else {
                None
            }
        }
    }
}

fn get_type(expr: Expr, env: &mut TypeEnvironment) -> Option<Type> {
    match expr {
        Expr::Var(var_name) => env.get(&var_name).cloned(),
        Expr::Lambda(_, type_ann, _body) => Some(type_ann.into()),
        Expr::App(_name, _arg) => todo!(),
        Expr::If(_cond, true_branch, _false_branch) => get_type(*true_branch, env),
        Expr::Lit(literal) => Some(match literal {
            Literal::Int(_) => Type::Int,
            Literal::Bool(_) => Type::Bool,
        }),
        Expr::BinOp(op, _, _) => Some(match op {
            Op::Add | Op::Subtract => Type::Int,
            Op::And | Op::Or => Type::Bool,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_check_valid_binop() {
        let expr = Expr::BinOp(
            Op::Add,
            Box::new(Expr::Lit(Literal::Int(5))),
            Box::new(Expr::Lit(Literal::Int(3))),
        );

        let result = type_check(expr);
        assert!(result.is_none());
    }

    #[test]
    fn type_check_invalid_binop() {
        let expr = Expr::BinOp(
            Op::Add,
            Box::new(Expr::Lit(Literal::Int(5))),
            Box::new(Expr::Lit(Literal::Bool(true))),
        );

        let result = type_check(expr);
        assert!(matches!(result, Some(TypeError::MisMatch(_))));
    }

    #[test]
    fn type_check_simple_lambda() {
        // \(x: Int) -> x + 5
        let expr = Expr::Lambda(
            "x".to_string(),
            TypeAnn::Int,
            Box::new(Expr::BinOp(
                Op::Add,
                Box::new(Expr::Var("x".to_string())),
                Box::new(Expr::Lit(Literal::Int(5))),
            )),
        );

        let result = type_check(expr);
        assert!(result.is_none());
    }

    #[test]
    fn type_check_invalid_simple_lambda() {
        // \(x: Int) -> x && True
        let expr = Expr::Lambda(
            "x".to_string(),
            TypeAnn::Int,
            Box::new(Expr::BinOp(
                Op::And,
                Box::new(Expr::Var("x".to_string())),
                Box::new(Expr::Lit(Literal::Bool(true))),
            )),
        );

        let result = type_check(expr);
        assert!(matches!(result, Some(TypeError::MisMatch(_))));
    }

    #[test]
    fn type_check_nested_lambda() {
        // \(x: Int) -> \(y: Int) -> x + y
        let expr = Expr::Lambda(
            "x".to_string(),
            TypeAnn::Int,
            Box::new(Expr::Lambda(
                "y".to_string(),
                TypeAnn::Int,
                Box::new(Expr::BinOp(
                    Op::Add,
                    Box::new(Expr::Var("x".to_string())),
                    Box::new(Expr::Var("y".to_string())),
                )),
            )),
        );

        let result = type_check(expr);
        assert!(result.is_none());
    }
}
