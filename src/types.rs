use std::collections::HashMap;

use crate::ast::{BinaryOperator, Expr};


#[derive(PartialEq, Eq, Debug, Clone)]
pub enum Type {
    TInt,
    TBool
}

#[derive(PartialEq, Eq, Debug, Clone)]
pub enum TypeError {
    InvalidType(String),
    UnknownVariableType(String)
}

struct TypeBindings {
    bindings : HashMap<String, Type>
}

impl TypeBindings {
    fn new() -> Self {
        Self { bindings: HashMap::new() }
    }

    fn get(&self, name: &str) -> Option<Type> {
        self.bindings.get(name).map(|x| x.clone())
    }

    fn insert(&mut self, name: &str, typ: Type) {
        if let Some(x) = self.bindings.insert(name.to_string(), typ) {
            panic!("Overwriting exisiting type information for {}", name);
        }
    }
}

pub fn type_check(expr: Expr) -> Result<Type, TypeError> {
    let mut bindings = TypeBindings::new();

    inner_type_check(expr, &mut bindings)
}

fn expect(a: Result<Type, TypeError>, typ: Type) -> Result<Type, TypeError> {
    a.and_then(|x| if x == typ { Ok(x) } else { Err(TypeError::InvalidType(String::new())) })
}

fn inner_type_check(expr: Expr, bindings: &mut TypeBindings) -> Result<Type, TypeError> {
    match expr {
        Expr::Int(_) => Ok(Type::TInt),
        Expr::Bool(_) => Ok(Type::TBool),
        Expr::Var(name) => bindings.get(&name).ok_or(TypeError::UnknownVariableType(name.to_string())),
        Expr::Let(name, expr) => {
            let expr_type = inner_type_check(*expr, bindings)?;
            bindings.insert(&name, expr_type.clone());
            Ok(expr_type)
        },
        Expr::If(expr, expr1, expr2) => todo!(),
        Expr::Lambda(exprs, expr) => todo!(),
        Expr::Apply(expr, exprs) => todo!(),
        Expr::Binary(binary_operator, left, right) => {
            let tleft = inner_type_check(*left, bindings);
            let tright = inner_type_check(*right, bindings);

            use BinaryOperator::*;
            match binary_operator {
                Add | Subtract | Multiply | Divide => match tleft.and(tright) {
                    Ok(x) => if x == Type::TInt { Ok(x) } else { Err(TypeError::InvalidType(String::from(""))) },
                    x => x,
                },
                And | Or => match tleft.and(tright) {
                    Ok(x) => if x == Type::TInt { Ok(x) } else { Err(TypeError::InvalidType(String::from(""))) },
                    x => x,
                },
                Equals => Ok(Type::TBool)
            }
        },
    }
}

#[cfg(test)]
mod test {
    use crate::parser::parse;

    use super::*;

    #[test]
    fn basic_check_int() {
        let actual = type_check(Expr::Int(1));
        let expected = Ok(Type::TInt);

        assert_eq!(actual, expected)
    }

    #[test]
    fn basic_check_bool() {
        let actual = type_check(Expr::Bool(true));
        let expected = Ok(Type::TBool);

        assert_eq!(actual, expected)
    }

    #[test]
    fn basic_bool_var() {
        let leta = Expr::Let("a".to_string(), Box::new(Expr::Bool(true)));
        let actual = type_check(leta);
        let expected = Ok(Type::TBool);

        assert_eq!(actual, expected)
    }

    #[test]
    fn basic_int_var() {
        let leta = Expr::Let("a".to_string(), Box::new(Expr::Int(0)));
        let actual = type_check(leta);
        let expected = Ok(Type::TInt);

        assert_eq!(actual, expected)
    }

    #[test]
    fn basic_binary_int_var() {
        let input = "let a = 1 + (2 + 10);";
        let leta = parse(input)[0].clone();
        let actual = type_check(leta);
        let expected = Ok(Type::TBool);

        assert_eq!(actual, expected)
    }
}
