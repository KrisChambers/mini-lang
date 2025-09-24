use std::collections::HashMap;
use std::rc::Rc;

use crate::ast::Expr;

pub fn eval_program(p: Vec<Expr>) -> Option<Expr> {
    let mut bindings = Rc::new(HashMap::new());
    let mut result = None;

    for expr in p {
        result = Some(inner_eval(Box::new(expr), &mut bindings));
    }

    return result;
}

pub fn eval(t: Expr) -> Expr {
    let mut bindings = Rc::new(HashMap::new());

    return inner_eval(Box::new(t), &mut bindings);
}

fn inner_eval(t: Box<Expr>, bindings: &mut Rc<HashMap<String, Expr>>) -> Expr {
    use Expr::*;

    match *t {
        Let(name, expr) => {
            let result = inner_eval(expr, bindings);

            let variables = Rc::get_mut(bindings).unwrap();
            variables.insert(name, result.clone());

            result
        }

        Var(name) => bindings.get(&name).unwrap().clone(),

        If(condition, true_branch, false_branch) => match (
            inner_eval(condition, bindings),
            inner_eval(true_branch, bindings),
            inner_eval(false_branch, bindings),
        ) {
            (Bool(cond), a, b) => {
                if cond {
                    a
                } else {
                    b
                }
            }
            _ => panic!("if condition needs to be a boolean"),
        },

        Add(left, right) => match (inner_eval(left, bindings), inner_eval(right, bindings)) {
            (Int(a), Int(b)) => Int(a + b),
            _ => panic!("Can only add integers"),
        },
        Subtract(left, right) => match (inner_eval(left, bindings), inner_eval(right, bindings)) {
            (Int(a), Int(b)) => Int(a - b),
            _ => panic!("Can only add integers"),
        },
        Mult(left, right) => match (inner_eval(left, bindings), inner_eval(right, bindings)) {
            (Int(a), Int(b)) => Int(a * b),
            _ => panic!("Can only add integers"),
        },
        Divide(left, right) => match (inner_eval(left, bindings), inner_eval(right, bindings)) {
            (Int(a), Int(b)) => Int(a / b),
            _ => panic!("Can only add integers"),
        },

        Int(x) => Int(x),
        Bool(x) => Bool(x),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Expr::*;

    #[test]
    fn add() {
        let expr = Add(Box::new(Int(1)), Box::new(Int(2)));
        let result = eval(expr);

        assert_eq!(result, Int(3));
    }

    #[test]
    fn conditional() {
        let expr = If(Box::new(Bool(true)), Box::new(Int(1)), Box::new(Int(2)));
        let result = eval(expr);

        assert_eq!(result, Int(1));

        let expr = If(Box::new(Bool(false)), Box::new(Int(1)), Box::new(Int(2)));
        let result = eval(expr);

        assert_eq!(result, Int(2));
    }

    #[test]
    fn let_binding() {
        let expr = Let("a".to_string(), Box::new(Int(1)));
        let mut variables = Rc::new(HashMap::new());
        let result = inner_eval(Box::new(expr), &mut variables);

        let value = variables.get("a");

        assert_eq!(result, Int(1));
        assert_eq!(value, Some(&Int(1)));
    }

    #[test]
    fn var_in_bool_expression() {
        let t = Box::new(Bool(true));
        let f = Box::new(Bool(false));

        let bind = Let("a".to_string(), t.clone());
        let cond = If(Box::new(Var("a".to_string())), t.clone(), f.clone());

        let result = eval_program(vec![bind, cond]).unwrap();

        assert_eq!(result, Bool(true))
    }

    #[test]
    fn var_in_int_expression() {
        let bind = Let("a".to_string(), Box::new(Int(1)));
        let cond = Add(Box::new(Var("a".to_string())), Box::new(Int(10)));
        let result = eval_program(vec![bind, cond]).unwrap();

        assert_eq!(result, Int(11))
    }
}

