use std::collections::HashMap;
use std::rc::Rc;

use crate::ast::{Expr, BinaryOperator};

pub fn eval_program(p: Vec<Expr>) -> Option<Expr> {
    let mut bindings = Rc::new(HashMap::new());
    let mut result = None;

    for expr in p {
        result = Some(inner_eval(expr, &mut bindings));
    }

    result
}

pub fn eval(t: Expr) -> Expr {
    let mut bindings = Rc::new(HashMap::new());

    inner_eval(t, &mut bindings)
}

fn inner_eval(t: Expr, bindings: &mut Rc<HashMap<String, Expr>>) -> Expr {
    use Expr::*;

    match t {
        Let(name, expr) => {
            // We don't necessarily want the expression to be evaluated yet.
            // Expr can only be evaluated if it has no free terms.

            // This is to handle lambdas since what : let add = fn(a, b) { a + b }

            let result = match *expr {
                Lambda(_, _) => *expr,
                _ => inner_eval(*expr, bindings)
            };

            let bindings = Rc::get_mut(bindings).unwrap();
            bindings.insert(name, result.clone());

            result
        },
        Lambda(_, body) => {
            inner_eval(*body, bindings)
        },

        Apply(alias, args) => {
            let fn_name = match *alias {
                Var(name) => name,
                _ => panic!("Expected a binding name")
            };

            let lambda_expr = bindings.get(&fn_name).unwrap_or_else(|| panic!("No binding named {fn_name}")).clone();

            // Create a subset of bindings that we need for the lambda
            let mut inner_bindings = match lambda_expr {
                // This seems a bit messy
                Lambda(ref params, _) => {
                    let mut bindings = Rc::get_mut(bindings).unwrap().clone();
                    let param_names: Vec<_> = params.iter().map(|x| match x {
                        Var(name) => name.clone(),
                        _ => panic!("Not a valid parameter name")

                    }).collect();

                    for (param_name, arg) in param_names.iter().zip(args.iter()) {
                        let _ = bindings.insert(param_name.clone(), arg.clone());
                    }

                    Rc::new(bindings)
                },
                _ => panic!("Invalid Lambda Expression")
            };

            inner_eval(lambda_expr, &mut inner_bindings)
        },

        Var(name) => bindings.get(&name).unwrap().clone(),

        If(condition, true_branch, false_branch) => match (
            inner_eval(*condition, bindings),
            inner_eval(*true_branch, bindings),
            inner_eval(*false_branch, bindings),
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

        Binary(op, left, right) => {
            use BinaryOperator::*;
            let left = inner_eval(*left, bindings);
            let right = inner_eval(*right, bindings);

            match op {
                Add => match (left, right) {
                    (Int(a), Int(b)) => Int(a + b),
                    _ => panic!("Can only add integers"),
                },
                Subtract => match (left, right) {
                    (Int(a), Int(b)) => Int(a - b),
                    _ => panic!("Can only add integers"),
                },
                Multiply => match (left, right) {
                    (Int(a), Int(b)) => Int(a * b),
                    _ => panic!("Can only add integers"),
                },

                Divide => match (left, right) {
                    (Int(a), Int(b)) => Int(a / b),
                    _ => panic!("Can only add integers"),
                },
                And => match (left, right) {
                    (Bool(a), Bool(b)) => Bool(a && b),
                    _ => panic!("Boolen operation on non-booleans"),
                },
                Or => match (left, right) {
                     (Bool(a), Bool(b)) => Bool(a || b),
                    _ => panic!("Boolen operation on non-booleans"),
                }
                Equals => match (left, right) {
                    (Int(a), Int(b)) => Bool(a == b),
                    (Bool(a), Bool(b)) => Bool(a == b),
                    _ => Bool(false) // Could also just not compare them?
                },
            }
        },
        Int(x) => Int(x),
        Bool(x) => Bool(x),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Expr::*;
    use BinaryOperator::*;

    #[test]
    fn add() {
        let expr = Binary(Add, Box::new(Int(1)), Box::new(Int(2)));
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
        let result = inner_eval(expr, &mut variables);

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
        let cond = Binary(Add, Box::new(Var("a".to_string())), Box::new(Int(10)));
        let result = eval_program(vec![bind, cond]).unwrap();

        assert_eq!(result, Int(11))
    }

    fn create_add_expr() -> Expr {
        Lambda(vec![Var("a".to_string()), Var("b".to_string())], Box::new(Binary(Add,Box::new(Var("a".to_string())), Box::new(Var("b".to_string())))))
    }

    #[test]
    fn aliased_lambda_creation() {
        // Aliased lambda's should not be evaluated further until an apply call
        let lambda = Let("add".to_string(), Box::new(create_add_expr()));

        let result = eval(lambda.clone());

        assert_eq!(result, create_add_expr());
    }

    #[test]
    fn lambda_evaluation() {
        let lambda = create_add_expr();

        let mut bindings = HashMap::new();
        bindings.insert("a".to_string(), Int(1));
        bindings.insert("b".to_string(), Int(2));

        let mut bindings = Rc::new(bindings);
        let result = inner_eval(lambda, &mut bindings);

        assert_eq!(result, Int(3));
    }

    #[test]
    fn lambda_application_evaluation() {
        let lambda = Let("add".to_string(), Box::new(create_add_expr()));
        let apply = Apply(Box::new(Var("add".to_string())), vec![Int(1), Int(2)]);

        assert_eq!(eval_program(vec![lambda, apply]), Some(Int(3)));
    }

}

