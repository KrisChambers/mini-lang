use std::{collections::HashMap, rc::Rc};

/// Expression AST
#[derive(Eq, PartialEq, PartialOrd, Debug, Clone)]
enum Expr {
    // Integer Value Ex: Int(1)
    Int(i32),
    // Boolean Value Ex: Bool(true)
    Bool(bool),
    // Variable Name Ex: Add(Var("a"), Int(1))
    Var(String),
    //// Let bindings : Assign an expression to a name
    Let(String, Box<Expr>),
    // Conditional
    If(Box<Expr>, Box<Expr>, Box<Expr>),

    /// Int Operators
    Add(Box<Expr>, Box<Expr>),
    Subtract(Box<Expr>, Box<Expr>),
    Mult(Box<Expr>, Box<Expr>),
    Divide(Box<Expr>, Box<Expr>),
}

fn main() {
}

fn eval_program(p: Vec<Expr>) -> Option<Expr> {
    let mut variables = Rc::new(HashMap::new());
    let mut result = None;

    for expr in p {
        result = Some(inner_eval(Box::new(expr), &mut variables));
    }

    return result
}

fn eval(t: Expr) -> Expr {
    let mut variables = Rc::new(HashMap::new());

    return inner_eval(Box::new(t), &mut variables);
}

fn inner_eval(t: Box<Expr>, variables: &mut Rc<HashMap<String, Expr>>) -> Expr {
    use Expr::*;

    match *t {
        Let(name, expr) => {
            let result = inner_eval(expr, variables);

            let variables = Rc::get_mut(variables).unwrap();
            variables.insert(name, result.clone());

            result
        },

        Var(name) => variables.get(&name).unwrap().clone(),

        If(condition, true_branch, false_branch) => match (
            inner_eval(condition, variables),
            inner_eval(true_branch, variables),
            inner_eval(false_branch, variables),
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

        Add(left, right) => match (inner_eval(left, variables), inner_eval(right, variables)) {
            (Int(a), Int(b)) => Int(a + b),
            _ => panic!("Can only add integers"),
        },
        Subtract(left, right) => match (inner_eval(left, variables), inner_eval(right, variables)) {
            (Int(a), Int(b)) => Int(a - b),
            _ => panic!("Can only add integers"),
        },
        Mult(left, right) => match (inner_eval(left, variables), inner_eval(right, variables)) {
            (Int(a), Int(b)) => Int(a * b),
            _ => panic!("Can only add integers"),
        },
        Divide(left, right) => match (inner_eval(left, variables), inner_eval(right, variables)) {
            (Int(a), Int(b)) => Int(a / b),
            _ => panic!("Can only add integers"),
        },

        x => x,
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

        let value  = variables.get("a");

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
