mod eval;
mod ast;

use eval::eval_program;
use ast::Expr;

fn main() {
    use Expr::*;

    let bind = Let("a".to_string(), Box::new(Int(1)));
    let cond = Add(Box::new(Var("a".to_string())), Box::new(Int(10)));
    let result = eval_program(vec![bind, cond]).unwrap();

    assert_eq!(result, Int(11))
}

