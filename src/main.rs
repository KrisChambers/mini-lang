mod eval;
mod ast;
mod parser;
mod types;

use parser::parse;

fn main() {
    let input = "let x = 1;";

    let _ = parse(input);

}

