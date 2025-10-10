use crate::ast::{BinaryOperator, Expr};

use nom::{
    branch::alt, bytes::complete::{tag, take_while1}, character::complete::multispace0, combinator::{map, opt, value}, error::Error, multi::{many1, separated_list0, separated_list1}, sequence::{self, delimited, terminated}, AsChar, Err, IResult, Parser
};

fn parse_bool(input: &str) -> IResult<&str, Expr> {
    alt((
        value(Expr::Bool(true), tag("true")),
        value(Expr::Bool(false), tag("false")),
    ))
    .parse(input)
}

fn parse_int(input: &str) -> IResult<&str, Expr> {
    map(
        space_terminated(take_while1(AsChar::is_dec_digit)),
        |x: &str| Expr::Int(x.parse().expect("Could Not parse to int")),
    )
    .parse(input)
}

fn parse_if(input: &str) -> IResult<&str, Expr> {
    let (input, _) = space_terminated(tag("if")).parse(input)?;
    // Need to handle boolean operators
    let (input, cond) = parse_binary_operation.parse(input)?;
    let (input, _) = space_terminated(tag("then")).parse(input)?;
    let (input, true_branch) = parse_binary_operation.parse(input)?;
    let (input, _) = space_terminated(tag("else")).parse(input)?;

    space_terminated(parse_binary_operation)
        .parse(input)
        .map(|(input, false_branch)| {
            (
                input,
                Expr::If(
                    Box::new(cond),
                    Box::new(true_branch),
                    Box::new(false_branch),
                ),
            )
        })
}

fn name(input: &str) -> IResult<&str, &str> {
    take_while1(AsChar::is_alphanum)(input)
}

fn parse_var(input: &str) -> IResult<&str, Expr> {
    map(name, |x: &str| Expr::Var(x.to_string())).parse(input)
}

fn space_terminated<'a, P, T>(
    parser: P,
) -> impl nom::Parser<&'a str, Output = T, Error = Error<&'a str>>
where
    P: Parser<&'a str, Error = Error<&'a str>, Output = T>,
{
    sequence::terminated(parser, multispace0)
}

fn parse_lambda(input: &str) -> IResult<&str, Expr> {
    let (input, args) = sequence::preceded(
        tag("fn"),
        delimited(
            tag("("),
            separated_list1(space_terminated(tag(",")), name),
            space_terminated(tag(")")),
        ),
    )
    .parse(input)?;

    map(
        delimited(
            space_terminated(tag("{")),
            space_terminated(parse_body),
            space_terminated(tag("}")),
        ),
        |x| {
            Expr::Lambda(
                args.iter().map(|x| Expr::Var(x.to_string())).collect(),
                Box::new(x),
            )
        },
    )
    .parse(input)
}

fn binary_operator(input: &str) -> IResult<&str, BinaryOperator> {
    alt((
        value(BinaryOperator::Add, tag("+")),
        value(BinaryOperator::Subtract, tag("-")),
        value(BinaryOperator::Multiply, tag("*")),
        value(BinaryOperator::Divide, tag("/")),
        value(BinaryOperator::Or, tag("||")),
        value(BinaryOperator::And, tag("&&")),
        value(BinaryOperator::Equals,      tag("=="))
    ))
    .parse(input)
}

fn parse_binary_operation(input: &str) -> IResult<&str, Expr> {
    // 1
    // true
    // (x && y) || z
    // z && (x && y)
    // x && y

    let (input, left) = space_terminated(alt((
        parse_int,
        parse_bool,
        parse_var,
        delimited(
            space_terminated(tag("(")),
            parse_binary_operation,
            space_terminated(tag(")")),
        ),
    )))
    .parse(input)?;

    map(
        opt(sequence::pair(
            space_terminated(binary_operator),
            space_terminated(parse_binary_operation),
        )),
        |tail| match tail {
            Some((op, right)) => Expr::Binary(op, Box::new(left.clone()), Box::new(right)),
            None => left.clone(),
        },
    )
    .parse(input)

}

fn parse_body(input: &str) -> IResult<&str, Expr> {
    alt((
        parse_if,
        parse_lambda,
        parse_apply,
        parse_binary_operation,
        parse_var,
    ))
    .parse(input)
}

fn parse_let_expr(input: &str) -> IResult<&str, Expr> {
    alt((
        parse_if,
        parse_lambda,
        parse_apply,
        parse_binary_operation
        //parse_boolean_operation,
        //parse_numeric_operation,
    ))
    .parse(input)
}

fn parse_let_line(input: &str) -> IResult<&str, Expr> {
    let (input, _) = space_terminated(tag("let")).parse(input)?;
    let (input, var_name) = space_terminated(name).parse(input)?;
    let (input, _) = space_terminated(tag("=")).parse(input)?;

    map(parse_let_expr, |expr| {
        Expr::Let(var_name.to_string(), Box::new(expr))
    })
    .parse(input)
}

fn parse_apply(input: &str) -> IResult<&str, Expr> {
    let (input, name) = name(input)?;

    map(
        delimited(
            space_terminated(tag("(")),
            separated_list0(
                space_terminated(tag(",")),
                alt((parse_bool, parse_int, parse_var)),
            ),
            space_terminated(tag(")")),
        ),
        |params| Expr::Apply(Box::new(Expr::Var(name.to_string())), params),
    )
    .parse(input)
}

fn parse_line(input: &str) -> IResult<&str, Expr> {
    alt((parse_let_line, parse_apply)).parse(input)
}

pub fn parse(input: &str) -> Vec<Expr> {
    let result = many1(terminated( parse_line, space_terminated(tag(";")))).parse(input);

    match result {
        Ok((rest, lines)) => {
            println!("unparsed input: {}", rest);
            lines
        }
        Err(err) => match err {
            Err::Incomplete(m) => panic!("Incomplete: {:?}", m),
            Err::Error(m) => panic!("{}", m),
            Err::Failure(m) => panic!("{}", m),
        },
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn parse_simple_if() {
        let input = "if true then false else true";
        let actual = parse_if(input);

        let expected = Expr::If(
            Box::new(Expr::Bool(true)),
            Box::new(Expr::Bool(false)),
            Box::new(Expr::Bool(true)),
        );
        assert_eq!(actual, Ok(("", expected)))
    }

    #[test]
    fn parse_simple_numeric_variable() {
        let input = "let x = 1";
        let actual = parse_let_line(input);
        let expected = Expr::Let("x".to_string(), Box::new(Expr::Int(1)));

        assert_eq!(actual, Ok(("", expected)));
    }

    #[test]
    fn parse_simple_boolean() {
        let input = "let x = true";

        let actual = parse_let_line(input);
        let expected = Expr::Let("x".to_string(), Box::new(Expr::Bool(true)));

        assert_eq!(actual, Ok(("", expected)));
    }

    #[test]
    fn parse_simple_lambda() {
        let input = "let add = fn(a,b) { a + b }";
        let actual = parse_let_line(input);

        let params = vec![Expr::Var("a".to_string()), Expr::Var("b".to_string())];
        let body = Expr::Binary(BinaryOperator::Add,
            Box::new(Expr::Var("a".to_string())),
            Box::new(Expr::Var("b".to_string())),
        );
        let expected = Expr::Let(
            "add".to_string(),
            Box::new(Expr::Lambda(params, Box::new(body))),
        );

        assert_eq!(actual, Ok(("", expected)))
    }

    #[test]
    fn parse_apply_simple() {
        let input = "add(1, 2)";
        let actual = parse_apply(input);

        let expected = Expr::Apply(
            Box::new(Expr::Var("add".to_string())),
            vec![Expr::Int(1), Expr::Int(2)],
        );

        assert_eq!(actual, Ok(("", expected)));
    }

    #[test]
    fn parse_let_apply_simple() {
        let input = "let x = add(1, 2)";
        let actual = parse_let_line(input);

        let expected = Expr::Apply(
            Box::new(Expr::Var("add".to_string())),
            vec![Expr::Int(1), Expr::Int(2)],
        );

        let expected = Expr::Let("x".to_string(), Box::new(expected));

        assert_eq!(actual, Ok(("", expected)));
    }

    #[test]
    fn parse_simple_numeric_experession() {
        let input = "1 + 2";
        let actual = parse_binary_operation(input);

        let expected = Expr::Binary(BinaryOperator::Add,Box::new(Expr::Int(1)), Box::new(Expr::Int(2)));

        assert_eq!(actual, Ok(("", expected)));
    }

    #[test]
    fn parse_simple_int() {
        let input = "22222";
        let actual = parse_int(input);

        let expected = Expr::Int(22222);

        assert_eq!(actual, Ok(("", expected)));
    }

    #[test]
    fn boolean_op1() {
        let input = "true";
        let actual = parse_binary_operation(input);
        let tru = Expr::Bool(true);

        assert_eq!(actual, Ok(("", tru)))
    }

    #[test]
    fn boolean_op2() {
        let input = "true && false";
        let actual = parse_binary_operation(input);
        let tru = Expr::Bool(true);
        let fls = Expr::Bool(false);
        let and = Expr::Binary(BinaryOperator::And,Box::new(tru.clone()), Box::new(fls.clone()));

        assert_eq!(actual, Ok(("", and)))
    }

    #[test]
    fn boolean_op3() {
        let input = "true && (true || false)";
        let actual = parse_binary_operation(input);
        let tru = Expr::Bool(true);
        let fls = Expr::Bool(false);
        let or_d = Expr::Binary(BinaryOperator::Or,Box::new(tru.clone()), Box::new(fls.clone()));

        let expected = Expr::Binary(BinaryOperator::And,Box::new(tru.clone()), Box::new(or_d.clone()));

        assert_eq!(actual, Ok(("", expected)))
    }

    #[test]
    fn boolean_op4() {
        let input = "(true || false) && true";
        let actual = parse_binary_operation(input);
        let tru = Expr::Bool(true);
        let fls = Expr::Bool(false);
        let or_d = Expr::Binary(BinaryOperator::Or,Box::new(tru.clone()), Box::new(fls.clone()));

        let expected = Expr::Binary(BinaryOperator::And,Box::new(or_d.clone()), Box::new(tru.clone()));

        assert_eq!(actual, Ok(("", expected)))
    }

    #[test]
    fn numeric_op1() {
        let input = "11";
        let actual = parse_binary_operation(input);

        let expected = Expr::Int(11);

        assert_eq!(actual, Ok(("", expected)))
    }

    #[test]
    fn numeric_op2() {
        let input = "11 + 2";
        let actual = parse_binary_operation(input);
        let el = Expr::Int(11);
        let two = Expr::Int(2);

        let expected = Expr::Binary(BinaryOperator::Add,Box::new(el), Box::new(two));

        assert_eq!(actual, Ok(("", expected)))
    }

    #[test]
    fn numeric_op3() {
        let input = "11 + (5 - 2)";
        let actual = parse_binary_operation(input);
        let el = Expr::Int(11);
        let two = Expr::Int(2);
        let five = Expr::Int(5);

        let sub = Expr::Binary(BinaryOperator::Subtract,Box::new(five), Box::new(two.clone()));
        let expected = Expr::Binary(BinaryOperator::Add,Box::new(el), Box::new(sub));

        assert_eq!(actual, Ok(("", expected)))
    }

    #[test]
    fn test_lambda_line() {
        let input = "let add2 = fn(x) { x + 2 }";
        let actual = parse_let_line(input);
        println!("Result: {:?}", actual);
        assert!(actual.is_ok());
    }

    #[test]
    fn example_program1() {
        let input = "
let x = 2;
let add2 = fn(x) { x + 2 };
let mult2 = fn(x) { x * 2 };
".trim()
        .trim();
        let actual = parse(input);
        let letx = Expr::Let("x".to_string(), Box::new(Expr::Int(2)));
        let add2 = Expr::Let("add2".to_string(), Box::new(Expr::Lambda(vec![Expr::Var("x".to_string())], Box::new(Expr::Binary(BinaryOperator::Add, Box::new(Expr::Var("x".to_string())), Box::new(Expr::Int(2)))))));
        let mult2 = Expr::Let("mult2".to_string(), Box::new(Expr::Lambda(vec![Expr::Var("x".to_string())], Box::new(Expr::Binary(BinaryOperator::Multiply, Box::new(Expr::Var("x".to_string())), Box::new(Expr::Int(2)))))));
        let expected = vec![letx, add2, mult2];

        assert_eq!(actual, expected)
    }
}
