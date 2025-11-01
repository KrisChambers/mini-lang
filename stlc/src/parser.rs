use nom::{
    branch::alt, bytes::complete::tag, character::complete::{alpha1, alphanumeric1, digit1, multispace0}, combinator::{map, not, opt, value}, multi::{many0, many1, separated_list1}, sequence::{self, delimited, preceded, terminated}, IResult, Parser
};

#[derive(Eq, PartialEq, Clone, Debug)]
pub enum Expr {
    Var(String),
    Lambda(String, Option<TypeAnn>, Box<Expr>),
    App(Box<Expr>, Box<Expr>),
    If(Box<Expr>, Box<Expr>, Box<Expr>),
    Lit(Literal),
    BinOp(Op, Box<Expr>, Box<Expr>),
    Let(String, Box<Expr>, Box<Expr>),
}

#[derive(Eq, PartialEq, Clone, Debug)]
pub enum TypeAnn {
    Int,
    Bool,
    Arrow(Box<TypeAnn>, Box<TypeAnn>),
}

#[derive(Eq, PartialEq, Clone, Debug)]
pub enum Literal {
    Int(i32),
    Bool(bool),
}

#[derive(Eq, PartialEq, Clone, Debug)]
pub enum Op {
    Add,
    Subtract,
    And,
    Or,
}

pub fn parse(mut input: &str) -> IResult<&str, Vec<Expr>> {
    let mut result = vec![];

    while !input.is_empty() {
        let (i, expr) = parse_expr(input)?;

        input = i;
        result.push(expr)
    }

    nom::IResult::Ok((input, result))
}

fn parse_binary_op(input: &str) -> IResult<&str, Op> {
    delimited(
        opt_whitespace,
        alt((
            value(Op::Or, tag("||")),
            value(Op::And, tag("&&")),
            value(Op::Add, tag("+")),
            value(Op::Subtract, tag("-")),
        )),
        opt_whitespace,
    )
    .parse(input)
}

pub fn parse_binary_expr(input: &str) -> IResult<&str, Expr> {
    // 5 + 5
    // (5 + 5) + 5
    // (True or True) + 5
    // Expr OP Expr
    // Lit OP Lit
    // Expr OP Lit
    // Lit OP Expr
    let (input, left) = alt((
        delimited(
            tag("("),
            delimited(opt_whitespace, parse_binary_expr, opt_whitespace),
            tag(")"),
        ),
        parse_literal,
    ))
    .parse(input)?;

    let (input, op) = delimited(opt_whitespace, parse_binary_op, opt_whitespace).parse(input)?;

    map(alt((parse_binary_expr, parse_literal)), |right| {
        Expr::BinOp(op.clone(), Box::new(left.clone()), Box::new(right.clone()))
    })
    .parse(input)
}

fn parse_literal_int(input: &str) -> IResult<&str, Expr> {
    map(digit1, |x: &str| {
        Expr::Lit(Literal::Int(x.parse().expect("Expected Integer")))
    })
    .parse(input)
}

fn parse_literal_bool(input: &str) -> IResult<&str, Expr> {
    alt((
        value(Expr::Lit(Literal::Bool(false)), tag("False")),
        value(Expr::Lit(Literal::Bool(true)), tag("True")),
    ))
    .parse(input)
}

fn parse_variable(input: &str) -> IResult<&str, Expr> {
    map(
        delimited(opt_whitespace, alphanumeric1, opt_whitespace),
        |x| Expr::Var(x.to_string()),
    )
    .parse(input)
}

fn keyword(input: &str) -> IResult<&str, &str> {
    alt((tag("if"), tag("else"), tag("then"), tag("let"), tag("in"))).parse(input)
}

pub fn parse_literal(input: &str) -> IResult<&str, Expr> {
    preceded(
        not(keyword),
        delimited(
            opt_whitespace,
            alt((parse_literal_bool, parse_literal_int, parse_variable)),
            opt_whitespace,
        ),
    )
    .parse(input)
}

/*
* NOTE (kc): the Application parsing here is causing an issue.
*
* I think we need to be more clear where applications are viable.
*
* for instance: if (\(x: Int) -> x == 2) 5 then ... else ...
*
* Should should probably be possible...
*
* But should (5 + 2) true be possible?
*
* Maybe we need to thing this through.
* Almost anything can be an application ...
*/

pub fn parse_expr(input: &str) -> IResult<&str, Expr> {
    alt((
        parse_let_expr,
        parse_if_expr,
        parse_lambda,
        parse_binary_expr,
        parse_application_expr,
        parse_literal,
    ))
    .parse(input)
}

pub fn parse_lambda(input: &str) -> IResult<&str, Expr> {
    // \(x:Int) -> x + 5
    // \x -> x + 5

    let (input, _) = opt_whitespace(input)?;

    let (input, (param, tparam)) = alt((
        map(
            delimited(tag(r"\("), parse_lambda_parameter, tag(")")),
            |(p, t)| (p, Some(t)),
        ),
        map(preceded(tag(r"\"), alpha1), |x: &str| (x.to_string(), None)),
    ))
    .parse(input)?;

    let (input, _) = parse_arrow(input)?;

    map(parse_expr, |expr| {
        Expr::Lambda(param.clone(), tparam.clone(), Box::new(expr))
    })
    .parse(input)
}

fn opt_whitespace(input: &str) -> IResult<&str, Option<&str>> {
    opt(multispace0).parse(input)
}

fn parse_lambda_parameter(input: &str) -> IResult<&str, (String, TypeAnn)> {
    /*
     * x
     * x : Int
     * y : Bool
     * z : Int -> Int
     * */
    let (input, name) = map(delimited(opt_whitespace, alpha1, opt_whitespace), |s| {
        s.to_string()
    })
    .parse(input)?;

    let (input, _) = delimited(opt_whitespace, tag(":"), opt_whitespace).parse(input)?;

    opt(parse_type)
        .map(|ot| match ot {
            Some(t) => (name.clone(), t),
            None => panic!("Missing Type Annotation"),
        })
        .parse(input)
}

fn parse_lambda_parameter_name(input: &str) -> IResult<&str, String> {
    map(alpha1, |s: &str| s.to_string()).parse(input)
}

fn parse_arrow(input: &str) -> IResult<&str, &str> {
    terminated(preceded(opt(multispace0), tag("->")), opt(multispace0)).parse(input)
}

pub fn parse_type(input: &str) -> IResult<&str, TypeAnn> {
    /*
     * Int
     * Bool
     * Int -> Int
     * (Int -> Int) -> Int
     */
    let (input, first) = alt((
        sequence::terminated(sequence::preceded(tag("("), parse_type), tag(")")),
        value(TypeAnn::Int, tag("Int")),
        value(TypeAnn::Bool, tag("Bool")),
    ))
    .parse(input)?;

    if let (input, Some(_)) = opt(parse_arrow).parse(input)? {
        map(parse_type, |snd| {
            TypeAnn::Arrow(Box::new(first.clone()), Box::new(snd))
        })
        .parse(input)
    } else {
        Ok((input, first))
    }
}

fn parse_keyword(keyword: &str) -> impl Fn(&str) -> IResult<&str, &str> {
    move |input: &str| {
        preceded(opt_whitespace, terminated(tag(keyword), opt_whitespace)).parse(input)
    }
}

pub fn parse_let_assignment(input: &str) -> IResult<&str, (String, Expr)> {
    let (input, var) = preceded(opt_whitespace, alphanumeric1).parse(input)?;
    let (input, _) = preceded(opt_whitespace, tag("=")).parse(input)?;
    let (input, e1) = parse_expr(input)?;

    nom::IResult::Ok((input, (var.to_string(), e1)))
}

pub fn parse_nested_let_assignment(input:&str) -> IResult<&str, Vec<(String, Expr)>> {
    separated_list1(
        preceded(opt_whitespace, tag(",")),
        parse_let_assignment
    ).parse(input)
}

pub fn parse_let_expr(input: &str) -> IResult<&str, Expr> {
    let (input, _) = parse_keyword("let")(input)?;
    let (input, assignments) = parse_nested_let_assignment(input)?;

    // let (input, (var, e1)) = parse_let_assignment(input)?;
    let (input, _) = preceded(opt_whitespace, tag("in")).parse(input)?;
    let (input, e2) = parse_expr(input)?;

    nom::IResult::Ok((
        input,
     assignments.iter().rev().fold(e2, |acc, (var, e)|
        Expr::Let(var.clone(), Box::new(e.clone()), Box::new(acc.clone())))
    ))
}

pub fn parse_if_expr(input: &str) -> IResult<&str, Expr> {
    let (input, _) = parse_keyword("if")(input)?;
    let (input, cond) = parse_expr(input)?;
    let (input, _) = parse_keyword("then")(input)?;
    let (input, true_branch) = parse_expr(input)?;
    let (input, _) = parse_keyword("else")(input)?;
    map(parse_expr, |false_branch| {
        Expr::If(
            Box::new(cond.clone()),
            Box::new(true_branch.clone()),
            Box::new(false_branch.clone()),
        )
    })
    .parse(input)
}

pub fn parse_application_expr(input: &str) -> IResult<&str, Expr> {
    /*
     * (\(x: Int) -> x + 2) 5
     * add2 5
     * 5 + add2 5
     * isTrue true
     * add 2 5 == (add 2) 5
     * */
    let mut parse_atomic_term = preceded(opt_whitespace,alt((
        // Either (<lambda_definition>) ...
        delimited(tag("("), parse_expr, tag(")")),
        // add x y = (add x) y
        //parse_application_expr,
        // or a named lambda ex: add2 5
        parse_literal,
    )));

    let (input, initial) = parse_atomic_term.parse(input)?;

    map(many0(parse_atomic_term), |exprs| exprs.iter().fold(initial.clone(), |acc, arg|
        Expr::App(Box::new(acc), Box::new(arg.clone()))
    ) ).parse(input)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn parsing_let_assignment_sugar() {
        let input = "
x = 2,
y = 3
".trim();

        let (_, result) = parse_nested_let_assignment(input).unwrap();

        assert_eq!(result.len(), 2);

        assert_eq!(result[0], ("x".to_string(), Expr::Lit(Literal::Int(2))));
        assert_eq!(result[1], ("y".to_string(), Expr::Lit(Literal::Int(3))));

        let input = "x = 1";
        let (_, result) = parse_nested_let_assignment(input).unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0], ("x".to_string(), Expr::Lit(Literal::Int(1))));

    }

    #[test]
    fn parsing_sugared_let_expression() {
        let input = "
let
    x = 2,
    y = 3
in
    x + y

".trim();
        let (_, result) = parse_let_expr(input).unwrap();
        let expected = Expr::Let(
            "x".to_string(),
            Box::new(Expr::Lit(Literal::Int(2))),
            Box::new(
                Expr::Let(
                    "y".to_string(),
                    Box::new(Expr::Lit(Literal::Int(3))),
                    Box::new(Expr::BinOp(Op::Add,
                        Box::new(Expr::Var("x".to_string())),
                        Box::new(Expr::Var("y".to_string()))
                    ))

                ))
        );

        assert_eq!(result, expected);

    }
}
