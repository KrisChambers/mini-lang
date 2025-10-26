use nom::{
    IResult, Parser,
    branch::alt,
    bytes::complete::tag,
    character::complete::{alpha1, alphanumeric1, digit1, multispace0},
    combinator::{map, not, opt, value},
    sequence::{self, delimited, preceded, terminated},
};

#[derive(Eq, PartialEq, Clone, Debug)]
pub enum Expr {
    Var(String),
    Lambda(String, TypeAnn, Box<Expr>),
    App(Box<Expr>, Box<Expr>),
    If(Box<Expr>, Box<Expr>, Box<Expr>),
    Lit(Literal),
    BinOp(Op, Box<Expr>, Box<Expr>),
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

//pub fn parse(input: &str) -> IResult<&str, Vec<Expr>> {
//    Ok(vec![Expr::Var("Blah".to_string())])
//}
//
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

fn parse_binary_expr(input: &str) -> IResult<&str, Expr> {
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
    map(delimited(opt_whitespace, alphanumeric1, opt_whitespace), |x| {
        Expr::Var(x.to_string())
    })
    .parse(input)
}

fn keyword(input: &str) -> IResult<&str, &str> {
    alt((tag("if"), tag("else"), tag("then"), tag("let"))).parse(input)
}

fn parse_literal(input: &str) -> IResult<&str, Expr> {
    preceded(
        not(keyword),
        alt((parse_literal_bool, parse_literal_int, parse_variable)),
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

fn parse_expr(input: &str) -> IResult<&str, Expr> {
    alt((
        parse_if_expr,
        parse_lambda,
        parse_application_expr,
        parse_binary_expr,
        parse_literal,
    ))
    .parse(input)
}

fn parse_lambda(input: &str) -> IResult<&str, Expr> {
    // \(x:Int) -> x + 5

    let (input, (param, tparam)) =
        delimited(tag(r"\("), parse_lambda_parameter, tag(")")).parse(input)?;
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

fn parse_type(input: &str) -> IResult<&str, TypeAnn> {
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

fn parse_if_expr(input: &str) -> IResult<&str, Expr> {
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

fn parse_application_expr(input: &str) -> IResult<&str, Expr> {
    /*
     * "(\(x: Int) -> x + 2) 5
     * add2 5
     * 5 + add2 5
     * isTrue true
     * */
    let (input, expr) = alt((
        // Either (<lambda_definition>) ...
        delimited(tag("("), parse_lambda, tag(")")),
        // or a named lambda ex: add2 5
        parse_literal
    ))
    .parse(input)?;

    map(
        alt((
            delimited(tag("("), parse_expr, tag(")")),
            preceded(opt_whitespace, parse_expr),
        )),
        |x| Expr::App(Box::new(expr.clone()), Box::new(x)),
    )
    .parse(input)
}

/* NOTE (kc):
    Maybe clean this up at some point.
    Can probably format / group this stuff better
*/

#[test]
fn parsing_int_type() {
    let input = "Int";
    let (_, actual) = parse_type(input).unwrap();

    assert_eq!(actual, TypeAnn::Int);
}

#[test]
fn parsing_bool_type() {
    let input = "Bool";
    let (_, actual) = parse_type(input).unwrap();

    assert_eq!(actual, TypeAnn::Bool);
}

#[test]
fn parsing_arrow_type() {
    let input = "Int -> Int";
    let (_, actual) = parse_type(input).unwrap();

    assert_eq!(
        actual,
        TypeAnn::Arrow(Box::new(TypeAnn::Int), Box::new(TypeAnn::Int))
    );
}

#[test]
fn parsing_complex_arrow() {
    let input = "(Int -> Int) -> Int";
    let (_, actual) = parse_type(input).unwrap();

    assert_eq!(
        actual,
        TypeAnn::Arrow(
            Box::new(TypeAnn::Arrow(Box::new(TypeAnn::Int), Box::new(TypeAnn::Int))),
            Box::new(TypeAnn::Int)
        )
    );
}

#[test]
fn parse_simple_lambda() {
    let input = "\\(x: Int) -> x + 5";
    let expected = Expr::Lambda(
        "x".to_string(),
        TypeAnn::Int,
        Box::new(Expr::BinOp(
            Op::Add,
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Lit(Literal::Int(5))),
        )),
    );
    let (_, actual) = parse_lambda(input).unwrap();

    assert_eq!(actual, expected)
}

#[test]
fn parse_simple_if() {
    let input = "if True then False else 1";
    let expected = Expr::If(
        Box::new(Expr::Lit(Literal::Bool(true))),
        Box::new(Expr::Lit(Literal::Bool(false))),
        Box::new(Expr::Lit(Literal::Int(1))),
    );

    let (_, actual) = parse_if_expr(input).unwrap();

    assert_eq!(actual, expected)
}

#[test]
fn parse_lambda_definition_application() {
    let input = r"(\(x: Int) -> x + 2) 5";
    let expected = Expr::App(
        Box::new(Expr::Lambda(
            "x".to_string(),
            TypeAnn::Int,
            Box::new(Expr::BinOp(
                Op::Add,
                Box::new(Expr::Var("x".to_string())),
                Box::new(Expr::Lit(Literal::Int(2))),
            )),
        )),
        Box::new(Expr::Lit(Literal::Int(5))),
    );

    let (_, actual) = parse_application_expr(input).unwrap();

    assert_eq!(actual, expected);
}

#[test]
fn parse_multi_lambda() {
    let input = r"\(f: Int -> Int) -> \(x: Int) -> f x";
    let expected = Expr::Lambda(
        "f".to_string(),
        TypeAnn::Arrow(Box::new(TypeAnn::Int), Box::new(TypeAnn::Int)),
        Box::new(Expr::Lambda(
            "x".to_string(),
            TypeAnn::Int,
            Box::new(Expr::App(
                Box::new(Expr::Var("f".to_string())),
                Box::new(Expr::Var("x".to_string())),
            )),
        )),
    );

    let (_, actual) = parse_expr(input).unwrap();

    assert_eq!(actual, expected);
}

#[test]
fn parse_lambda_var_application() {
    let input = "add2 5";
    let expected = Expr::App(
        Box::new(Expr::Var("add2".to_string())),
        Box::new(Expr::Lit(Literal::Int(5))),
    );

    let (_, actual) = parse_application_expr(input).unwrap();

    assert_eq!(actual, expected);
}

// Binary Operation Tests
#[test]
fn parse_simple_addition() {
    let input = "5 + 3";
    let expected = Expr::BinOp(
        Op::Add,
        Box::new(Expr::Lit(Literal::Int(5))),
        Box::new(Expr::Lit(Literal::Int(3))),
    );
    let (_, actual) = parse_binary_expr(input).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn parse_boolean_and() {
    let input = "True && False";
    let expected = Expr::BinOp(
        Op::And,
        Box::new(Expr::Lit(Literal::Bool(true))),
        Box::new(Expr::Lit(Literal::Bool(false))),
    );
    let (_, actual) = parse_binary_expr(input).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn parse_boolean_or() {
    let input = "False || True";
    let expected = Expr::BinOp(
        Op::Or,
        Box::new(Expr::Lit(Literal::Bool(false))),
        Box::new(Expr::Lit(Literal::Bool(true))),
    );
    let (_, actual) = parse_binary_expr(input).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn parse_nested_binary_ops() {
    let input = "(5 + 3) + 2";
    let expected = Expr::BinOp(
        Op::Add,
        Box::new(Expr::BinOp(
            Op::Add,
            Box::new(Expr::Lit(Literal::Int(5))),
            Box::new(Expr::Lit(Literal::Int(3))),
        )),
        Box::new(Expr::Lit(Literal::Int(2))),
    );
    let (_, actual) = parse_binary_expr(input).unwrap();
    assert_eq!(actual, expected);
}

// Complex Type Tests
#[test]
fn parse_right_associative_arrow_types() {
    let input = "Int -> Int -> Int";
    let expected = TypeAnn::Arrow(
        Box::new(TypeAnn::Int),
        Box::new(TypeAnn::Arrow(
            Box::new(TypeAnn::Int),
            Box::new(TypeAnn::Int)
        ))
    );
    let (_, actual) = parse_type(input).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn parse_bool_to_bool_arrow() {
    let input = "Bool -> Bool";
    let expected = TypeAnn::Arrow(
        Box::new(TypeAnn::Bool),
        Box::new(TypeAnn::Bool)
    );
    let (_, actual) = parse_type(input).unwrap();
    assert_eq!(actual, expected);
}

// Complex If Expression Tests
#[test]
fn parse_nested_if_expr() {
    let input = "if True then if False then 1 else 2 else 3";
    let expected = Expr::If(
        Box::new(Expr::Lit(Literal::Bool(true))),
        Box::new(Expr::If(
            Box::new(Expr::Lit(Literal::Bool(false))),
            Box::new(Expr::Lit(Literal::Int(1))),
            Box::new(Expr::Lit(Literal::Int(2))),
        )),
        Box::new(Expr::Lit(Literal::Int(3))),
    );
    let (_, actual) = parse_if_expr(input).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn parse_if_with_binary_condition() {
    let input = "if 5 + 3 then True else False";
    let expected = Expr::If(
        Box::new(Expr::BinOp(
            Op::Add,
            Box::new(Expr::Lit(Literal::Int(5))),
            Box::new(Expr::Lit(Literal::Int(3))),
        )),
        Box::new(Expr::Lit(Literal::Bool(true))),
        Box::new(Expr::Lit(Literal::Bool(false))),
    );
    let (_, actual) = parse_if_expr(input).unwrap();
    assert_eq!(actual, expected);
}

// Application Tests
#[test]
fn parse_application_with_literal_arg() {
    let input = "myFunc True";
    let expected = Expr::App(
        Box::new(Expr::Var("myFunc".to_string())),
        Box::new(Expr::Lit(Literal::Bool(true))),
    );
    let (_, actual) = parse_application_expr(input).unwrap();
    assert_eq!(actual, expected);
}

// Lambda Tests
#[test]
fn parse_lambda_with_bool_param() {
    let input = r"\(b: Bool) -> b";
    let expected = Expr::Lambda(
        "b".to_string(),
        TypeAnn::Bool,
        Box::new(Expr::Var("b".to_string())),
    );
    let (_, actual) = parse_lambda(input).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn parse_lambda_with_function_type() {
    let input = r"\(f: Int -> Bool) -> f";
    let expected = Expr::Lambda(
        "f".to_string(),
        TypeAnn::Arrow(Box::new(TypeAnn::Int), Box::new(TypeAnn::Bool)),
        Box::new(Expr::Var("f".to_string())),
    );
    let (_, actual) = parse_lambda(input).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn parse_lambda_returning_if() {
    let input = r"\(x: Int) -> if True then x else 0";
    let expected = Expr::Lambda(
        "x".to_string(),
        TypeAnn::Int,
        Box::new(Expr::If(
            Box::new(Expr::Lit(Literal::Bool(true))),
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Lit(Literal::Int(0))),
        )),
    );
    let (_, actual) = parse_lambda(input).unwrap();
    assert_eq!(actual, expected);
}

// Complex Combined Tests
#[test]
fn parse_if_with_lambda_in_branch() {
    let input = r"if True then \(x: Int) -> x else \(y: Int) -> y + 1";
    let expected = Expr::If(
        Box::new(Expr::Lit(Literal::Bool(true))),
        Box::new(Expr::Lambda(
            "x".to_string(),
            TypeAnn::Int,
            Box::new(Expr::Var("x".to_string())),
        )),
        Box::new(Expr::Lambda(
            "y".to_string(),
            TypeAnn::Int,
            Box::new(Expr::BinOp(
                Op::Add,
                Box::new(Expr::Var("y".to_string())),
                Box::new(Expr::Lit(Literal::Int(1))),
            )),
        )),
    );
    let (_, actual) = parse_if_expr(input).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn parse_subtraction() {
    let input = "10 - 5";
    let expected = Expr::BinOp(
        Op::Subtract,
        Box::new(Expr::Lit(Literal::Int(10))),
        Box::new(Expr::Lit(Literal::Int(5))),
    );
    let (_, actual) = parse_binary_expr(input).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn parse_variable_in_binary_op() {
    let input = "x + 5";
    let expected = Expr::BinOp(
        Op::Add,
        Box::new(Expr::Var("x".to_string())),
        Box::new(Expr::Lit(Literal::Int(5))),
    );
    let (_, actual) = parse_binary_expr(input).unwrap();
    assert_eq!(actual, expected);
}
