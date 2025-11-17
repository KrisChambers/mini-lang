use nom::{
    branch::alt, bytes::complete::tag, character::complete::{alpha1, alphanumeric1, digit1, multispace0, multispace1}, combinator::{eof, map, not, opt, peek, value}, multi::{many0, many1, separated_list1}, sequence::{self, delimited, preceded, terminated, tuple}, IResult, Parser
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

fn newline(input:&str) -> IResult<&str, ()> {
    value((), tag("\n")).parse(input)
}
fn semicolon(input:&str) -> IResult<&str, ()> {
    value((), tag(";")).parse(input)
}

fn opt_spaces(input:&str) -> IResult<&str, ()> {
    value((), many0(
        alt((
            tag(" "),
            tag(r"\t")
        ))
    )).parse(input)
}

fn parse_definition_delimiter(input:&str) -> IResult<&str, ()> {
    alt((
        value((), (newline, opt_spaces, newline)),
        value((), semicolon)
    )).parse(input)
}

fn parse_top_level_let(input: &str) -> IResult<&str, Expr> {
    let (input, _) = value((), (opt_whitespace, tag("let"), opt_whitespace)).parse(input)?;
    let (input, (name, e)) = parse_let_assignment(input)?;

    nom::IResult::Ok((
        input,
        Expr::Let(
            name.clone(),
            Box::new(e.clone()),
            Box::new(Expr::Var(name.clone())),
        ),
    ))
}


pub fn parse_program(input: &str) -> IResult<&str, Expr> {
    let (input, lets) = separated_list1(
        parse_definition_delimiter,
        parse_top_level_let
    ).parse(input)?;

    let assignments: Vec<(String, Expr)> = lets.into_iter().map(|expr| {
        match expr {
            Expr::Let(name, value, _) => (name, *value),
            _ => panic!("Expected Let expression from parse_top_level_let"),
        }
    }).collect();

    let last_var = assignments.last().map(|(name, _)| name.clone())
        .expect("Expected at least one definition");

    let result = assignments.iter().rev().fold(
        Expr::Var(last_var),
        |acc, (name, expr)| {
            Expr::Let(name.clone(), Box::new(expr.clone()), Box::new(acc))
        }
    );

    Ok((input, result))
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

fn next<T>(input: &str, to_return: T) -> IResult<&str, T>
    where T : Clone
{
    value(to_return, tag("")).parse(input)

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
        parse_literal,
        delimited(
            tag("("),
            delimited(opt_whitespace, parse_expr, opt_whitespace),
            tag(")"),
        ),
    ))
    .parse(input)?;

    let (input, op) = delimited(opt_whitespace, parse_binary_op, opt_whitespace).parse(input)?;

    let (input, right) = alt((parse_literal, parse_expr)).parse(input)?;

    next(input, Expr::BinOp(op.clone(), Box::new(left.clone()), Box::new(right.clone())))
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
        (opt_whitespace, alphanumeric1),
        |(_, x)| Expr::Var(x.to_string()),
    )
    .parse(input)
}

fn keyword(input: &str) -> IResult<&str, &str> {
    preceded(
        opt_whitespace,
        terminated(alt((tag("if"), tag("else"), tag("then"), tag("let"), tag("in"))),
            peek(alt((
                value((), multispace1),
                value((), eof)
            )))
        ),
    )
    .parse(input)
}

pub fn parse_literal(input: &str) -> IResult<&str, Expr> {
    map(
(not(keyword), opt_whitespace, alt((
            parse_literal_bool,
            parse_literal_int,
            parse_variable
        ))), |(_, _, x)| x.clone())
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

pub fn parse_nested_let_assignment(input: &str) -> IResult<&str, Vec<(String, Expr)>> {
    separated_list1(preceded(opt_whitespace, tag(",")), parse_let_assignment).parse(input)
}

pub fn parse_let_expr(input: &str) -> IResult<&str, Expr> {
    let (input, _) = parse_keyword("let")(input)?;
    let (input, assignments) = parse_nested_let_assignment(input)?;

    // let (input, (var, e1)) = parse_let_assignment(input)?;
    let (input, _) = preceded(opt_whitespace, tag("in")).parse(input)?;
    let (input, e2) = parse_expr(input)?;

    nom::IResult::Ok((
        input,
        assignments.iter().rev().fold(e2, |acc, (var, e)| {
            Expr::Let(var.clone(), Box::new(e.clone()), Box::new(acc.clone()))
        }),
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

fn parse_atomic_term(input: &str) -> IResult<&str, Expr> {
    preceded(
        opt_whitespace,
        alt((
            // Either (<lambda_definition>) ...
            delimited(tag("("), parse_expr, tag(")")),
            // add x y = (add x) y
            //parse_application_expr,
            // or a named lambda ex: add2 5
            parse_literal,
        )),
    ).parse(input)
}

pub fn parse_application_expr(input: &str) -> IResult<&str, Expr> {
    /*
     * (\(x: Int) -> x + 2) 5
     * add2 5
     * 5 + add2 5
     * isTrue true
     * add 2 5 == (add 2) 5
     * */
    let (input, initial) = parse_atomic_term(input)?;

    map(many0(parse_atomic_term), |exprs| {
        exprs.iter().fold(initial.clone(), |acc, arg| {
            Expr::App(Box::new(acc), Box::new(arg.clone()))
        })
    })
    .parse(input)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn parsing_let_assignment_sugar() {
        let input = "
x = 2,
y = 3
"
        .trim();

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

"
        .trim();
        let (_, result) = parse_let_expr(input).unwrap();
        let expected = Expr::Let(
            "x".to_string(),
            Box::new(Expr::Lit(Literal::Int(2))),
            Box::new(Expr::Let(
                "y".to_string(),
                Box::new(Expr::Lit(Literal::Int(3))),
                Box::new(Expr::BinOp(
                    Op::Add,
                    Box::new(Expr::Var("x".to_string())),
                    Box::new(Expr::Var("y".to_string())),
                )),
            )),
        );

        assert_eq!(result, expected);
    }

    #[test]
    fn test_parse_definition_delimiter_semicolon() {
        let input = ";";
        let result = parse_definition_delimiter(input);
        assert_eq!(result, Ok(("", ())));
    }

    #[test]
    fn test_parse_definition_delimiter_double_newline() {
        // Test simple double newline
        let input1 = "\n\n";
        let result1 = parse_definition_delimiter(input1);
        assert_eq!(result1, Ok(("", ())));

        // Test double newline with whitespace between
        let input2 = "\n  \n";
        let result2 = parse_definition_delimiter(input2);
        assert_eq!(result2, Ok(("", ())));
    }

    #[test]
    fn test_parse_top_level_let_with_delimiter() {
        // Test with semicolon
        let input1 = "let x = 5";
        let (remaining1, expr1) = parse_top_level_let(input1).unwrap();
        assert_eq!(remaining1, "");
        assert_eq!(
            expr1,
            Expr::Let(
                "x".to_string(),
                Box::new(Expr::Lit(Literal::Int(5))),
                Box::new(Expr::Var("x".to_string()))
            )
        );

        // Test with double newline
        let input2 = "let y = True";
        let (remaining2, expr2) = parse_top_level_let(input2).unwrap();
        assert_eq!(remaining2, "");
        assert_eq!(
            expr2,
            Expr::Let(
                "y".to_string(),
                Box::new(Expr::Lit(Literal::Bool(true))),
                Box::new(Expr::Var("y".to_string()))
            )
        );
    }

    #[test]
    fn test_parse_top_level_let_requires_let_keyword() {
        let input = "x = 5;";
        let result = parse_top_level_let(input);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_program_multiple_definitions() {
        let input = r"
let a = 1;
let b = 2

let c = 3".trim();

        let (remaining, expr) = parse_program(input).unwrap();
        assert_eq!(remaining, "");

        // Should create a nested let expression:
        // Let("a", 1, Let("b", 2, Let("c", 3, Var("c"))))
        let expected = Expr::Let(
            "a".to_string(),
            Box::new(Expr::Lit(Literal::Int(1))),
            Box::new(Expr::Let(
                "b".to_string(),
                Box::new(Expr::Lit(Literal::Int(2))),
                Box::new(Expr::Let(
                    "c".to_string(),
                    Box::new(Expr::Lit(Literal::Int(3))),
                    Box::new(Expr::Var("c".to_string()))
                ))
            ))
        );

        assert_eq!(expr, expected);
    }

    #[test]
    fn test_parse_simple_application() {
        let input = "compose inc inc";
        let result = parse_expr(input);
        println!("Parsing '{}'", input);
        println!("Result: {:?}", result);
        assert!(result.is_ok());
        let (remaining, expr) = result.unwrap();
        println!("Remaining: '{}'", remaining);
        println!("Expr: {:?}", expr);

        // Should parse as App(App(Var("compose"), Var("inc")), Var("inc"))
        match expr {
            Expr::App(_, _) => {},
            _ => panic!("Expected application expression, got {:?}", expr)
        }
    }

    #[test]
    fn test_addition_application() {
        let input = r"
(f 1) + (g 1)
".trim();

        let (remaining, expr) = parse_binary_expr(input).unwrap();
        assert_eq!(remaining, "");

        let expected = Expr::BinOp(Op::Add,
            Box::new(Expr::App(
                Box::new(Expr::Var("f".to_string())),
                Box::new(Expr::Lit(Literal::Int(1)))
            )),
            Box::new(Expr::App(
                Box::new(Expr::Var("g".to_string())),
                Box::new(Expr::Lit(Literal::Int(1)))
            ))
        );

        assert_eq!(expr, expected);
    }

    #[test]
    fn test_parse_five_statements() {
        let input = r"
let compose = \f -> \g -> \x -> f (g x)

let inc = \n -> n + 1

let not = \b -> if b then False else True

let f = compose inc inc

let g = compose not not
    "
        .trim();

        let (remaining, expr) = parse_program(input).unwrap();
        assert_eq!(remaining, "");

        // Should be: Let(compose, ..., Let(inc, ..., Let(not, ..., Let(f, ..., Let(g, ..., Var(g))))))
        if let Expr::Let(name1, _, body1) = expr {
            assert_eq!(name1, "compose");
            if let Expr::Let(name2, _, body2) = *body1 {
                assert_eq!(name2, "inc");
                if let Expr::Let(name3, _, body3) = *body2 {
                    assert_eq!(name3, "not");
                    if let Expr::Let(name4, _, body4) = *body3 {
                        assert_eq!(name4, "f");
                        if let Expr::Let(name5, _, body5) = *body4 {
                            assert_eq!(name5, "g");
                            if let Expr::Var(final_var) = *body5 {
                                assert_eq!(final_var, "g");
                            } else {
                                panic!("Expected Var(g)");
                            }
                        } else {
                            panic!("Expected Let(g, ...)");
                        }
                    } else {
                        panic!("Expected Let(f, ...)");
                    }
                } else {
                    panic!("Expected Let(not, ...)");
                }
            } else {
                panic!("Expected Let(inc, ...)");
            }
        } else {
            panic!("Expected Let(compose, ...)");
        }
    }
}
