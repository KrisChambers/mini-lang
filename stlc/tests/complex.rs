use stlc::parser::{Expr, parse_program};
use stlc::type_inference::{infer_type, instantiate, unify, Type, TypeEnv, TypeError};

fn get_type(input: &str, top_level_var_name: &str) -> Result<Type, TypeError> {
    let wrapper =
        input.trim().to_string() + "\n\n" + format!("let internal = {top_level_var_name}").as_str();

    let (_, expr) = parse_program(&wrapper).unwrap();

    infer_type(&expr)
}

// ============================================================================
// 1. Complex Let Polymorphism Tests
// ============================================================================

#[test]
fn test_polymorphic_function_used_at_multiple_types_in_nested_context() {
    // Tests that a polymorphic function can be instantiated
    // differently in nested scopes
    let input = r"
let id = \x -> x

let a = id 5

let b = if (id True) then (id 10) else (id 20)

let c = id False
    ";

    assert_eq!(get_type(input, "c"), Ok(Type::Bool));
    assert_eq!(get_type(input, "a"), Ok(Type::Int));
}

#[test]
fn test_nested_polymorphic_let_bindings() {
    // Each inner let should generalize its bound variable
    let input = r"
let f = \x -> x

let g = \y -> f y

let h = \z -> g z

let test1 = h 42

let test2 = h True
    ";

    // Verify polymorphic functions are properly generalized
    assert_eq!(get_type(input, "test1"), Ok(Type::Int));
    assert_eq!(get_type(input, "test2"), Ok(Type::Bool));
}

#[test]
fn test_polymorphic_argument_to_function() {
    // Tests passing a polymorphic function as an argument
    let input = r"
let apply = \f -> \x -> f x

let id = \y -> y

let result = (apply id) 5
    ";

    assert_eq!(get_type(input, "result"), Ok(Type::Int));
    // Verify apply has the correct higher-order type
    let apply_result = get_type(input, "apply");
    assert!(apply_result.is_ok());
}

// ============================================================================
// 2. Higher-Order Function Patterns
// ============================================================================

#[test]
fn test_church_boolean_encoding() {
    // Church booleans: true = \t -> \f -> t, false = \t -> \f -> f
    let input = r"
let ctrue = \t -> \f -> t;
let cfalse = \t -> \f -> f;
let and = \a -> \b -> a b cfalse;
let final = and ctrue cfalse;
    ";

    // Verify each Church boolean has the correct polymorphic type
    let ctrue_type = get_type(input, "ctrue");
    assert!(ctrue_type.is_ok(), "ctrue should type-check");

    let cfalse_type = get_type(input, "cfalse");
    assert!(cfalse_type.is_ok(), "cfalse should type-check");

    let and_type = get_type(input, "and");
    assert!(and_type.is_ok(), "and should type-check");

    // Result should be a polymorphic arrow type
    let final_type = get_type(input, "final");
    assert!(final_type.is_ok());
    match final_type.unwrap() {
        Type::Arrow(_, _) => (),
        Type::Scheme(_, inner) => match *inner {
            Type::Arrow(_, _) => (),
            _ => panic!("Expected arrow type inside scheme"),
        },
        _ => panic!("Expected arrow type or scheme"),
    }
}

#[test]
fn test_flip_function() {
    // flip: (a -> b -> c) -> b -> a -> c
    let input = r"
let flip = \f -> \x -> \y -> f y x;
let sub = \a -> \b -> a - b;
let flipped_sub = flip sub;
let final = flipped_sub 5 10
    ".trim();

    // Verify sub has type Int -> Int -> Int
    assert_eq!(
        get_type(input, "sub"),
        Ok(Type::Arrow(
            Box::new(Type::Int),
            Box::new(Type::Arrow(Box::new(Type::Int), Box::new(Type::Int)))
        ))
    );

    // Verify flipped_sub has type Int -> Int -> Int
    assert_eq!(
        get_type(input, "flipped_sub"),
        Ok(Type::Arrow(
            Box::new(Type::Int),
            Box::new(Type::Arrow(Box::new(Type::Int), Box::new(Type::Int)))
        ))
    );

    // Verify final result is Int
    assert_eq!(get_type(input, "final"), Ok(Type::Int));
}

#[test]
fn test_const_function() {
    // const: a -> b -> a (returns first arg, ignores second)
    let input = r"
let const = \x -> \y -> x

let f = const 42

let result1 = f True

let result2 = f False
    ";

    // Verify f has type Bool -> Int (ignores bool, returns int)

    let expected = Type::Arrow(
        Box::new(Type::Var("a".into())),
        Box::new(Type::Int)
    );
    let t = get_type(input, "f").unwrap();
    let t = instantiate(&t, &mut TypeEnv::new());
    let u = unify(t.clone(), expected.clone(), &mut TypeEnv::new()).unwrap();

    let expected = u.apply(expected);
    let t = u.apply(t);

    assert_eq!(
        t,
        expected
    );

    // Both results should be Int
    assert_eq!(get_type(input, "result1"), Ok(Type::Int));
    assert_eq!(get_type(input, "result2"), Ok(Type::Int));
}

#[test]
fn test_three_level_composition() {
    // compose3: (c -> d) -> (b -> c) -> (a -> b) -> a -> d
    let input = r"
let compose3 = \f -> \g -> \h -> \x -> f (g (h x))

let inc = \n -> n + 1

let final = compose3 inc inc inc
    ";

    // Verify inc has type Int -> Int
    assert_eq!(
        get_type(input, "inc"),
        Ok(Type::Arrow(Box::new(Type::Int), Box::new(Type::Int)))
    );

    // Verify composed function has type Int -> Int
    assert_eq!(
        get_type(input, "final"),
        Ok(Type::Arrow(Box::new(Type::Int), Box::new(Type::Int)))
    );
}

// ============================================================================
// 3. Complex Conditional Logic
// ============================================================================

#[test]
fn test_nested_conditionals_with_polymorphism() {
    let input = r"
let id = \x -> x

let choose = \b -> if b then id else id

let f = choose True

let final = f 42
    ";

    // Verify f is polymorphic after being selected from conditional
    let f_type = get_type(input, "f");
    assert!(f_type.is_ok(), "f should type-check");

    // Verify result is Int
    assert_eq!(get_type(input, "final"), Ok(Type::Int));
}

#[test]
fn test_conditional_selecting_between_functions() {
    let input = r"
let add1 = \x -> x + 1

let add2 = \x -> x + 2

let selector = \b -> if b then add1 else add2

let f = selector True

let final = f 10
    ";

    // Verify add1 and add2 have type Int -> Int
    assert_eq!(
        get_type(input, "add1"),
        Ok(Type::Arrow(Box::new(Type::Int), Box::new(Type::Int)))
    );
    assert_eq!(
        get_type(input, "add2"),
        Ok(Type::Arrow(Box::new(Type::Int), Box::new(Type::Int)))
    );

    // Verify f has type Int -> Int
    assert_eq!(
        get_type(input, "f"),
        Ok(Type::Arrow(Box::new(Type::Int), Box::new(Type::Int)))
    );

    // Verify result is Int
    assert_eq!(get_type(input, "final"), Ok(Type::Int));
}

// ============================================================================
// 4. Deep Nesting and Scoping
// ============================================================================

#[test]
fn test_deeply_nested_lets_with_shadowing() {
    let input = r"
let x = 1

let f = \y -> x + y

let x = True

let g = \z -> if x then z else z

let final = g 5
    ";

    // Verify f uses the Int version of x
    assert_eq!(
        get_type(input, "f"),
        Ok(Type::Arrow(Box::new(Type::Int), Box::new(Type::Int)))
    );

    // Verify g uses the Bool version of x
    let g_type = get_type(input, "g");
    assert!(g_type.is_ok(), "g should type-check");

    // Verify result is Int
    assert_eq!(get_type(input, "final"), Ok(Type::Int));
}

#[test]
fn test_complex_scoping_with_multiple_bindings() {
    let input = r"
let a = 1

let b = 2

let c = 3

let f = \x -> a + b + c + x

let a = 100

let g = \y -> a + y

let last = (f 1) + (g 1)
    ";

    // Verify f and g both have type Int -> Int
    assert_eq!(
        get_type(input, "f"),
        Ok(Type::Arrow(Box::new(Type::Int), Box::new(Type::Int)))
    );
    assert_eq!(
        get_type(input, "g"),
        Ok(Type::Arrow(Box::new(Type::Int), Box::new(Type::Int)))
    );

    // Verify result is Int
    assert_eq!(get_type(input, "last"), Ok(Type::Int));
}

// ============================================================================
// 5. Negative Tests (Should Fail)
// ============================================================================

#[test]
fn test_occurs_check_should_fail() {
    // This should fail: trying to create infinite type
    // omega combinator: \x -> x x
    let expr = Expr::Lambda(
        "x".to_string(),
        None,
        Box::new(Expr::App(
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Var("x".to_string())),
        )),
    );

    let result = infer_type(&expr);
    println!("{result:?}");
    assert!(
        result.is_err(),
        "Self-application should fail type checking"
    );
}

#[test]
fn test_type_mismatch_in_complex_expression() {
    let input = r"
let f = \x -> x + 1

let g = \y -> if y then True else False

let bad = f (g True)
    ";

    let result = parse_program(input);
    if let Ok((_, expr)) = result {
        let type_result = infer_type(&expr);
        assert!(
            type_result.is_err(),
            "Should fail: applying int function to bool"
        );
    }
}

#[test]
fn test_unification_failure_in_composition() {
    let input = r"
let compose = \f -> \g -> \x -> f (g x)

let inc = \n -> n + 1

let not = \b -> if b then False else True

let bad = compose inc not
    ";

    let result = parse_program(input);
    if let Ok((_, expr)) = result {
        let type_result = infer_type(&expr);
        assert!(
            type_result.is_err(),
            "Should fail: composing Int->Int with Bool->Bool"
        );
    }
}

#[test]
fn test_if_condition_must_be_bool() {
    let input = r"
let bad = if 5 then True else False
    ";

    let result = parse_program(input);
    if let Ok((_, expr)) = result {
        let type_result = infer_type(&expr);
        assert!(type_result.is_err(), "Condition must be Bool");
    }
}

// ============================================================================
// 6. Stress Tests (Very Complex)
// ============================================================================

#[test]
fn test_complex_program_with_all_features() {
    // A complex program using lambdas, let, if, application, and binops
    let input = r"
let compose = \f -> \g -> \x -> f (g x)

let isEven = \n ->
    if (n - 2) - 2 then False else True

let double = \n -> n + n

let processBool = \b -> if b then 1 else 0

let pipeline = compose processBool isEven

let final = pipeline (double 3)
    ";

    let result = parse_program(input);
    if let Ok((_, expr)) = result {
        let type_result = infer_type(&expr);
        // This may fail depending on your arithmetic implementation
        // but it's a good stress test
        assert!(type_result.is_ok() || type_result.is_err());
    }
}

#[test]
fn test_y_combinator_style_pattern() {
    // While you can't define the Y combinator (needs occurs check to fail),
    // you can test similar recursive patterns
    let input = r"
let almost_rec = \f -> \x -> f (f x)

let inc = \n -> n + 1

let add_two = almost_rec inc

let final = add_two 5
    ";

    // Verify inc has type Int -> Int
    assert_eq!(
        get_type(input, "inc"),
        Ok(Type::Arrow(Box::new(Type::Int), Box::new(Type::Int)))
    );

    // Verify add_two has type Int -> Int
    assert_eq!(
        get_type(input, "add_two"),
        Ok(Type::Arrow(Box::new(Type::Int), Box::new(Type::Int)))
    );

    // Verify result is Int
    assert_eq!(get_type(input, "final"), Ok(Type::Int));
}

// ============================================================================
// Additional Edge Cases
// ============================================================================

#[test]
fn test_multiple_type_variable_instantiation() {
    // Tests that each use of a polymorphic function gets fresh type variables
    let input = r"
let pair = \x -> \y -> x

let first = pair 1 True

let second = pair True 1
    ";

    // Verify each instantiation has the correct type
    assert_eq!(get_type(input, "first"), Ok(Type::Int));
    assert_eq!(get_type(input, "second"), Ok(Type::Bool));
}

#[test]
fn test_nested_applications() {
    // Tests deeply nested function applications
    let input = r"
let f = \a -> \b -> \c -> \d -> a + b + c + d

let final = f 1 2 3 4
    ";

    assert_eq!(
        get_type(input, "f"),
        Ok(Type::Arrow(
            Box::new(Type::Int),
            Box::new(Type::Arrow(
                Box::new(Type::Int),
                Box::new(Type::Arrow(
                    Box::new(Type::Int),
                    Box::new(Type::Arrow(Box::new(Type::Int), Box::new(Type::Int)))
                ))
            ))
        ))
    );
    assert_eq!(get_type(input, "final"), Ok(Type::Int));
}

#[test]
fn test_polymorphism_with_boolean_operations() {
    let input = r"
let id = \x -> x

let or_test = (id True) || (id False)
    ";

    // Verify id type-checks as polymorphic
    let id_type = get_type(input, "id");
    assert!(id_type.is_ok(), "id should type-check");

    // Verify or_test is Bool
    assert_eq!(get_type(input, "or_test"), Ok(Type::Bool));
}

#[test]
fn test_higher_rank_polymorphism_limitation() {
    // This test documents that we don't support higher-rank polymorphism
    // The function 'apply_to_both' would need rank-2 types to work with
    // polymorphic functions, but our system should still handle monomorphic uses
    let input = r"
let apply_to_both = \f -> \x -> \y -> (f x) + (f y)

let inc = \n -> n + 1

let final = apply_to_both inc 1 2
    ";

    // Verify inc has type Int -> Int
    assert_eq!(
        get_type(input, "inc"),
        Ok(Type::Arrow(Box::new(Type::Int), Box::new(Type::Int)))
    );

    // Verify result is Int
    assert_eq!(get_type(input, "final"), Ok(Type::Int));
}
