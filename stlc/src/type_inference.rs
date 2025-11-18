use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    rc::Rc,
};

use crate::parser::{Expr, Op};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Int,
    Var(String),
    Bool,
    Arrow(Box<Type>, Box<Type>),
    // Represents something like forall a. a -> a
    Scheme(HashSet<String>, Box<Type>),
}

impl Type {
    fn free_vars(&self) -> HashSet<String> {
        match self {
            Type::Var(name) => {
                let v = vec![name.clone()];

              v.into_iter().collect()
            //    v.into_iter().filter(|x| !bound_vars.contains(&x)).collect()
            }
            Type::Arrow(t1, t2) => {
                let mut s: HashSet<String> = HashSet::new();
                let free_vars = t1.free_vars()
                    .into_iter()
                    .chain(t2.free_vars());

                for v in free_vars {
                    s.insert(v);
                }

                s.into_iter().collect()
            }
            Type::Scheme(items, _) => items.clone(),
            Type::Int | Type::Bool => HashSet::new(),
        }
    }

}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeError {
    Error(String),
}

pub fn infer_type(expr: &Expr) -> Result<Type, TypeError> {
    let binary = Type::Scheme(
        ["a".into()].into(),
        Box::new(Type::Arrow(
            Box::new(Type::Var("a".into())),
            Box::new(Type::Arrow(
                Box::new(Type::Var("a".into())),
                Box::new(Type::Var("a".into())),
            )),
        )),
    );

    let env = TypeEnv::new().append("binary", &binary);

    w(expr.clone(), env)
        .map_err(TypeError::Error)
        .map(|(_, t)| Ok(t))?
}

/// A substitution [ t -> v ] replaces v with t in a term
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Substitution(Vec<(Type, String)>);

/// Generates fresh Type::Var
#[derive(Debug, Clone)]
struct FreshVarNameGen(usize);

impl FreshVarNameGen {
    pub fn next(&mut self) -> String {
        let next_idx = self.0;
        self.0 += 1;

        format!("v{}", next_idx)
    }
}

/// The Type Environment
#[derive(Debug, Clone)]
pub struct TypeEnv {
    map: HashMap<String, Type>,
    var_gen: Rc<RefCell<FreshVarNameGen>>,
}

impl Default for TypeEnv {
    fn default() -> Self {
        Self::new()
    }
}

impl TypeEnv {
    pub fn new() -> Self {
        TypeEnv {
            map: HashMap::new(),
            var_gen: Rc::new(RefCell::new(FreshVarNameGen(0))),
        }
    }

    fn get(&self, name: &str) -> Option<Type> {
        self.map.get(name).cloned()
    }

    /// Get all the type variables defined in the environment
    fn type_vars(&self) -> Vec<&String> {
        let mut v = vec![];

        for value in self.map.values() {
            match value {
                Type::Var(name) => v.push(name),
                _ => continue,
            }
        }

        v
    }

    fn append(&self, name: &str, t: &Type) -> TypeEnv {
        let mut inner_map = self.map.clone();
        inner_map.insert(name.to_string(), t.clone());

        Self {
            map: inner_map,
            var_gen: self.var_gen.clone(),
        }
    }

    fn extend_fresh(&self, name: &str) -> TypeEnv {
        let fresh = self.var_gen.borrow_mut().next();

        self.append(name, &Type::Var(fresh))
    }

    fn apply_sub(&self, sub: &Substitution) -> TypeEnv {
        let mut new_set = HashMap::new();

        self.map.iter().for_each(|(name, t)| {
            new_set.insert(name.to_string(), sub.apply(t.clone()));
        });

        Self {
            map: new_set,
            var_gen: self.var_gen.clone(),
        }
    }

    fn get_fresh_var(&self) -> String {
        self.var_gen.borrow_mut().next()
    }
}

pub fn w(e: Expr, mut te: TypeEnv) -> Result<(Substitution, Type), String> {
    use Expr::*;
    match e {
        Var(name) => match te.get(&name) {
            Some(t) => Ok((Substitution::id(), instantiate(&t, &mut te))),
            None => Err(format!("Could not find variable '{name}'")),
        },

        App(e1, e2) => {
            let (s1, t1) = w(*e1.clone(), te.clone())?;
            let (s2, t2) = w(*e2.clone(), te.apply_sub(&s1))?;

            let u = te.get_fresh_var();
            let t2 = instantiate(&t2, &mut te);

            let t = Type::Arrow(Box::new(t2.clone()), Box::new(Type::Var(u.clone())));

            let t1 = instantiate(&t1, &mut te);
            let s = s2.compose(&s1);

            let t1 = s.apply(t1);
            let t = s.apply(t);

            if let Some(s3) = unify(t1.clone(), t.clone(), &mut te) {
                let s = s3.compose(&s);
                let t = s.apply(Type::Var(u));
                Ok((s.clone(), t))
            } else {
                Err(format!("Could not unify {t1:?} of {e1:?} with {t:?}"))
            }
        }

        Lambda(var_name, _type_ann, expr) => {
            let new_te = te.extend_fresh(&var_name);
            let u = new_te.get(&var_name).unwrap();

            let (s1, t1) = w(*expr, new_te)?;

            let t = Type::Arrow(Box::new(s1.apply(u)), Box::new(t1));
            println!("{t:?}");

            Ok((s1, t))
        }

        If(cond, t_e, f_e) => {
            let (s0, tc) = w(*cond, te.clone())?;

            // Make sure that the type of the conditional can be unified to Bool
            let s1 = unify(tc.clone(), Type::Bool, &mut te)
                .ok_or("Type of Conditional must unify to Bool")?;

            let (s2, t1) = w(*t_e, te.apply_sub(&s1))?;
            let (s3, t2) = w(*f_e, te.clone())?;

            let i1 = instantiate(&t1, &mut te);
            let i2 = instantiate(&t2, &mut te);

            if let Some(s4) = unify(s3.apply(i1), s3.apply(i2), &mut te) {
                let s = s4.compose(&s3).compose(&s2).compose(&s1).compose(&s0);

                Ok((s, s4.apply(t2.clone())))
            } else {
                Err("Invalid If type".to_string())
            }
        }

        Lit(literal) => Ok((
            Substitution::id(),
            match literal {
                crate::parser::Literal::Int(_) => Type::Int,
                crate::parser::Literal::Bool(_) => Type::Bool,
            },
        )),

        // For binary operations we are going to desugar this into a function
        BinOp(op, e1, e2) => {
            let desugared = Expr::App(
                Box::new(Expr::App(Box::new(Expr::Var("binary".into())), e1.clone())),
                e2.clone(),
            );

            let expected_t = match op {
                Op::Add | Op::Subtract => Type::Int,
                Op::And | Op::Or => Type::Bool,
            };

            let (s1, t1) = w(*e1.clone(), te.clone())?;
            let (s2, t2) = w(*e2.clone(), te.apply_sub(&s1))?;
            let s = s2.compose(&s1);

            let s3 = unify(s.apply(t1.clone()), expected_t.clone(), &mut te)
                    .ok_or("Could not unify {t1:?} with {expected_t:?}")?;

            let s4 = unify(s.apply(t2.clone()), expected_t.clone(), &mut te)
                    .ok_or("Could not unify {t1:?} with {expected_t:?}")?;

            let s = s4.compose(&s3).compose(&s);
            let (s5, t) = w(desugared, te.apply_sub(&s))?;
            let s = s5.compose(&s);

            unify(s.apply(t), expected_t.clone(), &mut te)
                .ok_or("Could not unify".into())
                .map(|s6| {
                    let s = s6.compose(&s);
                    println!("{s:?}");
                    (s, expected_t)
                })
        }

        Let(var_name, e1, e2) => {
            //1. Create a fresh type variable for x
            // pair : v0
            let sub_te = te.extend_fresh(&var_name);
            let a = sub_te.get(&var_name).unwrap();

            //2. Calculate the type of e1 to get t1 and s1
            let (s1, t1) = w(*e1.clone(), sub_te.clone())?;
            let t1 = generalize(&t1, &te);

            let t = instantiate(&t1, &mut te);
            println!(">> {t:?}");

            //3. Unify t1 and `a` to get s2, this should give us a substitution for the most general t1
            let s2 = unify(s1.apply(a.clone()), t.clone(), &mut te)
                .ok_or(format!("Could not unify {a:?} with {t1:?}").to_string())?;

            println!(">> add {var_name} :: {t1:?} to env");

            let sub_te = te.apply_sub(&s2.compose(&s1)).append(&var_name, &t1);
            let (s3, t2) = w(*e2.clone(), sub_te.clone())?;

            Ok((s3.compose(&s2.compose(&s1)), generalize(&t2, &te)))
        }
    }
}

/// Return all free variables in the type.
/// Anything that is defined inside the type environment is not considered a free variable.
fn get_free_vars(t: &Type, env: &TypeEnv) -> HashSet<String> {
    let bound_vars = env.type_vars();
    match t {
        Type::Var(name) => {
            let v = vec![name.clone()];

            v.into_iter().filter(|x| !bound_vars.contains(&x)).collect()
        }
        Type::Arrow(t1, t2) => {
            let mut s: HashSet<String> = HashSet::new();
            let free_vars = get_free_vars(t1, env)
                .into_iter()
                .chain(get_free_vars(t2, env));

            for v in free_vars {
                s.insert(v);
            }

            s.into_iter().collect()
        }
        Type::Scheme(items, _) => items.clone(),
        Type::Int | Type::Bool => HashSet::new(),
    }
}

/// Generalize a type T to forall X1..Xn.T where X1..Xn \in FV(T) and \not\in TypeEnv
fn generalize(t: &Type, env: &TypeEnv) -> Type {
    match t {
        Type::Arrow(_, _) => {
            let free_vars = get_free_vars(t, env);
            // Get the free type variables in the term t

            if !free_vars.is_empty() {
                // Create scheme
                Type::Scheme(free_vars, Box::new(t.clone()))
            } else {
                t.clone()
            }
        }
        _ => t.clone(),
    }
}

/// Instantiates the type t to create a new type
pub fn instantiate(t: &Type, env: &mut TypeEnv) -> Type {
    match t {
        Type::Scheme(vars, typ) => {
            let s = Substitution(
                vars.iter()
                    .map(|x| (Type::Var(env.get_fresh_var()), x.clone()))
                    .collect(),
            );

            s.apply(*typ.clone())
        }
        _ => t.clone(),
    }
}

fn apply_sub(t: Type, sub: &(Type, String)) -> Type {
    let (new_t, var_name) = &sub;
    match t {
        Type::Int | Type::Bool => t.clone(),
        Type::Var(ref name) => {
            if *name == *var_name {
                new_t.clone()
            } else {
                t
            }
        }
        Type::Arrow(t1, t2) => {
            Type::Arrow(Box::new(apply_sub(*t1, sub)), Box::new(apply_sub(*t2, sub)))
        }

        Type::Scheme(_, _) => t.clone(),
    }
}

impl Substitution {
    pub fn new(pairs: &[(Type, &str)]) -> Self {
        pairs
            .iter()
            .map(|(t, s)| (t.clone(), s.to_string()).into())
            .fold(Self::id(), |prev, item| prev.compose(&item))
    }
    pub fn id() -> Self {
        Self(vec![])
    }

    fn compose(&self, other: &Substitution) -> Substitution {
        Substitution(other.0.iter().chain(self.0.iter()).cloned().collect())
    }

    pub fn apply(&self, t: Type) -> Type {
        self.0.iter().fold(t, apply_sub)
    }
}

impl From<(Type, String)> for Substitution {
    fn from(value: (Type, String)) -> Self {
        Substitution(vec![value])
    }
}

pub fn unify(s: Type, t: Type, env: &mut TypeEnv) -> Option<Substitution> {
    use Type::*;

    match (s.clone(), t.clone()) {
        (Int, Int) | (Bool, Bool) => Some(Substitution::id()),
        (Var(n1), Var(n2)) => {
            // We create a fresh variable and map both to the new one.
            let fresh = env.get_fresh_var();
            Some(Substitution::new(&[
                (Type::Var(fresh.clone()), &n1),
                (Type::Var(fresh), &n2),
            ]))
        }
        (typ, Var(x)) | (Var(x), typ) if !typ.free_vars().contains(&x) => Some((typ.clone(), x).into()),
        (Arrow(t11, t12), Arrow(t21, t22)) => {
            let s1 = unify(*t11, *t21, env)?;
            let s2 = unify(s1.apply(*t12), s1.apply(*t22), env)?;

            Some(s2.compose(&s1))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use crate::parser::parse_lambda;

    use super::*;

    #[test]
    fn apply_sub_primitives_unchanged() {
        let sub = (Type::Bool, "a".to_string());
        assert_eq!(apply_sub(Type::Int, &sub), Type::Int);
        assert_eq!(apply_sub(Type::Bool, &sub), Type::Bool);
    }

    #[test]
    fn apply_sub_matching_var() {
        let sub = (Type::Int, "a".to_string());
        let result = apply_sub(Type::Var("a".to_string()), &sub);
        assert_eq!(result, Type::Int);
    }

    #[test]
    fn apply_sub_non_matching_var() {
        let sub = (Type::Int, "a".to_string());
        let result = apply_sub(Type::Var("b".to_string()), &sub);
        assert_eq!(result, Type::Var("b".to_string()));
    }

    #[test]
    fn apply_sub_arrow_type() {
        let sub = (Type::Int, "a".to_string());
        let arrow = Type::Arrow(
            Box::new(Type::Var("a".to_string())),
            Box::new(Type::Var("a".to_string())),
        );
        let result = apply_sub(arrow, &sub);
        assert_eq!(
            result,
            Type::Arrow(Box::new(Type::Int), Box::new(Type::Int))
        );
    }

    #[test]
    fn apply_sub_nested_arrow() {
        let sub = (Type::Int, "a".to_string());
        let arrow = Type::Arrow(
            Box::new(Type::Arrow(
                Box::new(Type::Var("a".to_string())),
                Box::new(Type::Var("b".to_string())),
            )),
            Box::new(Type::Var("a".to_string())),
        );
        let result = apply_sub(arrow, &sub);
        let expected = Type::Arrow(
            Box::new(Type::Arrow(
                Box::new(Type::Int),
                Box::new(Type::Var("b".to_string())),
            )),
            Box::new(Type::Int),
        );
        assert_eq!(result, expected);
    }

    #[test]
    fn id_substitution_is_empty() {
        let id = Substitution::id();
        assert_eq!(id.0.len(), 0);
    }

    #[test]
    fn id_substitution_leaves_types_unchanged() {
        let id = Substitution::id();
        assert_eq!(id.apply(Type::Int), Type::Int);
        assert_eq!(id.apply(Type::Bool), Type::Bool);
        assert_eq!(
            id.apply(Type::Var("a".to_string())),
            Type::Var("a".to_string())
        );
    }

    #[test]
    fn substitution_apply_preserves_primitives() {
        let sub = Substitution::from((Type::Int, "a".to_string()));
        assert_eq!(sub.apply(Type::Int), Type::Int);
        assert_eq!(sub.apply(Type::Bool), Type::Bool);
    }

    #[test]
    fn substitution_apply_single_var() {
        let sub = Substitution::from((Type::Int, "a".to_string()));
        let result = sub.apply(Type::Var("a".to_string()));
        assert_eq!(result, Type::Int);
    }

    #[test]
    fn substitution_apply_arrow_type() {
        let sub = Substitution::from((Type::Int, "a".to_string()));
        let arrow = Type::Arrow(Box::new(Type::Var("a".to_string())), Box::new(Type::Bool));
        let result = sub.apply(arrow);
        assert_eq!(
            result,
            Type::Arrow(Box::new(Type::Int), Box::new(Type::Bool))
        );
    }

    #[test]
    fn substitution_compose_with_id() {
        let sub = Substitution::from((Type::Int, "a".to_string()));
        let id = Substitution::id();
        let result = sub.compose(&id);
        assert_eq!(result.0.len(), 1);
    }

    #[test]
    fn substitution_compose_two_substitutions() {
        let s1 = Substitution::from((Type::Int, "a".to_string()));
        let s2 = Substitution::from((Type::Bool, "b".to_string()));
        let result = s1.compose(&s2);
        assert_eq!(result.0.len(), 2);
    }

    #[test]
    fn unify_identical_primitives() {
        let result = unify(Type::Int, Type::Int, &mut TypeEnv::new());
        assert!(result.is_some());
        assert_eq!(result.unwrap().0.len(), 0);

        let result = unify(Type::Bool, Type::Bool, &mut TypeEnv::new());
        assert!(result.is_some());
        assert_eq!(result.unwrap().0.len(), 0);
    }

    #[test]
    fn unify_different_primitives() {
        let result = unify(Type::Int, Type::Bool, &mut TypeEnv::new());
        assert!(result.is_none());
    }

    #[test]
    fn unify_var_with_type() {
        let result = unify(Type::Var("a".to_string()), Type::Int, &mut TypeEnv::new());
        assert!(result.is_some());
        let sub = result.unwrap();
        assert_eq!(sub.0.len(), 1);
        assert_eq!(sub.apply(Type::Var("a".to_string())), Type::Int);
    }

    #[test]
    fn unify_type_with_var() {
        let result = unify(Type::Int, Type::Var("a".to_string()), &mut TypeEnv::new());
        assert!(result.is_some());
        let sub = result.unwrap();
        assert_eq!(sub.apply(Type::Var("a".to_string())), Type::Int);
    }

    #[test]
    fn unify_two_vars() {
        let result = unify(
            Type::Var("a".to_string()),
            Type::Var("b".to_string()),
            &mut TypeEnv::new(),
        );
        assert!(result.is_some());
    }

    #[test]
    fn unify_arrow_types_identical() {
        let arrow1 = Type::Arrow(Box::new(Type::Int), Box::new(Type::Bool));
        let arrow2 = Type::Arrow(Box::new(Type::Int), Box::new(Type::Bool));
        let result = unify(arrow1, arrow2, &mut TypeEnv::new());
        assert!(result.is_some());
    }

    #[test]
    fn unify_arrow_with_vars() {
        let arrow1 = Type::Arrow(
            Box::new(Type::Var("a".to_string())),
            Box::new(Type::Var("b".to_string())),
        );
        let arrow2 = Type::Arrow(Box::new(Type::Int), Box::new(Type::Bool));
        let result = unify(arrow1, arrow2, &mut TypeEnv::new());
        assert!(result.is_some());
    }

    #[test]
    fn unify_incompatible_arrows() {
        let arrow1 = Type::Arrow(Box::new(Type::Int), Box::new(Type::Bool));
        let arrow2 = Type::Arrow(Box::new(Type::Bool), Box::new(Type::Int));
        let result = unify(arrow1, arrow2, &mut TypeEnv::new());
        assert!(result.is_none());
    }

    #[test]
    fn unify_arrow_with_primitive() {
        let arrow = Type::Arrow(Box::new(Type::Int), Box::new(Type::Bool));
        let result = unify(arrow, Type::Int, &mut TypeEnv::new());
        assert!(result.is_none());
    }

    #[test]
    fn infer_identity() {
        let e = Expr::Lambda("x".to_string(), None, Box::new(Expr::Var("x".to_string())));
        let id = Type::Arrow(
            Box::new(Type::Var("v0".to_string())),
            Box::new(Type::Var("v0".to_string())),
        );

        assert_eq!(infer_type(&e), Ok(id));
    }

    #[test]
    fn maybe_function() {
        use Type::*;

        let (_, es) =
            parse_lambda(r"\t -> \f -> \c -> if c then t else f").expect("Should parse if");
        let expected = Arrow(
            Box::new(Var("a".to_string())),
            Box::new(Arrow(
                Box::new(Var("a".to_string())),
                Box::new(Arrow(
                    Box::new(Type::Bool),
                    Box::new(Type::Var("a".to_string())),
                )),
            )),
        );
        let t = infer_type(&es).unwrap();

        let s = unify(t.clone(), expected.clone(), &mut TypeEnv::new()).unwrap();

        assert_eq!(s.apply(t), s.apply(expected));
    }
}

#[cfg(test)]
mod test {
    use crate::parser::parse_program;

    use super::*;

    fn id_type() -> Type {
        Type::Arrow(
            Box::new(Type::Var("a".to_string())),
            Box::new(Type::Var("a".to_string())),
        )
    }

    #[cfg(test)]
    mod scheme {
        use super::*;

        #[test]
        fn simple_generalize() {
            let t = id_type();
            let env = TypeEnv::new();
            let scheme = generalize(&t, &env);

            assert_eq!(
                scheme,
                Type::Scheme(["a".to_string()].into(), Box::new(id_type()))
            );
        }

        #[test]
        fn simple_instantiate() {
            let t = id_type();
            let mut env = TypeEnv::new();
            let scheme = generalize(&t, &env);

            let id = instantiate(&scheme, &mut env);

            assert_ne!(scheme, id);
            assert_ne!(id, t);
        }

        #[test]
        fn should_not_be_a_scheme() {
            let input = r"let sub = \a -> \b -> a - b";
            let (input, expr) = parse_program(input).unwrap();
            let t = infer_type(&expr).unwrap();

            let expected = Type::Arrow(
                Box::new(Type::Int),
                Box::new(Type::Arrow(Box::new(Type::Int), Box::new(Type::Int))),
            );

            assert_eq!(input, "");
            assert_eq!(t, expected);
        }

    }

    #[cfg(test)]
    mod unify {
        use super::*;

        #[test]
        fn occurs_check_simple() {
            // Should fail: unifying a variable with a type that contains it
            // e.g., unify(a, a -> b) where a is free in (a -> b)
            let mut env = TypeEnv::new();
            let var_a = Type::Var("a".to_string());
            let arrow_with_a = Type::Arrow(
                Box::new(Type::Var("a".to_string())),
                Box::new(Type::Var("b".to_string())),
            );

            let result = unify(var_a, arrow_with_a, &mut env);
            assert!(result.is_none(), "Occurs check should prevent infinite types");
        }

        #[test]
        fn occurs_check_nested() {
            let mut env = TypeEnv::new();
            let var_a = Type::Var("a".to_string());
            let nested_arrow = Type::Arrow(
                Box::new(Type::Var("b".to_string())),
                Box::new(Type::Var("a".to_string())),
            );

            let result = unify(var_a, nested_arrow, &mut env);
            assert!(result.is_none(), "Occurs check should catch variable in nested position");
        }

        #[test]
        fn occurs_check_deep_nesting() {
            let mut env = TypeEnv::new();
            let var_a = Type::Var("a".to_string());
            let deep_arrow = Type::Arrow(
                Box::new(Type::Arrow(
                    Box::new(Type::Var("a".to_string())),
                    Box::new(Type::Var("b".to_string())),
                )),
                Box::new(Type::Var("c".to_string())),
            );

            let result = unify(var_a, deep_arrow, &mut env);
            assert!(result.is_none());
        }

        // === Nested Arrow Unification ===

        #[test]
        fn deeply_nested_arrows() {
            // (a -> b) -> c  unified with  (Int -> Bool) -> d
            let arrow1 = Type::Arrow(
                Box::new(Type::Arrow(
                    Box::new(Type::Var("a".to_string())),
                    Box::new(Type::Var("b".to_string())),
                )),
                Box::new(Type::Var("c".to_string())),
            );

            let arrow2 = Type::Arrow(
                Box::new(Type::Arrow(
                    Box::new(Type::Int),
                    Box::new(Type::Bool),
                )),
                Box::new(Type::Var("d".to_string())),
            );

            let result = unify(arrow1, arrow2, &mut TypeEnv::new());
            assert!(result.is_some());

            let sub = result.unwrap();
            assert_eq!(sub.apply(Type::Var("a".to_string())), Type::Int);
            assert_eq!(sub.apply(Type::Var("b".to_string())), Type::Bool);
        }

        #[test]
        fn triple_nested_arrows() {
            // ((a -> b) -> c) -> d  with  ((Int -> Bool) -> e) -> f
            let arrow1 = Type::Arrow(
                Box::new(Type::Arrow(
                    Box::new(Type::Arrow(
                        Box::new(Type::Var("a".to_string())),
                        Box::new(Type::Var("b".to_string())),
                    )),
                    Box::new(Type::Var("c".to_string())),
                )),
                Box::new(Type::Var("d".to_string())),
            );

            let arrow2 = Type::Arrow(
                Box::new(Type::Arrow(
                    Box::new(Type::Arrow(
                        Box::new(Type::Int),
                        Box::new(Type::Bool),
                    )),
                    Box::new(Type::Var("e".to_string())),
                )),
                Box::new(Type::Var("f".to_string())),
            );

            let result = unify(arrow1, arrow2, &mut TypeEnv::new());
            assert!(result.is_some());
        }

        // === Substitution Propagation ===

        #[test]
        fn substitution_propagates_through_arrows() {
            // (a -> a) with (Int -> b) should result in a=Int, b=Int
            let arrow1 = Type::Arrow(
                Box::new(Type::Var("a".to_string())),
                Box::new(Type::Var("a".to_string())),
            );

            let arrow2 = Type::Arrow(
                Box::new(Type::Int),
                Box::new(Type::Var("b".to_string())),
            );

            let result = unify(arrow1, arrow2, &mut TypeEnv::new());
            assert!(result.is_some());

            let sub = result.unwrap();
            // Both a and b should be Int
            assert_eq!(sub.apply(Type::Var("a".to_string())), Type::Int);
            assert_eq!(sub.apply(Type::Var("b".to_string())), Type::Int);
        }

        #[test]
        fn chained_variable_constraints() {
            // (a -> b) with (c -> d) should create fresh var and map all to it
            let arrow1 = Type::Arrow(
                Box::new(Type::Var("a".to_string())),
                Box::new(Type::Var("b".to_string())),
            );

            let arrow2 = Type::Arrow(
                Box::new(Type::Var("c".to_string())),
                Box::new(Type::Var("d".to_string())),
            );

            let result = unify(arrow1, arrow2, &mut TypeEnv::new());
            assert!(result.is_some());

            let sub = result.unwrap();
            // After substitution, a and c should unify, b and d should unify
            let a_result = sub.apply(Type::Var("a".to_string()));
            let c_result = sub.apply(Type::Var("c".to_string()));
            let b_result = sub.apply(Type::Var("b".to_string()));
            let d_result = sub.apply(Type::Var("d".to_string()));

            assert_eq!(a_result, c_result);
            assert_eq!(b_result, d_result);
        }

        // === Reflexivity and Symmetry ===

        #[test]
        fn reflexivity_complex_arrow() {
            // Any type should unify with itself
            let complex = Type::Arrow(
                Box::new(Type::Arrow(
                    Box::new(Type::Int),
                    Box::new(Type::Var("a".to_string())),
                )),
                Box::new(Type::Bool),
            );

            let result = unify(complex.clone(), complex.clone(), &mut TypeEnv::new());
            assert!(result.is_some());
        }

        #[test]
        fn symmetry_var_and_concrete() {
            // unify(Var, Int) should equal unify(Int, Var)
            let mut env1 = TypeEnv::new();
            let result1 = unify(Type::Var("a".to_string()), Type::Int, &mut env1);

            let mut env2 = TypeEnv::new();
            let result2 = unify(Type::Int, Type::Var("a".to_string()), &mut env2);

            assert!(result1.is_some());
            assert!(result2.is_some());

            // Both should produce equivalent substitutions
            assert_eq!(
                result1.unwrap().apply(Type::Var("a".to_string())),
                Type::Int
            );
            assert_eq!(
                result2.unwrap().apply(Type::Var("a".to_string())),
                Type::Int
            );
        }

        #[test]
        fn symmetry_arrows() {
            // Order shouldn't matter for arrow unification
            let arrow1 = Type::Arrow(
                Box::new(Type::Var("a".to_string())),
                Box::new(Type::Int),
            );
            let arrow2 = Type::Arrow(
                Box::new(Type::Bool),
                Box::new(Type::Var("b".to_string())),
            );

            let result1 = unify(arrow1.clone(), arrow2.clone(), &mut TypeEnv::new());
            let result2 = unify(arrow2, arrow1, &mut TypeEnv::new());

            assert_eq!(result1.is_some(), result2.is_some());
        }

        // === Scheme Handling ===

        #[test]
        fn scheme_with_type_fails() {
            // Schemes should not unify with regular types
            let scheme = Type::Scheme(
                ["a".to_string()].into(),
                Box::new(Type::Arrow(
                    Box::new(Type::Var("a".to_string())),
                    Box::new(Type::Var("a".to_string())),
                )),
            );

            let result = unify(scheme, Type::Int, &mut TypeEnv::new());
            assert!(result.is_none());
        }

        #[test]
        fn scheme_with_arrow_fails() {
            let scheme = Type::Scheme(
                ["a".to_string()].into(),
                Box::new(Type::Var("a".to_string())),
            );

            let arrow = Type::Arrow(Box::new(Type::Int), Box::new(Type::Bool));

            let result = unify(scheme, arrow, &mut TypeEnv::new());
            assert!(result.is_none());
        }

        #[test]
        fn two_schemes_fails() {
            let scheme1 = Type::Scheme(
                ["a".to_string()].into(),
                Box::new(Type::Var("a".to_string())),
            );

            let scheme2 = Type::Scheme(
                ["b".to_string()].into(),
                Box::new(Type::Var("b".to_string())),
            );

            let result = unify(scheme1, scheme2, &mut TypeEnv::new());
            assert!(result.is_none());
        }

        // === Complex Arrow Patterns ===

        #[test]
        fn curried_function_types() {
            // (a -> b -> c) with (Int -> Bool -> d)
            let curried1 = Type::Arrow(
                Box::new(Type::Var("a".to_string())),
                Box::new(Type::Arrow(
                    Box::new(Type::Var("b".to_string())),
                    Box::new(Type::Var("c".to_string())),
                )),
            );

            let curried2 = Type::Arrow(
                Box::new(Type::Int),
                Box::new(Type::Arrow(
                    Box::new(Type::Bool),
                    Box::new(Type::Var("d".to_string())),
                )),
            );

            let result = unify(curried1, curried2, &mut TypeEnv::new());
            assert!(result.is_some());

            let sub = result.unwrap();
            assert_eq!(sub.apply(Type::Var("a".to_string())), Type::Int);
            assert_eq!(sub.apply(Type::Var("b".to_string())), Type::Bool);
        }

        #[test]
        fn mixed_concrete_and_vars() {
            // (Int -> a) with (b -> Bool)
            let arrow1 = Type::Arrow(
                Box::new(Type::Int),
                Box::new(Type::Var("a".to_string())),
            );

            let arrow2 = Type::Arrow(
                Box::new(Type::Var("b".to_string())),
                Box::new(Type::Bool),
            );

            let result = unify(arrow1, arrow2, &mut TypeEnv::new());
            assert!(result.is_some());

            let sub = result.unwrap();
            assert_eq!(sub.apply(Type::Var("b".to_string())), Type::Int);
            assert_eq!(sub.apply(Type::Var("a".to_string())), Type::Bool);
        }

        // === Edge Cases ===

        #[test]
        fn same_variable_both_sides() {
            // a with a should succeed
            let result = unify(
                Type::Var("a".to_string()),
                Type::Var("a".to_string()),
                &mut TypeEnv::new(),
            );
            assert!(result.is_some());
        }

        #[test]
        fn var_with_arrow_containing_different_var() {
            // a with (b -> c) where a is not in (b -> c)
            let var_a = Type::Var("a".to_string());
            let arrow_bc = Type::Arrow(
                Box::new(Type::Var("b".to_string())),
                Box::new(Type::Var("c".to_string())),
            );

            let result = unify(var_a.clone(), arrow_bc.clone(), &mut TypeEnv::new());
            assert!(result.is_some());

            let sub = result.unwrap();
            assert_eq!(sub.apply(var_a), arrow_bc);
        }

        #[test]
        fn incompatible_nested_structure() {
            // (a -> Int) with (Bool -> (c -> d)) should fail
            // because Int cannot unify with (c -> d)
            let arrow1 = Type::Arrow(
                Box::new(Type::Var("a".to_string())),
                Box::new(Type::Int),
            );

            let arrow2 = Type::Arrow(
                Box::new(Type::Bool),
                Box::new(Type::Arrow(
                    Box::new(Type::Var("c".to_string())),
                    Box::new(Type::Var("d".to_string())),
                )),
            );

            let result = unify(arrow1, arrow2, &mut TypeEnv::new());
            assert!(result.is_none());
        }
    }
}

