use std::{cell::RefCell, collections::{HashMap, HashSet}, rc::Rc};

use crate::parser::Expr;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Int,
    Var(String),
    Bool,
    Arrow(Box<Type>, Box<Type>),
    // Represents something like forall a. a -> a
    Scheme(HashSet<String>, Box<Type>),
}



#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeError {
    Error(String),
}

pub fn infer_type(expr: &Expr) -> Result<Type, TypeError> {
    let binary_int_f = Type::Arrow(
        Box::new(Type::Int),
        Box::new(Type::Arrow(
            Box::new(Type::Int),
            Box::new(Type::Int)
        )
        )
    );

    let binary_bool_f = Type::Arrow(
        Box::new(Type::Int),
        Box::new(Type::Arrow(
            Box::new(Type::Int),
            Box::new(Type::Int)
        )
        )
    );

    let env = TypeEnv::new()
        .append("binary_int", &binary_int_f)
        .append("binary_bool", &binary_bool_f);

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
            Some(t) => Ok((Substitution::id(), t)),
            None => Err(format!("Could not find variable '{name}'")),
        },
        App(e1, e2) => {
            let (s1, t1) = w(*e1.clone(), te.clone())?;
            //println!("app func {e1:?} : {t1:?}");
            let (s2, t2) = w(*e2.clone(), te.apply_sub(&s1))?;
            //println!("app arg  {e2:?} : {t2:?}");
            let u = te.get_fresh_var();
            let t = Type::Arrow(Box::new(t2), Box::new(Type::Var(u.clone())));
            let t1 = instantiate(&t1, &mut te);
            //println!("unify {t1:?} and {t:?}");
            //println!("s2 {s2:?}");
            //println!("s1 {s1:?}");
            if let Some(s3) = unify(s2.apply(t1.clone()), t.clone()) {
                let s = s3.compose(&s2).compose(&s1);
             //   println!("sub {s3:?}");
                let t = s.apply(Type::Var(u));
                //println!("final T = {t:?}");
                Ok((s.clone(), t))
            } else {
                Err(format!("Could not unify {t1:?} of {e1:?} with {t:?}"))
            }
        }
        Lambda(var_name, _type_ann, expr) => {
            let new_te = te.extend_fresh(&var_name);
            let u = new_te.get(&var_name).unwrap();
            println!("{te:?}");

            let (s1, t1) = w(*expr, new_te)?;
            let t = Type::Arrow(Box::new(s1.apply(u)), Box::new(t1));

            println!("{s1:?}");

            Ok((s1, t))
        }
        If(cond, t_e, f_e) => {
            let (s0, tc) = w(*cond, te.clone())?;

            // Make sure that the type of the conditional can be unified to Bool
            let s1 =
                unify(tc.clone(), Type::Bool).ok_or("Type of Conditional must unify to Bool")?;

            let (s2, t1) = w(*t_e, te.apply_sub(&s1))?;
            let (s3, t2) = w(*f_e, te.clone())?;

            if let Some(s4) = unify(s3.apply(t1), s3.apply(t2.clone())) {
                let s = s4
                    .compose(&s3)
                    .compose(&s2)
                    .compose(&s1)
                    .compose(&s0);

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
        BinOp(op, e1, e2) => {
            // e1 + e2
            // This needs to be thought out a bit more.

            use crate::parser::Op::*;
            let desugared_func = match op {
                Add | Subtract => "binary_int".to_string(),
                And | Or => "binary_bool".to_string(),
            };

            let desugared = Expr::App(
                Box::new(Expr::App(
                    Box::new(Expr::Var(desugared_func)),
                    e1.clone()
                )),
                e2.clone()
            );

            // We desugar the expression and throw it back through w

            w(desugared, te.clone())
        }
        Let(var_name, e1, e2) => {
            //println!("");
            // TODO(kc):
            // We need to implement the generalization and instantiation
            // id = \x -> x Should be typed as forall a. a -> a
            // When we want to do something like id 5.
            // We first look up the type of id.
            //   Then we instantiate it with a fresh type variable
            //      This gives us a type b -> b
            //          We then unify this type with type of 5
            //      Same goes for id True
            //
            //      We are not properly handling the generalization / instantiation of types

            // First we assign a type to the variable `var_name`

            // let x = e1 in e2

            //1. Create a fresh type variable for x
            let sub_te = te.extend_fresh(&var_name);
            let a = sub_te.get(&var_name).unwrap();
            //println!("{var_name:?} : {a:?}");

            // Next we infer the type of e1 and then unify with var_name
            // I think it is here that we want to create a TypeScheme.
            // TODO (kc): GO through some pen and pencil examples for this.

            // The type here should always end up being a type scheme (maybe w should always return a type
            // scheme?)

            //2. Calculate the type of e1 to get t1 and s1
            let (s1, t1) = w(*e1.clone(), sub_te.clone())?;
            let t1 = generalize(&t1, &te);
            //println!("{e1:?} : {t1:?}");

            //3. Unify t1 and `a` to get s2, this should give us a substitution for the most general t1
            let s2 = unify(s1.apply(a), t1.clone()).ok_or("Invalid".to_string())?;
            //println!("unify via {s2:?}");


            let sub_te = te.apply_sub(&s2.compose(&s1)).append(&var_name, &t1);
            //println!("sub_te :: {sub_te:?}");
            let (s3, t2) = w(*e2, sub_te.clone())?;

            Ok((s3.compose(&s2.compose(&s1)), t2.clone()))
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
    // Get the free type variables in the term t
    let free_vars = get_free_vars(t, env);

    if !free_vars.is_empty() {
        // Create scheme
        Type::Scheme(free_vars, Box::new(t.clone()))
    } else {
        t.clone()
    }

}

/// Instantiates the type t to create a new type
fn instantiate(t: &Type, env: &mut TypeEnv) -> Type {
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
    fn id() -> Self {
        Self(vec![])
    }

    fn compose(&self, other: &Substitution) -> Substitution {
        Substitution(other.0.iter().chain(self.0.iter()).cloned().collect())
    }

    fn apply(&self, t: Type) -> Type {
        self.0.iter().fold(t, apply_sub)
    }
}

impl From<(Type, String)> for Substitution {
    fn from(value: (Type, String)) -> Self {
        Substitution(vec![value])
    }
}

fn unify(t1: Type, t2: Type) -> Option<Substitution> {
    use Type::*;

    match (t1, t2) {
        (Int, Int) | (Bool, Bool) => Some(Substitution::id()),
        (typ, Var(name)) | (Var(name), typ) => Some((typ, name).into()),
        //(Var(name), typ) => Some((typ, name).into()),
        (Arrow(t11, t12), Arrow(t21, t22)) => {
            let s1 = unify(*t11, *t21)?;
            let s2 = unify(s1.apply(*t12), s1.apply(*t22))?;

            Some(s2.compose(&s1))
        }
        (Int | Bool, _) => None,
        (_, Int | Bool) => None,
        _ => None, //(Arrow(_, _), Scheme(items, _)) => todo!(),
                   //(Scheme(items, _), Arrow(_, _)) => todo!(),
                   //(Scheme(items, _), Scheme(items, _)) => todo!(),
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
        let result = unify(Type::Int, Type::Int);
        assert!(result.is_some());
        assert_eq!(result.unwrap().0.len(), 0);

        let result = unify(Type::Bool, Type::Bool);
        assert!(result.is_some());
        assert_eq!(result.unwrap().0.len(), 0);
    }

    #[test]
    fn unify_different_primitives() {
        let result = unify(Type::Int, Type::Bool);
        assert!(result.is_none());
    }

    #[test]
    fn unify_var_with_type() {
        let result = unify(Type::Var("a".to_string()), Type::Int);
        assert!(result.is_some());
        let sub = result.unwrap();
        assert_eq!(sub.0.len(), 1);
        assert_eq!(sub.apply(Type::Var("a".to_string())), Type::Int);
    }

    #[test]
    fn unify_type_with_var() {
        let result = unify(Type::Int, Type::Var("a".to_string()));
        assert!(result.is_some());
        let sub = result.unwrap();
        assert_eq!(sub.apply(Type::Var("a".to_string())), Type::Int);
    }

    #[test]
    fn unify_two_vars() {
        let result = unify(Type::Var("a".to_string()), Type::Var("b".to_string()));
        assert!(result.is_some());
    }

    #[test]
    fn unify_arrow_types_identical() {
        let arrow1 = Type::Arrow(Box::new(Type::Int), Box::new(Type::Bool));
        let arrow2 = Type::Arrow(Box::new(Type::Int), Box::new(Type::Bool));
        let result = unify(arrow1, arrow2);
        assert!(result.is_some());
    }

    #[test]
    fn unify_arrow_with_vars() {
        let arrow1 = Type::Arrow(
            Box::new(Type::Var("a".to_string())),
            Box::new(Type::Var("b".to_string())),
        );
        let arrow2 = Type::Arrow(Box::new(Type::Int), Box::new(Type::Bool));
        let result = unify(arrow1, arrow2);
        assert!(result.is_some());
    }

    #[test]
    fn unify_incompatible_arrows() {
        let arrow1 = Type::Arrow(Box::new(Type::Int), Box::new(Type::Bool));
        let arrow2 = Type::Arrow(Box::new(Type::Bool), Box::new(Type::Int));
        let result = unify(arrow1, arrow2);
        assert!(result.is_none());
    }

    #[test]
    fn unify_arrow_with_primitive() {
        let arrow = Type::Arrow(Box::new(Type::Int), Box::new(Type::Bool));
        let result = unify(arrow, Type::Int);
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
        // if function
        // \b -> \t -> \f -> if b then t else f
        use Type::*;

        let (_, es) =
            parse_lambda(r"\t -> \f -> \c -> if c then t else f").expect("Should parse if");
        let expected = Arrow(
            Box::new(Var("v0".to_string())),
            Box::new(Arrow(
                Box::new(Var("v0".to_string())),
                Box::new(Arrow(
                    Box::new(Type::Bool),
                    Box::new(Type::Var("v0".to_string())),
                )),
            )),
        );

        assert_eq!(infer_type(&es), Ok(expected));
    }
}

#[cfg(test)]
mod scheme {
    use super::*;

    fn id_type() -> Type {
        Type::Arrow(
            Box::new(Type::Var("a".to_string())),
            Box::new(Type::Var("a".to_string())),
        )
    }

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
}
