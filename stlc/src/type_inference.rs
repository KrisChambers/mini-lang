use std::collections::HashMap;

use crate::parser::Expr;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Int,
    Var(String),
    Bool,
    Arrow(Box<Type>, Box<Type>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Substitution(Vec<(Type, String)>);

#[derive(Debug, Clone)]
pub struct TypeEnv {
    map: HashMap<String, Type>,
    var_gen: FreshVarGen,
}

/// Generates fresh Type::Var
#[derive(Debug, Clone)]
struct FreshVarGen(usize);

impl FreshVarGen {
    pub fn next(&mut self) -> Type {
        let next_idx = self.0;
        self.0 += 1;

        Type::Var(format!("v{}", next_idx))
    }
}

impl TypeEnv {
    pub fn new() -> Self {
        TypeEnv {
            map: HashMap::new(),
            var_gen: FreshVarGen(0),
        }
    }

    fn get(&self, name: &str) -> Option<Type> {
        self.map.get(name).cloned()
    }

    fn append(&self, name: &str, t: &Type) -> TypeEnv {
        let mut inner_map = self.map.clone();
        inner_map.insert(name.to_string(), t.clone());

        Self {
            map: inner_map,
            var_gen: self.var_gen.clone(),
        }
    }

    fn extend_fresh(&mut self, name: &str) -> TypeEnv {
        let fresh = self.var_gen.next();

        self.append(name, &fresh)
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

    fn get_fresh_var(&mut self) -> Type {
        self.var_gen.next()
    }
}

pub fn w(e: Expr, mut te: TypeEnv) -> Result<(Substitution, Type), String> {
    use Expr::*;
    match e {
        Var(name) => match te.get(&name) {
            Some(t) => Ok((Substitution::id(), t)),
            None => Err("Invalid".to_string()),
        },
        App(e1, e2) => {
            let (s1, t1) = w(*e1, te.clone())?;
            let (s2, t2) = w(*e2, te.apply_sub(&s1))?;
            let u = te.get_fresh_var();
            let t = Type::Arrow(Box::new(t2), Box::new(u.clone()));
            if let Some(s3) = unify(s2.apply(t1), t.clone()) {
                Ok((s3.compose(&s2).compose(&s1), s3.apply(u.clone())))
            } else {
                Err("Invalid".to_string())
            }
        }
        Lambda(var_name, _type_ann, expr) => {
            let new_te = te.extend_fresh(&var_name);
            let u = new_te.get(&var_name).unwrap();

            let (s1, t1) = w(*expr, new_te)?;
            let t = Type::Arrow(Box::new(s1.apply(u)), Box::new(t1));

            Ok((s1, t))
        }
        If(cond, t_e, f_e) => {
            let (_, tc) = w(*cond, te.clone())?;

            if tc != Type::Bool {
                return Err("Invalid condition".to_string());
            };

            let (s1, t1) = w(*t_e, te.clone())?;
            let (s2, t2) = w(*f_e, te.apply_sub(&s1))?;
            match unify(s2.apply(t1), t2.clone()) {
                Some(s) => Ok((s.compose(&s2).compose(&s1), t2.clone())),
                None => Err("Invalid".to_string()),
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
            // This needs to be thought out a bit more.

            // First lets get the type of e1 and e2
            let (_ , t1)= w(*e1, te.clone())?;
            let (_, t2) = w(*e2, te.clone())?;

            // Once we have this type we need both of these to unify with a type assocated with the
            // operation.
            use crate::parser::Op::*;
            let t_op = match op {
                Add | Subtract => Type::Int,
                And | Or => Type::Bool,
            };
            let s11 = unify(t1.clone(), t_op.clone()).ok_or("Invalid")?;
            let s22 = unify(s11.apply(t2.clone()), t_op.clone()).ok_or("Invalid")?;

            // If we get this far then we have substitutions that can get t1 and t2 to an int.
            // Which is what we need? Maybe there is an edge case where this isn't possible without
            // First having a substitution to make the type of e1 = type of e2?

            Ok((s22, t_op))
        }
        Let(var_name, e1, e2) => {
            let sub_te = te.extend_fresh(&var_name);
            let a = sub_te.get(&var_name).unwrap();
            let (s1, t1) = w(*e1, sub_te)?;
            let s2 = if let Some(s) = unify(s1.apply(a), t1.clone()) {
                s
            } else {
                return Err("Invalid".to_string());
            };

            let sub_te = te.apply_sub(&s2.compose(&s1)).append(&var_name, &t1);
            let (s3, t2) = w(*e2, sub_te)?;

            Ok((s3.compose(&s2.compose(&s1)), t2.clone()))
        },
    }
}

fn apply_sub(t: Type, sub: &(Type, String)) -> Type {
    let (new_t, var_name) = &sub;
    match t {
        // Primitive types don't substitute anything
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
        (typ, Var(name)) => Some((typ, name).into()),
        (Var(name), typ) => Some((typ, name).into()),
        (Arrow(t11, t12), Arrow(t21, t22)) => {
            let s1 = unify(*t11, *t21)?;
            let s2 = unify(*t12, *t22)?;

            Some(s2.compose(&s1))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
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
}
