# mini-lang

A basic language built in rust with a minimal type checker.


# Language Definition

The language has the following components
- Let bindings
- functions
- integers
- booleans
- function application
- conditionals

```
let x = 42;
let b = true;
let add = fn(a,b) { a + b };
let result = add(5, 10);
let n = if b then x else 0;
```

