## Lambda with type annotations (STLC style)
\(x : Int) -> x + 1
\(f : Int -> Int) -> \(x : Int) -> f x

## Application
(\(x : Int) -> x + 1) 5

## Basic arithmetic and boolean operations
2 + 3
true && false
if true then 1 else 2

## Sugar for high level let .. in

```
add = \h -> \m -> h + m
main = add 5

let add2 = \x -> x + 2 in
let is2 = \x -> x == 2 in
let (main =
    let hour = 2 in
    let min = 3 in hour + 2)
    in
    is2 main

----

let
    add2 = \x -> x + 2,
    is2 = \x -> x == 2,
    main =
        let
            hour = 2,
            min = 3
        in
            hour + 2 + 3
    in
        main


doStuff =
    let
        hour = 2,
        min = 3
    in
        add hour min

let (doStuff =
    let hour = 2 in
    let min = 3 in hour + min)
    in
    doStuff

dostuff =
    let hour = 2 in
    let min = 3 in
    add hour min




```
