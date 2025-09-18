### Example:
(define (problem assembly-ex1)
(:domain assembly)
(:objects b1 b2 b3 - block
          orange yellow green - color)
(:init
(is-cube b1) (color b1 orange)
(is-cube b2) (color b2 yellow)
(is-cube b3) (color b3 green)

(on-table b1)
(on-table b2)
(on-table b3)

(clear b1)
(clear b2)
(clear b3)

(arm-empty)
)
(:goal (and
(on b1 b3)
(on b2 b1))
)

### Example:
(define (problem assembly-ex2)
(:domain assembly)
(:objects b1 b2 b3 - block
          yellow green purple - color)
(:init
(is-cube b1) (color b1 yellow)
(is-cube b2) (color b2 purple)
(is-cube b3) (color b3 green)

(on-table b1)
(on-table b2)
(on-table b3)

(clear b1)
(clear b2)
(clear b3)

(arm-empty)
)
(:goal (and
(left b1 b2)
(above b3 b2 b1))
)

### Example:
(define (problem assembly-ex3)
(:domain assembly)
(:objects b1 b2 - block
          red blue - color)
(:init
(is-cube b1) (color b1 blue)
(is-cube b2) (color b2 red)

(on-table b1)
(on-table b2)

(clear b1)
(clear b2)

(arm-empty)
)
(:goal
(on b2 b1)
)

### Example:
(define (problem assembly-ex4)
(:domain assembly)
(:objects b1 b2 b3 b4 - block
          red orange green purple - color)
(:init
(is-cube b1) (color b1 green)
(is-cube b2) (color b2 purple)
(is-cube b3) (color b3 orange)
(is-cube b4) (color b4 red)

(on-table b1)
(on-table b2)
(on-table b3)
(on-table b4)

(clear b1)
(clear b2)
(clear b3)
(clear b4)

(arm-empty)
)
(:goal (and
(right b1 b3)
(on b4 b1)
(on b2 b3))
)
