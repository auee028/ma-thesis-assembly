(define (problem assembly-01)
  (:domain assembly)
  (:objects 
    b1 b2 b3 b4 b5 b6 - block
    red orange yellow green blue purple - color)
  (:init
    (is-cube b1) (color b1 red)
    (is-cube b2) (color b2 orange)
    (is-cube b3) (color b3 yellow)
    (is-cube b4) (color b4 green)
    (is-cube b5) (color b5 blue)
    (is-cube b6) (color b6 purple)

    (on-table b1)
    (on-table b2)
    (on-table b3)
    (on-table b4)
    (on-table b5)
    (on-table b6)

    (clear b1)
    (clear b2)
    (clear b3)
    (clear b4)
    (clear b5)
    (clear b6)

    (arm-empty)
  )

  (:goal (and
    (left b1 b5)
    (left b6 b1)
    (above b2 b6 b1)
    (above b3 b1 b5)
    (above b4 b2 b3))
  )
)
