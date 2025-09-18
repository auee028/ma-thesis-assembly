(define (domain assembly)
  (:requirements :strips :typing)
  
  (:types
    block color
  )

  (:predicates 
    (is-cube ?b - block)
    (is-triangle ?b - block)
    (color ?b - block ?c - color)
    (on-table ?b - block)
    (clear ?b - block)
    (arm-empty)
    (holding ?b - block)

    (left ?x - block ?y - block)
    (right ?x - block ?y - block)
    (front ?x - block ?y - block)
    (behind ?x - block ?y - block)
    (on ?x - block ?y - block)    ; direct support
    (above ?x - block ?y - block ?z - block)    ; dual support
  )

  (:action pickup
    :parameters (?ob - block)
    :precondition (and (clear ?ob) (arm-empty))
    :effect (and (holding ?ob) 
                 (not (clear ?ob)) 
                 (not (on-table ?ob))
                 (not (arm-empty)))
  )

  (:action putdown-left
    :parameters (?ob - block ?supp - block)
    :precondition (holding ?ob)
    :effect (and (clear ?ob)
                 (arm-empty)
                 (left ?ob ?supp)
                 (not (holding ?ob)))
  )

  (:action putdown-right
    :parameters (?ob - block ?supp - block)
    :precondition (holding ?ob)
    :effect (and (clear ?ob)
                 (arm-empty)
                 (right ?ob ?supp)
                 (not (holding ?ob)))
  )

  (:action putdown-front
    :parameters (?ob - block ?supp - block)
    :precondition (holding ?ob)
    :effect (and (clear ?ob)
                 (arm-empty)
                 (front ?ob ?supp)
                 (not (holding ?ob)))
  )

  (:action putdown-behind
    :parameters (?ob - block ?supp - block)
    :precondition (holding ?ob)
    :effect (and (clear ?ob)
                 (arm-empty)
                 (behind ?ob ?supp)
                 (not (holding ?ob)))
  )

  (:action stack-on-another
    :parameters (?ob - block ?supp - block)
    :precondition (and (holding ?ob) (clear ?supp))
    :effect (and (arm-empty)
                 (clear ?ob)
                 (on ?ob ?supp)
                 (not (clear ?supp))
                 (not (holding ?ob)))
  )

  (:action stack-above-two
    :parameters (?ob - block ?supp1 - block ?supp2 - block)
    :precondition (and (holding ?ob) 
                      (clear ?supp1) 
                      (clear ?supp2))
    :effect (and (arm-empty)
                 (clear ?ob)
                 (above ?ob ?supp1 ?supp2)
                 (not (clear ?supp1))
                 (not (clear ?supp2))
                 (not (holding ?ob)))
  )
)
