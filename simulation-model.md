# Experiment 1: Recovering weights from simulated responses


**Step 1:** Helper functions and setup

The following model computes the utility of an option as the weighted sum of the features (weighted by three parameters reflecting the target's preferences for each feature). Choices are generated stochastically using the Luce choice axiom.

NB: Right now, the temperature parameter is set to 1. One reason why the model might be performing poorly is that the choice is too stochastic--it's not being 'greedy' enough. Next step is to test the model with different settings of the temperature parameter.

```
;; ——————————————— SETTING UP IMPORTANT VARIABLES ———————————————
; Define target weights
(define test-weights (list '(target1 (-1 0 -1))
                           '(target2 (.7 1 -.2))
                           '(target3 (-1 -1 0))
                           '(target4 (.5 .5 .5))
                           '(target5 (-1 0 1))
                           '(target6 (-.8 .5 0))))

; Helper function to retrieve true weights
(define true-weights
  (lambda (target)
    (second (assoc target test-weights))))

; Define options
(define movie-categories
  (list
   (list 1 1 1)
   (list 1 1 -1)
   (list 1 -1 1)
   (list -1 1 1)
   (list 1 -1 -1)
   (list -1 1 -1)
   (list -1 -1 1)
   (list -1 -1 -1))
  )

(define movie-choices
  (list
   (list 1 2) (list 1 3) (list 1 4) (list 1 5) (list 1 6) (list 1 7) (list 1 8)
   (list 2 3) (list 2 4) (list 2 5) (list 2 6) (list 2 7) (list 2 8)
   (list 3 4) (list 3 5) (list 3 6) (list 3 7) (list 3 8)
   (list 4 5) (list 4 6) (list 4 7) (list 4 8)
   (list 5 6) (list 5 7) (list 5 8)
   (list 6 7) (list 6 8)
   (list 7 8)
   )
  )

(define movie-choices
  (append movie-choices movie-choices movie-choices movie-choices))

movie-choices

;; ——————————————— SIMULATING CHOICES ———————————————
; Calculate utilities
(define calc-utility
  (lambda (weights options)
    (map (lambda (o) (sum (map * weights (list-elt movie-categories o))))
         options)))
   
; Define probabilities from utilities
(define luce
  (lambda (weights options)
    (map (lambda (u)
           (/ u (sum (map exp (calc-utility weights options)))))     
         (map exp (calc-utility weights options)))))
   
; Softmax decision function
; TODO: CHANGE INVERSE TEMPERATURE
(define softmax
  (lambda (weights options)
    (multinomial options (luce weights options))))
   
; Generate-responses
(define get-responses
  (lambda (weights)
    (map (lambda (options) (softmax weights options))
         movie-choices)))

; Preferences to be recovered
(define model-target 'target2)

; Responses to model
(define responses (get-responses (true-weights model-target)))

; Computing probability of responses, given weights
(define get-prob-responses
  (lambda (weights data)
    (map (lambda (d o)
           (if (equal? d (first o))
               (first (luce weights o))
               (second (luce weights o))))
         data
         movie-choices)))
```

**Step 2:** The model

```
;; ——————————————— THE MODEL ITSELF ———————————————

(define (samples data)
  (mh-query
   1000 10
      
   ; prior on weights (uniformly distributed--is this a reasonable assumption?)
   (define guess-weights
     (repeat 3 (lambda () (uniform -1 1))))
      
   ; query
   guess-weights
   
   ; condition
   (factor (sum (get-prob-responses guess-weights data)))
   ))


;; ——————————————— PLOT RESULTS ———————————————
; Final estimates of the weights
(display "Correct weights: " (true-weights model-target))
(hist (map first (samples responses)))
(hist (map second (samples responses)))
(hist (map third (samples responses)))

```