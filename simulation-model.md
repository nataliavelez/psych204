# Experiment 1: Recovering weights from simulated responses


**Step 1:** Helper functions and setup

The following model computes the utility of an option as the weighted sum of the features (weighted by three parameters reflecting the target's preferences for each feature). Choices are generated stochastically using the Luce choice axiom.

```
; Define target weights
(define test-weights (list '(target1 (-1 0 -1))
                           '(target2 (0 1 0))
                           '(target3 (-1 -1 0))))

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
(define softmax
  (lambda (weights options)
    (multinomial options (luce weights options))))
   
; Generate-responses
(define get-responses
  (lambda (weights)
    (map (lambda (options) (softmax weights options))
         movie-choices)))

(define model-target 'target1)

; Responses to be modeled
(define responses (get-responses (true-weights model-target)))

```

**Step 2:** The model

Here the model is stalling. Right now, weights are randomly initialized from a mixture of uniform distributions, responses are randomly generated from those weights, and the model is conditioned on the randomly generated sequence matchign the whole sequence of responses. I'm worried that this is taking *forever* to run simply because perfect matches between these two lists are extremely rare.

```
(define samples
  (mh-query
   1 10
      
   ; prior on weights (uniformly distributed--is this a reasonable assumption?)
   (define guess-weights
     (repeat 3 (lambda () (uniform -1 1))))
      
   ; query
   guess-weights
   
   ; condition
   (equal? (get-responses guess-weights) responses)
   ))

(hist (map first samples) "Posterior over the first weight (correct weight: -1)")
(hist (map second samples) "Posterior over the second weight (correct weight: 0)")
(hist (map third samples) "Posterior over the third weight (correct weight: -1)")

```