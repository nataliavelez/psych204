# Experiment 1: Recovering weights from simulated responses


**Step 1:** Setting up important variables and simulating responses

The following model computes the utility of an option as the weighted sum of the features (weighted by three parameters reflecting the target's preferences for each feature). Choices are generated stochastically using the Luce choice axiom.

```
; Define target weights
(define test-weights (list '(target1 (-1 0 -1))
                           '(target2 (.5 1 0))
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
```

The helper functions below calculate the utility of each option, given a set of weights and temperature, and randomly sample from each pair of movies using a softmax decision rule.
```
; Calculate utilities
(define calc-utility
  (lambda (weights options t)
    (map (lambda (o) (/ (sum (map * weights (list-elt movie-categories o))) t))
         options)))
   
; Define probabilities from utilities
(define luce
  (lambda (weights options t)
    (map (lambda (u)
           (/ u (sum (map exp (calc-utility weights options t)))))     
         (map exp (calc-utility weights options t)))))
   
; Softmax decision function
(define softmax
  (lambda (weights options t)
    (multinomial options (luce weights options t))))
   
; Generate-responses
(define get-responses
  (lambda (weights t)
    (map (lambda (options) (softmax weights options t))
         movie-choices)))

; Preferences to be recovered
(define model-target 'target2)

; Computing probability of responses, given weights
(define get-prob-responses
  (lambda (weights data t)
    (map (lambda (d o)
           (if (equal? d (first o))
               (first (luce weights o t))
               (second (luce weights o t))))
         data
         movie-choices)))
```

**Step 2:** The model

Finally, the model below attempts to recover the original weights by sampling weights according to the probability of the observed sequence of responses, given a set of weights. The mean sampled value for each weight is taken as the model's "best" estimate of the weight.

```
(define (samples data t)
  (mh-query
   500 10
      
   ; prior on weights (uniformly distributed--is this a reasonable assumption?)
   (define guess-weights
     (repeat 3 (lambda () (uniform -1 1))))
      
   ; query
   guess-weights
   
   ; condition
   (factor (sum (get-prob-responses guess-weights data t)))
   )))

; Iterating over temperature parameters
(define (test-temp t)
  (define s (samples (get-responses (true-weights model-target) t) t))
  (list (mean (map first s))
        (mean (map second s))
        (mean (map third s))))
(define temp-test (map test-temp '(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 2 3 4 5 6 7 8 9 10)))
```