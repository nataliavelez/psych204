# Experiment 2: Comparing human and model performance

*Step 1:* Setting up helper variables (choices, movie categories, responses)

```
;; ——————————————— SETTING UP IMPORTANT VARIABLES ———————————————
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
   (list 2 1) (list 2 3) (list 2 4) (list 2 5) (list 2 6) (list 2 7) (list 2 8) 
   (list 3 1) (list 3 2) (list 3 4) (list 3 5) (list 3 6) (list 3 7) (list 3 8)
   (list 4 1) (list 4 2) (list 4 3) (list 4 5) (list 4 6) (list 4 7) (list 4 8)
   (list 5 1) (list 5 2) (list 5 3) (list 5 4) (list 5 6) (list 5 7) (list 5 8)
   (list 6 1) (list 6 2) (list 6 3) (list 6 4) (list 6 5) (list 6 7) (list 6 8)
   (list 7 1) (list 7 2) (list 7 3) (list 7 4) (list 7 5) (list 7 6) (list 7 8)
   )
  )

(define movie-choices
  (append movie-choices movie-choices))

; Responses to be modeled (generated from weights: -1, 1, 0)
(define responses
  '(1, 1, 1, 5, 6, 1, 1, 2, 2, 2, 2, 6, 2, 2, 1, 2, 4, 3, 6, 3, 3, 4, 4, 4, 4,
       6, 4, 4, 1, 2, 5, 4, 6, 5, 8, 1, 6, 6, 6, 6, 6, 6, 7, 2, 7, 4, 5, 6, 8,
       1, 3, 3, 4, 5, 6, 8))
```

*Step 2:* Helper functions, which (1) calculate the utlity of each option, (2) calculate the probability of selecting each option, given its utility, and (3) generate responses using a softmax decision rule.

```
;; ——————————————— SIMULATING CHOICES ———————————————
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

**Step 3:** Define two models, one of which (samples), returns sampled weights from observing n trials, and another (model-accuracy), which takes the mean of the estimated weights from samples and returns the predicted response for the (n+1)th trial.

```
;; ——————————————— THE MODEL ITSELF ———————————————
; Cognitive model
(define (samples data t)
  (mh-query
   100 10
      
   ; prior on weights (uniformly distributed--is this a reasonable assumption?)
   (define guess-weights
     (repeat 3 (lambda () (uniform -1 1))))
      
   ; query
   guess-weights
   
   ; condition
   (factor (sum (get-prob-responses guess-weights data t)))
   ))

; Iterating model over number of responses
(define num-samples '(1 7 15 23 31 39 47 55))

; Sample weights from subset of responses


; Get estimate of the weights from sample
(define (model-accuracy n t)
  (mh-query
   50 10
   
   (define s (samples (take responses n) t))
   
   (define w (list (mean (map first s))
      (mean (map second s))
      (mean (map third s))))
   
   (equal? (softmax w (list-elt movie-choices (+ n 1)) t) (list-elt responses (+ n 1)))
   
   true
   ))

(model-accuracy 55 0.1)
```