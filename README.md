# tiny-lm-from-scratch

A complete, tiny language model written in Go with **no dependencies** — not even Go's
`math` package. Exponentials, logarithms, `tanh`, and the random number generator are all
implemented from first principles, so every number the model touches can be traced to code
you can read.

It learns to predict the next word from the previous two, trained on ten toy sentences.
197 parameters. Trains in under a second.

## What it does

```
$ make run

Vocabulary (13 tokens): [<S> <.> the cat sat on mat dog ran in park big small]
Trainable parameters: 197
Loss if the model were guessing randomly: 2.565

epoch | train loss | val loss  (val sentences were NEVER seen in training)
    1 |     2.2666 |   2.2813
  150 |     0.3123 |   0.6643
 1500 |     0.3045 |   1.0397

Next-word predictions (top 3):
Context: "the" "cat"          Context: "cat" "sat"
  1. "sat" (0.5165)             1. "on"  (0.9980)
  2. "ran" (0.4813)             2. "in"  (0.0008)

Generated sentences (temperature 0.8):
  "the small cat sat on the mat"
  "the dog ran in the park"
```

`"the cat"` splits roughly 50/50 between `sat` and `ran` because the training corpus uses
both equally after "cat" — the model learned the grammar rather than memorizing sentences.

## The pipeline

The code is organized as seven stages, each building on the last:

| Stage | Package | What it does |
|-------|---------|--------------|
| 1 | `internal/mathx` | `exp`, `ln`, `tanh`, and a seeded xorshift RNG — built from scratch |
| 2 | `internal/token` | Words → integer IDs; sentences → sliding-window training examples |
| 3 | `internal/model` (`model.go`) | The five parameter tables: embeddings, weights, biases |
| 4 | `internal/model` (`forward.go`) | Forward pass: two words → a probability for every vocabulary word |
| 5 | `internal/model` (`backward.go`) | Cross-entropy loss and backpropagation |
| 6 | `internal/sample` | Sampling with a temperature dial; generating whole sentences |
| 7 | `cmd/simple-lang-model` | Wires it together: train, evaluate, inspect, generate |

The core loop — embed → combine → predict → measure surprise → propagate blame backward →
nudge every parameter — is the same one that trains frontier models. The differences are
scale (197 parameters vs. billions, 2 words of context vs. thousands) and one architectural
idea this omits by design: attention.

## Layout

```
.
├── cmd/simple-lang-model/   # main entrypoint + training loop
├── internal/mathx/          # math primitives from scratch
├── internal/token/          # tokenizer, vocabulary, dataset
├── internal/model/          # parameters, forward pass, backprop
├── internal/sample/         # temperature sampling, generation
├── Makefile
└── go.mod
```

## Usage

```bash
make run      # train and generate
make build    # build to ./bin/simple-lang-model
make test     # run tests
make vet      # static analysis
```

## Notes

- **Reproducible by design.** The RNG is seeded with a fixed value (`42`), so every run
  produces identical output — which makes the training curve debuggable.
- **It overfits on purpose-built data.** Validation loss bottoms out at 0.4117 around epoch
  36 (the printed table samples every 150 epochs, so it doesn't show this) and then climbs
  while training loss stays flat. That gap is the textbook overfitting curve,
  visible in a model small enough to inspect by hand.
