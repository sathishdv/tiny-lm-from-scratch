package main

import (
	"fmt"
	"os"

	mathxI "github.com/sathishdv/tiny-lm-from-scratch/internal/mathx"
	"github.com/sathishdv/tiny-lm-from-scratch/internal/model"
	"github.com/sathishdv/tiny-lm-from-scratch/internal/sample"
	"github.com/sathishdv/tiny-lm-from-scratch/internal/token"
)

func main() {
	// if err := app.Run(os.Args[1:]); err != nil {
	// 	fmt.Fprintln(os.Stderr, "error:", err)
	// 	os.Exit(1)
	// }

	if err := Run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}

	// do not exit
	fmt.Scanln()
}

// ============================================================================
// STAGE 7: PUT IT ALL TOGETHER — train, validate, inspect, generate
// ============================================================================

func Run(args []string) error {
	rng := &mathxI.RNG{State: 42}
	vocab := token.NewVocab()
	vocab.Build(token.TrainSentences, token.TestSentences)
	train := token.NewDataset(token.TrainSentences, vocab)
	test := token.NewDataset(token.TestSentences, vocab)
	mdl := model.New(len(vocab.ToWord), rng)

	fmt.Printf("Vocabulary (%d tokens): %v\n", len(vocab.ToWord), vocab.ToWord)
	fmt.Printf("Trainable parameters: %d\n", mdl.ParamCount())
	fmt.Printf("Training examples: %d   Validation examples: %d\n\n", len(train.Examples), len(test.Examples))
	fmt.Printf("Loss if the model were guessing randomly: %.3f\n\n", mathxI.Ln(float64(len(vocab.ToWord))))

	//--- Training loop: run through the training examples multiple times, updating the model each time ---
	epochs, lr := 1500, 0.05
	fmt.Println("epoch | train loss | val loss  (val sentences were NEVER seen in training)")

	for epoch := 1; epoch <= epochs; epoch++ {
		// Shuffle (Fisher-Yates) the training examples each epoch to avoid biasing the model to the order of the data. Model doesn't learn sentence order.
		for i := len(train.Examples) - 1; i > 0; i-- {
			j := int(rng.Next() % uint64(i+1))
			train.Examples[i], train.Examples[j] = train.Examples[j], train.Examples[i]
		}
		for _, ex := range train.Examples {
			mdl.TrainStep(ex, lr)
		}

		if epoch%150 == 0 || epoch == 1 {
			trainLoss := model.AvgLoss(train, mdl)
			valLoss := model.AvgLoss(test, mdl)
			fmt.Printf("%5d | %10.4f | %8.4f\n", epoch, trainLoss, valLoss)
		}
	}

	// --- Inspect: what does the model believe follows a given context? ---
	fmt.Println("\nNext-word predictions (top 3):")

	inputSample := [][2]string{{"the", "cat"}, {"cat", "sat"}, {"sat", "on"}, {"in", "the"}}

	for _, ctx := range inputSample {
		ctxIDs := [2]int{vocab.ToID[ctx[0]], vocab.ToID[ctx[1]]}
		activations := mdl.Forward(ctxIDs)
		top3 := 3

		fmt.Printf("Context: %q %q\n", ctx[0], ctx[1])
		for i := 0; i < top3; i++ {
			best, bestP := 0, -1.0
			for j, p := range activations.OutputProbabilities {
				if p > bestP {
					best, bestP = j, p
				}
			}
			fmt.Printf("  %d. %q (%.4f)\n", i+1, vocab.ToWord[best], bestP)
			activations.OutputProbabilities[best] = -1 // Exclude it from the next iteration.
		}

		fmt.Println()
	}

	// --- Generate ---
	fmt.Println("\nGenerated sentences (temperature 0.8):")
	for i := 0; i < 6; i++ {
		sentence := sample.Generate(mdl, vocab, 0.8, rng)
		fmt.Printf("  %q\n", sentence)
	}

	fmt.Println("\nGenerated sentences (temperature 2.0 — watch quality degrade):")
	for i := 0; i < 3; i++ {
		sentence := sample.Generate(mdl, vocab, 2.0, rng)
		fmt.Printf("  %q\n", sentence)
	}

	return nil
}
