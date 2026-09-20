package model

import (
	mathxI "github.com/sathishdv/tiny-llm-from-scratch/internal/mathx"
	token "github.com/sathishdv/tiny-llm-from-scratch/internal/token"
)

// TrainStep lets the model learn from one correct next-word example and
// returns how surprising that correct word was before the update. For example,
// given the example "the cat" -> "sat", Forward first estimates probabilities
// for every possible next word. If it gives "sat" a low probability, the loss
// is high. TrainStep then moves the relevant parameters a small learningRate
// step that raises "sat"'s score and lowers competing scores.
//
// The update travels backward from the output scores, through the tanh hidden
// layer, to the two input embeddings. This is backpropagation: each parameter
// is changed according to how it affected the error for this one example.
func (m *Model) TrainStep(ex token.Example, learningRate float64) float64 {
	// Forward pass: compute activations for the given context.
	activations := m.Forward(ex.Context)
	l := Loss(activations, ex.Next)
	v := len(activations.OutputProbabilities)

	// Gradient at the output. This is the famous softmax+cross-entropy
	// simplification: dLoss/dLogit_j = prob_j - (1 if j is the target else 0).
	// Intuition: "reduce every word's score by its probability, then give
	// the correct word +1."
	outputGradient := make([]float64, v)
	for j := 0; j < v; j++ {
		outputGradient[j] = activations.OutputProbabilities[j]
		if j == ex.Next {
			outputGradient[j] -= 1
		}
	}

	// Pass the output error back to each hidden activation while updating the
	// connections from that activation to every output word score.
	hiddenGradient := make([]float64, hiddenDim)
	for j := 0; j < hiddenDim; j++ {
		for k := 0; k < v; k++ {
			hiddenGradient[j] += outputGradient[k] * m.OutputWeights[j][k]
			m.OutputWeights[j][k] -= learningRate * outputGradient[k] * activations.HiddenActivations[j]
		}
	}

	for j := 0; j < v; j++ {
		m.OutputBiases[j] -= learningRate * outputGradient[j]
	}

	// Back through tanh: d/dz Tanh(z) = 1 - Tanh(z)².
	dPreActivation := make([]float64, hiddenDim)
	for j := 0; j < hiddenDim; j++ {
		dPreActivation[j] = hiddenGradient[j] * (1 - activations.HiddenActivations[j]*activations.HiddenActivations[j])
	}

	// Gradients for HiddenWeights, HiddenBiases, and the input vector.
	dX := make([]float64, inputDim)
	for j := 0; j < hiddenDim; j++ {
		for i := 0; i < inputDim; i++ {
			dX[i] += dPreActivation[j] * m.HiddenWeights[i][j]
			m.HiddenWeights[i][j] -= learningRate * dPreActivation[j] * activations.ContextEmbeddings[i]
		}
	}

	for j := 0; j < hiddenDim; j++ {
		m.HiddenBiases[j] -= learningRate * dPreActivation[j]
	}

	// Finally, back into the embedding table — the context words themselves
	// learn better representations from every prediction they take part in.
	for i := 0; i < embeddingDim; i++ {
		m.Embeddings[ex.Context[0]][i] -= learningRate * dX[i]
		m.Embeddings[ex.Context[1]][i] -= learningRate * dX[i+embeddingDim]
	}

	return l
}

// Loss measures how wrong one prediction is for its correct next-word ID.
// It is the negative natural logarithm of the correct word's probability.
// For example, if Forward gives "sat" probability 0.80 for "the cat", the
// loss is about 0.22; if it gives "sat" probability 0.10, the loss is about
// 2.30. Smaller loss is better because it means the model assigned more
// probability to the correct word.
//
// The tiny added value prevents taking ln(0) if a probability underflows to
// zero, which would otherwise make the loss infinite.
func Loss(activations *Activations, target int) float64 {
	// Compute the cross-entropy loss for the target word.
	return -mathxI.Ln(activations.OutputProbabilities[target] + 1e-12)
}

// AvgLoss evaluates the model on every example in dataset without changing its
// parameters, then returns their average loss. For a dataset containing
// "the cat" -> "sat" and "the dog" -> "ran", it measures how much the model
// is surprised by both correct next words and averages the two results. It is a
// progress measure: a lower average generally means the model predicts that
// dataset's next words more confidently and correctly.
func AvgLoss(dataset *token.Dataset, m *Model) float64 {
	var totalLoss float64
	for _, ex := range dataset.Examples {
		activations := m.Forward(ex.Context)
		totalLoss += Loss(activations, ex.Next)
	}
	return totalLoss / float64(len(dataset.Examples))
}
