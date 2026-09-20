package model

import (
	mathxI "github.com/sathishdv/tiny-llm-from-scratch/internal/mathx"
)

// ============================================================================
// STAGE 4: FORWARD PASS — from context words to a probability distribution
// ============================================================================

// Activations saves the intermediate numbers from one Forward call. TrainStep
// uses them to determine which model parameters helped or hurt that prediction.
// They are temporary working notes, not learned model parameters.
type Activations struct {
	// ContextEmbeddings is the concatenation of the two word embeddings for the context words.
	ContextEmbeddings []float64 // [inputDim]float64

	// HiddenPreActivations is the linear combination of inputs and weights before applying the activation function.
	HiddenPreActivations []float64 // [hiddenDim]float64

	// HiddenActivations is the output of the hidden layer after applying the activation function.
	HiddenActivations []float64 // [hiddenDim]float64

	// OutputPreActivations is the linear combination of hidden activations and output weights before applying softmax.
	OutputPreActivations []float64 // [vocabSize]float64

	// OutputProbabilities is the final probability distribution over the vocabulary after applying softmax. softmax output: P(next word = w) for every w in the vocabulary.
	OutputProbabilities []float64 // [vocabSize]float64
}

// Forward makes one next-word probability distribution from two context word
// IDs without changing the model. For example, for the context "the cat", it
// looks up the embeddings for "the" and "cat", combines them through the
// hidden layer, and assigns a probability to every vocabulary word. If "sat"
// receives 0.60, Forward is saying that this model currently gives "sat" a
// 60 percent probability as the next word for that context.
//
// The returned Activations includes both the final probabilities and the
// intermediate values needed later by TrainStep. The probabilities sum to 1.
func (m *Model) Forward(context [2]int) *Activations {
	activations := &Activations{
		ContextEmbeddings:    make([]float64, inputDim),
		HiddenPreActivations: make([]float64, hiddenDim),
		HiddenActivations:    make([]float64, hiddenDim),
		OutputPreActivations: make([]float64, len(m.OutputBiases)),
		OutputProbabilities:  make([]float64, len(m.OutputBiases)),
	}

	// Step 1: Look up embeddings for the two context words and concatenate them.
	copy(activations.ContextEmbeddings[0:embeddingDim], m.Embeddings[context[0]])
	copy(activations.ContextEmbeddings[embeddingDim:], m.Embeddings[context[1]])

	// Step 2: Combine the input numbers into each hidden value, then apply tanh.
	for j := 0; j < hiddenDim; j++ {
		sum := m.HiddenBiases[j]
		for i := 0; i < inputDim; i++ {
			sum += activations.ContextEmbeddings[i] * m.HiddenWeights[i][j]
		}
		activations.HiddenPreActivations[j] = sum
		activations.HiddenActivations[j] = mathxI.Tanh(sum)
	}

	// Step 3: Create one unnormalized score (logit) per vocabulary word.
	for k := 0; k < len(m.OutputBiases); k++ {
		sum := m.OutputBiases[k]
		for j := 0; j < hiddenDim; j++ {
			sum += activations.HiddenActivations[j] * m.OutputWeights[j][k]
		}
		activations.OutputPreActivations[k] = sum
	}

	// Step 4: Convert scores to probabilities. Subtracting the largest score
	// changes no probability, but keeps the exponentials at a safe size.
	maxLogit := activations.OutputPreActivations[0]
	for _, logit := range activations.OutputPreActivations {
		if logit > maxLogit {
			maxLogit = logit
		}
	}

	var sumExp float64
	for k, logit := range activations.OutputPreActivations {
		expLogit := mathxI.Exp(logit - maxLogit)
		activations.OutputProbabilities[k] = expLogit
		sumExp += expLogit
	}
	for k := range activations.OutputProbabilities {
		activations.OutputProbabilities[k] /= sumExp
	}

	return activations
}
