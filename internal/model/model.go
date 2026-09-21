package model

import (
	"github.com/sathishdv/tiny-lm-from-scratch/internal/mathx"
)

// ============================================================================
// STAGE 3: THE MODEL — every parameter, explicitly
// ============================================================================

const (
	// The model has a vocabulary of 12 words, each represented by a 4-dimensional embedding vector. The model sees two words of context and predicts the next word.
	embeddingDim = 4                          // Each word is represented by a 4-dimensional vector.
	contextSize  = 2                          // The model sees two words of context.
	inputDim     = embeddingDim * contextSize // The input to the model is the concatenation of two word embeddings.
	hiddenDim    = 6                          // The hidden layer has 6 neurons.
)

// Model holds the numbers that a later training step can adjust to learn from
// text. For example, given the two-word context "the cat", a complete forward
// pass would look up one 4-number embedding for "the" and another for "cat",
// join them into 8 numbers, pass them through 6 hidden values, and produce one
// score for every vocabulary word. The largest score could then represent the
// model's strongest next-word guess, such as "sat".
//
// This type only stores those learnable numbers. It does not itself perform
// that lookup, forward pass, or choose a next word.
type Model struct {
	// Embeddings maps each word ID to its corresponding embedding vector.
	Embeddings [][]float64 //[][embeddingDim]float64

	// HiddenWeights is the weight matrix connecting the input layer to the hidden layer.
	HiddenWeights [][]float64 // [inputDim][hiddenDim]float64

	// HiddenBiases is the bias vector for the hidden layer.
	HiddenBiases []float64 // [hiddenDim]float64

	// OutputWeights is the weight matrix connecting the hidden layer to the output layer.
	OutputWeights [][]float64 // [hiddenDim][]float64

	// OutputBiases is the bias vector for the output layer.
	OutputBiases []float64
}

// New creates a Model with the parameter shapes needed for two-word contexts
// and vocabSize possible next words. It starts embeddings and both weight
// matrices with small pseudo-random values from rng, scaled by 0.3; their
// values are centered approximately around zero. It starts all biases at zero.
//
// For example, when vocabSize is 12 and the context is "the cat", New creates:
//
//   - 12 embeddings with 4 numbers each, one row for each vocabulary ID;
//   - 8 by 6 hidden weights, connecting the two joined 4-number embeddings to
//     6 hidden values; and
//   - 6 by 12 output weights, connecting those hidden values to 12 word scores.
//
// The small random starting values are important because they give different
// connections different starting points. During training, those differences
// allow connections to receive different updates and learn different patterns.
// Zero biases are a neutral starting point: before training, they do not favor
// any hidden value or vocabulary word on their own.
func New(vocabSize int, rng *mathx.RNG) *Model {
	return &Model{
		Embeddings:    randMatrix(vocabSize, embeddingDim, 0.3, rng),
		HiddenWeights: randMatrix(inputDim, hiddenDim, 0.3, rng),
		HiddenBiases:  make([]float64, hiddenDim),
		OutputWeights: randMatrix(hiddenDim, vocabSize, 0.3, rng),
		OutputBiases:  make([]float64, vocabSize),
	}
}

// randMatrix creates a table with rows by cols numbers and fills every position with
// a small pseudo-random value. Each value is drawn from rng.Norm and multiplied
// by scale, so scale controls how large the starting values usually are.
//
// For example, randMatrix(2, 3, 0.3, rng) makes a two-row, three-column table such as:
//
//	[ 0.12, -0.08,  0.31 ]
//	[-0.17,  0.04, -0.09 ]
//
// The actual values will depend on rng's current state. This helper supplies
// the initial embeddings and weights; training changes these values later.
func randMatrix(rows, cols int, scale float64, rng *mathx.RNG) [][]float64 {
	result := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		result[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			result[i][j] = rng.Norm() * scale
		}
	}
	return result
}

// ParamCount returns the number of adjustable numbers, called parameters, in
// the model. Think of parameters as tiny knobs that training turns so a context
// such as "the cat" can eventually give a higher score to "sat" than to other
// possible next words. This method only counts the knobs; it does not change
// them or make a prediction.
//
// For a model with 12 vocabulary words, the count is:
//
//	12 embeddings * 4 numbers = 48
//	8 input values * 6 hidden values = 48 hidden weights
//	6 hidden biases = 6
//	6 hidden values * 12 word scores = 72 output weights
//	12 output biases = 12
//	total = 186 parameters
//
// The total is a simple measure of model size. More parameters provide more
// numbers that training can adjust, but also require more memory and work to
// train. This small model has 186 parameters; large language models use the
// same idea with far more parameters.
func (m *Model) ParamCount() int {
	n := len(m.Embeddings)*embeddingDim + inputDim*hiddenDim + hiddenDim
	n += hiddenDim*len(m.OutputBiases) + len(m.OutputBiases)
	return n
}
