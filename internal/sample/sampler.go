package sample

import (
	"strings"

	mathxI "github.com/sathishdv/tiny-lm-from-scratch/internal/mathx"
	"github.com/sathishdv/tiny-lm-from-scratch/internal/model"
	"github.com/sathishdv/tiny-lm-from-scratch/internal/token"
)

// ============================================================================
// STAGE 6: GENERATION — the model writes sentences
// Same loop every chat LLM runs: predict, sample, append, repeat.
// ============================================================================

func Sample(probs []float64, temperature float64, rng *mathxI.RNG) int {
	// Step 1: Recover log-probabilities and divide them by the temperature.
	// Temperature reshapes the distribution: <1 = safer/greedier, >1 = wilder.
	// Scaling must happen in log space; scaling probabilities directly is not
	// the same curve. The 1e-12 guards Ln against a probability of exactly 0.
	scaled := make([]float64, len(probs))
	maxScaled := 0.0
	for i, p := range probs {
		scaled[i] = mathxI.Ln(p+1e-12) / temperature
		if i == 0 || scaled[i] > maxScaled {
			maxScaled = scaled[i]
		}
	}

	// Step 2: Exponentiate. Subtracting the largest value changes no
	// probability but keeps every exponent <= 0, so Exp cannot overflow.
	adjustedProbs := make([]float64, len(probs))
	var sum float64
	for i, s := range scaled {
		adjustedProbs[i] = mathxI.Exp(s - maxScaled)
		sum += adjustedProbs[i]
	}

	// Step 3: Walk the weighted number line until we pass a random threshold.
	r := rng.Float() * sum
	cumulative, best := 0.0, 0
	for i, p := range adjustedProbs {
		cumulative += p
		if cumulative >= r {
			return i
		}
		if p > adjustedProbs[best] {
			best = i
		}
	}

	// Fallback for floating-point rounding: the most likely word, never a
	// fixed index. A fixed index turns a NaN distribution into an infinite
	// generation loop, because NaN comparisons above are always false.
	return best
}

func Generate(model *model.Model, vocab *token.Vocab, temperature float64, rng *mathxI.RNG) string {
	ctx := [2]int{vocab.ToID[token.TokenStart], vocab.ToID[token.TokenStart]}
	var words []string

	for {
		activations := model.Forward(ctx)
		nextWordID := Sample(activations.OutputProbabilities, temperature, rng)
		if nextWordID == vocab.ToID[token.TokenEnd] {
			break
		}
		words = append(words, vocab.ToWord[nextWordID])
		ctx[0], ctx[1] = ctx[1], nextWordID // Slide the context window.
	}

	return strings.Join(words, " ")
}
