package mathx

// Language-model example (a typical use, not a complete model in this file):
// after reading "the cat sat on the", a model might assign raw scores of 2 to
// "mat" and -1 to "moon". A score is only a comparison: it can be negative
// and it is not a probability.
//
// exp turns the scores into positive weights: exp(2) is about 7.39 and
// exp(-1) is about 0.37. A softmax operation divides each weight by their
// total, 7.76, yielding about 0.952 for "mat" and 0.048 for "moon". The larger
// original score therefore receives the larger probability. Exponentiation is
// necessary here because it preserves score ordering while producing values
// that can be normalized into probabilities.
//
// ln goes in the mathematical opposite direction: ln(p) is the value whose
// exponential is p. For example, ln(0.952) is about -0.049 because exp(-0.049)
// is about 0.952. ln does not restore the raw score 2, because softmax also
// divided by the total weight. In language-model training, log probabilities
// are useful because probabilities from many token choices multiply together,
// while their logarithms add together.
//
// tanh is used for a different job inside a neural-network layer, before its
// output becomes word scores. It changes any input to a value between -1 and 1:
// tanh(2) is about 0.964, and tanh(-2) is about -0.964. This bounded nonlinear
// transformation lets a layer represent more than a straight-line calculation
// while keeping its intermediate values from growing without limit.

// ============================================================================
// STAGE 1: MATH LIBRARY FROM SCRATCH
// Everything a neural net needs: exp, ln, tanh, and randomness.
// ============================================================================

// exp computes e^x using range reduction + Taylor series.
// Trick: e^x = (e^(x/2^n))^(2^n). We shrink x until it's small (where the
// Taylor series converges fast), then square the result back up n times.
/*
Explain: computes the exponential function:

exp(x) = e^x. where e≈2.71828. It is a foundational nonlinear operation in machine learning, especially for converting arbitrary scores into positive weights such as in softmax:

softmax(zi) = e^(zi) / ∑j e^(zj)
​
The implementation avoids using Go’s standard math.Exp so it can build the math primitive from scratch.

For negative input, it uses  e^−x = 1/e^x (line 13).

For positive input, it repeatedly halves
x
x until it is at most 0.5. Taylor series converge much more quickly for small values.

It approximates e^x using the first 12 terms of the Taylor expansion (lines 31-39):
e^x = 1 + x + x^2/2! + x^3/3! + ...

It then squares the result once for every halving, using:
e^x = (e^(x/2^n))^(2^n)

This is range reduction, and it is what lets a short Taylor series remain useful for larger inputs.

One important caveat: the overflow branch at line 18 returns literal 700, rather than a very large approximation of
e^700 is roughly 10^304, returning 700 is not mathematically accurate and creates a sharp discontinuity. In a neural-network setting, a more typical approach is to stabilize
the caller, for example by subtracting the largest logit before applying exp in softmax.

Example:
// exp computes e^x, turning a number into a rapidly growing positive number.
// For example, exp(2) is about 7.39 and exp(3) is about 20.09.
//
// In a language model, exp turns ordinary scores into positive weights. Given
// scores such as "world" = 2, "friend" = 1, and "banana" = -1, the weights
// become about 7.39, 2.72, and 0.37. A softmax step can then divide each weight
// by their total to produce probabilities.
//
// This implementation calculates exp(2) by first shrinking the input:
//
//	2 -> 1 -> 0.5
//
// It estimates e^0.5 with the Taylor series
// 1 + 0.5 + 0.5^2/2 + 0.5^3/6 + ..., which is about 1.6487. It then squares
// twice: 1.6487^2 is about 2.7183 (e^1), and 2.7183^2 is about 7.389 (e^2).
// In short: make the input small, calculate where it is easy, then repeatedly
// square to return to the original scale.

*/
func Exp(x float64) float64 {
	if x < 0 {
		return 1 / Exp(-x) // 13
	}

	if x > 700 {
		// would overflow, so we return a large number instead of infinity
		return 700
	}

	n := 0

	for x > 0.5 {
		x /= 2
		n++
	}

	// Taylor Series: e^x = 1 + x + x^2/2! + x^3/3! + ...
	result := 1.0
	term := 1.0

	for i := 1; i <= 12; i++ {
		term *= x / float64(i)
		result += term
	}

	// square the result back up n times to get e^(original x)
	for i := 0; i < n; i++ {
		result *= result
	}

	return result
}

// ln computes the natural logarithm of x: the number y for which e^y equals x.
// For example, ln(2) is about 0.693 because e^0.693 is about 2.
//
// Think of this as finding the right dial setting. We start at y = 0, where
// e^0 is 1. If that result is too small, we turn y up; if it is too large, we
// turn y down. Each correction uses the current error to make a better guess.
//
// For ln(2), the guesses move from 0 toward 0.693:
//
//	guess y = 0:     e^0     = 1, so turn the dial up
//	guess y = 1:     e^1     = 2.718, so turn the dial down
//	guess y = 0.736: e^0.736 = 2.088, so make a smaller adjustment
//
// The loop stops when an adjustment is so small that the answer is accurate
// enough for this float64 calculation. Natural logs exist only for x > 0.
func Ln(x float64) float64 {
	// Zero and negative numbers cannot be produced by e^y.
	if x <= 0 {
		panic("ln(x) is undefined for x <= 0")
	}

	// e^0 is exactly 1, so ln(1) is exactly 0.
	if x == 1 {
		return 0
	}

	// Start with the simple guess y = 0, then improve it below.
	y := 0.0

	for i := 0; i < 100; i++ {
		// See what the current dial setting produces.
		ey := Exp(y)
		// Measure the error and convert it into a correction for y.
		step := (ey - x) / ey
		y -= step

		// Stop once changing y would no longer meaningfully improve the answer.
		if step < 1e-13 && step > -1e-13 {
			break
		}
	}
	return y
}

// tanh squashes any number into the range (-1, 1). This is useful in a neural
// network because it prevents a layer's output from growing without limit while
// still preserving whether its input was positive or negative.
//
// Think of tanh as a volume knob with hard limits. A value near 0 stays near
// 0, a large positive value moves close to 1, and a large negative value moves
// close to -1. For example, tanh(1) is about 0.762: it is positive, but it
// cannot exceed 1. Tanh(-1) is about -0.762 for the same reason.
//
// For inputs that are not already near a limit, the function uses:
//
//	tanh(x) = (e^(2*x) - 1) / (e^(2*x) + 1)
//
// For tanh(1), exp(2) is about 7.389. Substituting it gives
// (7.389 - 1) / (7.389 + 1), or about 0.762. As x becomes much larger, e^(2*x)
// dominates both parts of the fraction, making the result approach 1.
func Tanh(x float64) float64 {
	// At these extremes the result is indistinguishable from the limit, and
	// avoiding exp(2*x) prevents unnecessarily large intermediate values.
	if x > 20 {
		return 1
	}
	if x < -20 {
		return -1
	}

	// Turn the input into a positive exponential value for the formula above.
	e := Exp(2 * x)
	// Compare that value with 1 to produce a result between -1 and 1.
	return (e - 1) / (e + 1)
}
