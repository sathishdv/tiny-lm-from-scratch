package mathx

// Randomness helps a neural network begin with different small values instead
// of making every part of the network identical. For example, when a model is
// learning that "the cat sat on the" is often followed by "mat", its internal
// connections need different starting values so training can adjust them in
// different directions. If all connections started at exactly zero, identical
// connections would keep receiving identical updates and could not learn
// different jobs.
//
// RNG supplies those starting values. It is pseudo-random: its numbers look
// unpredictable, but a particular starting state always produces the same
// sequence. That repeatability is useful when debugging because the same
// training run can be reproduced.

// RNG is a xorshift64 pseudo-random number generator. state holds the current
// position in its deterministic sequence; callers choose the initial state.
type RNG struct {
	State uint64
}

// Next advances the generator by one step and returns its Next unsigned
// 64-bit number. It mixes the bits in state by shifting them left and right,
// then combining them with XOR. Think of it as shuffling a fixed row of 64
// on/off switches: a tiny change in the current pattern produces a very
// different-looking Next pattern.
//
// This is the low-level number source used by Float. It does not provide true
// physical randomness; the same initial state produces the same Next values.
func (r *RNG) Next() uint64 {
	r.State ^= r.State << 13
	r.State ^= r.State >> 7
	r.State ^= r.State << 17

	return r.State
}

// Float returns a pseudo-random decimal in the range [0, 1): it can return 0
// or a value such as 0.42, but it never returns 1. Values across that range are
// intended to be equally likely. It takes 53 bits from next and scales them
// into a float64, whose fraction has 53 bits of precision.
//
// For example, Float can be used to make a weighted choice after the sentence
// "the cat sat on the": if "mat" has probability 0.95, a draw of 0.42 selects
// it because 0.42 falls within its first 0.95 share of the [0, 1) range.
func (r *RNG) Float() float64 {
	return float64(r.Next()>>11) / float64(1<<53)
}

// Norm returns a pseudo-random value shaped approximately like a standard bell
// curve: most results are near 0, while values far from 0 are less common. Its
// average is 0 and its variance is 1. This shape is useful for starting neural
// network weights with small positive and negative differences instead of a
// uniform pattern.
//
// For example, a connection involved in recognizing that "cat" can be related
// to "mat" might start with a value near 0, such as 0.18 or -0.31. Training,
// rather than the random starting value, determines whether that connection
// later becomes important.
//
// The function calls Float 12 times, adds the results, and subtracts 6. Since
// 12 uniform values average to 6, subtracting 6 centers the result around 0.
// Adding many values makes middle-sized totals more common than extreme totals,
// which approximates a bell curve without requiring sqrt or log.
func (r *RNG) Norm() float64 {
	sum := 0.0
	for i := 0; i < 12; i++ {
		sum += r.Float()
	}
	return sum - 6.0
}
