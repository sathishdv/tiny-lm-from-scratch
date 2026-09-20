package token

import "strings"

// Example is one training pair: two context word IDs -> the correct next word.
// "the cat sat" yields: (<S>,<S>)->the, (<S>,the)->cat, (the,cat)->sat, (cat,sat)-><.>.

type Example struct {
	// Context is the two words that precede the next word. The first word is
	// the older context, and the second word is the more recent context.
	Context [2]int

	// Next is the correct next word that follows the context.
	Next int
}

// Dataset is a collection of training examples. It is built from sentences
// using a vocabulary to convert words to IDs.
type Dataset struct {
	Examples []Example
}

// NewDataset turns sentences into small next-word prediction exercises for the
// model. The vocabulary must already contain every word in the sentences; this
// function replaces each word with its vocabulary ID before making examples.
//
// For example, consider the sentence "the cat sat". The function places two
// <S> markers before it and an <.> marker after it, as though the model sees:
//
//	<S> <S> the cat sat <.>
//
// It then slides a three-word window from left to right. The first two words
// are the question (the context), and the third word is the answer (Next):
//
//	context: <S>, <S>  -> next: the
//	context: <S>, the  -> next: cat
//	context: the, cat  -> next: sat
//	context: cat, sat  -> next: <.>
//
// The two <S> markers let the model learn how a sentence begins, while <.>
// gives it a target for when a sentence ends. The returned Dataset stores the
// same pairs as integer IDs because the model works with numbers, not text.
func NewDataset(sentences []string, vocab *Vocab) *Dataset {
	dataset := &Dataset{}

	for _, sentence := range sentences {
		words := strings.Split(sentence, " ")
		ids := make([]int, len(words))
		for i, word := range words {
			ids[i] = vocab.ToID[word]
		}

		// Add start and end tokens
		ids = append([]int{vocab.ToID[TokenStart], vocab.ToID[TokenStart]}, ids...)
		ids = append(ids, vocab.ToID[TokenEnd])

		for i := 2; i < len(ids); i++ {
			example := Example{
				Context: [2]int{ids[i-2], ids[i-1]},
				Next:    ids[i],
			}
			dataset.Examples = append(dataset.Examples, example)
		}
	}

	return dataset
}
