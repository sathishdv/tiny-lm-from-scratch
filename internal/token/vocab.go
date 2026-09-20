package token

import "strings"

// ============================================================================
// STAGE 2: TOKENIZER + DATASET
// Real LLMs use subword tokenizers (BPE). At this scale, whole words are our
// tokens. Every word gets an integer ID; the model only ever sees IDs.
// ============================================================================

var TrainSentences = []string{
	"the cat sat on the mat",
	"the dog sat on the mat",
	"the cat ran in the park",
	"the dog ran in the park",
	"the big cat sat on the mat",
	"the small dog ran in the park",
	"the big dog sat on the mat",
	"the small cat ran in the park",
	"the big dog ran in the park",
	"the small cat sat on the mat",
}

// Held-out sentences the model NEVER trains on. If loss on these also drops,
// the model has generalized the grammar rather than memorized the corpus.
var TestSentences = []string{
	"the small dog sat on the mat",
	"the big cat ran in the park",
}

const (
	TokenStart = "<S>" // padding token for the beginning of a sentence
	TokenEnd   = "<.>" // padding token for the end of a sentence
)

// Vocab is a mapping between words and their integer IDs.
type Vocab struct {
	ToID   map[string]int // word → ID
	ToWord []string       // ID → word
}

func NewVocab() *Vocab {
	v := &Vocab{
		ToID:   make(map[string]int),
		ToWord: []string{},
	}

	// Add the special tokens to the vocabulary.
	v.add(TokenStart)
	v.add(TokenEnd)

	return v
}

func (v *Vocab) Build(sentenceSets ...[]string) {
	for _, sentences := range sentenceSets {
		for _, sentence := range sentences {
			words := strings.Split(sentence, " ")
			for _, word := range words {
				v.add(word)
			}
		}
	}
}

func (v *Vocab) add(word string) {
	if _, exists := v.ToID[word]; !exists {
		v.ToWord = append(v.ToWord, word)
		v.ToID[word] = len(v.ToWord) - 1
	}
}
