// Package app holds the core application logic, kept separate from the
// cmd entrypoint so it can be tested and reused.
package app

import "fmt"

// Run is the application entrypoint. It receives the CLI args (excluding the
// program name) and returns an error to be surfaced to the user.
func Run(args []string) error {
	fmt.Println("simple-lang-model: hello 👋")
	return nil
}
