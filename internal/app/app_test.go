package app

import "testing"

func TestRun(t *testing.T) {
	if err := Run(nil); err != nil {
		t.Fatalf("Run() returned error: %v", err)
	}
}
