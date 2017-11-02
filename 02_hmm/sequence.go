package main

import "fmt"

// Sequence represents a succession of observations which will be tested
// against Hidden Markov Models in order to determine the likeliest HMM.
type Sequence struct {
	ID           string              `json:"id"`
	HMMIDs       []string            `json:"hmm_ids"`
	Observations []ObservationSymbol `json:"observations"`
}

// Validate verifies the validity of the sequence.
func (seq Sequence) Validate(hmms []HMM) ValidationResult {
	validationResult := ValidationResult{}

	if seq.ID == "" {
		validationResult.Add("ID", "Sequence must have an ID")
	}

	if len(seq.HMMIDs) == 0 {
		validationResult.Add("HMMIDs", "No HMM Ids given")
	}

	// verify that the HMMs which the sequence tests against, exists
	hmmIDs := hmmIDsToStringSlice(hmms)
	for _, seqHMMID := range seq.HMMIDs {
		if !isInStringSlice(seqHMMID, hmmIDs) {
			validationResult.Add("HMMIDs",
				fmt.Sprintf("HMM with ID '%s' does not exist", seqHMMID))
			break
		}
	}

	// verify that the unique observation symbols do exist in all HMMs
	// the sequence is testing against
	for _, symbol := range seq.Observations {
		for _, hmm := range hmms {
			// test only against the target HMMs
			if !isInStringSlice(hmm.ID, seq.HMMIDs) {
				continue
			}
			if !isInObservationSlice(symbol, hmm.V) {
				validationResult.Add("Observations",
					fmt.Sprintf("Symbol '%q' doesn't exists in target HMMs", symbol))
				break
			}
		}
	}
	return validationResult
}
