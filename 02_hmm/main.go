package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
)

// Application is the main application and contains
// all necessary information for the whole program to work.
// It stores all defined models and all sequences of observations.
type Application struct {
	HMMs      []HMM
	Sequences []Sequence
}

func main() {
	// define command line flags
	precision := flag.String(
		"precision",
		"10",
		"the numerical precision for L(O|M) and log L(O|M)")
	hmmPath := flag.String(
		"models",
		"examples/models/models.json",
		"the path to the hmm json file")
	sequencePath := flag.String(
		"sequences",
		"examples/sequences/sequences.json",
		"the path to the sequence json file")
	flag.Parse()

	app := Application{}

	// read models
	err := app.ReadHMMs(*hmmPath)
	if err != nil {
		log.Fatalf("Couldn't read HMMs from %q: %s", hmmPath, err)
	}

	// validate all models
	validationResult := app.ValidateHMMs()
	if !validationResult.Valid() {
		log.Fatalf("Validation of all HMMs failed: \n%s",
			validationResult.Marshal())
	}
	// validate each models individually
	for _, hmm := range app.HMMs {
		validationResult := hmm.Validate()
		if !validationResult.Valid() {
			log.Fatalf("Validation of HMM %q failed: \n%s", hmm.ID,
				validationResult.Marshal())
		}
	}

	// read sequences
	err = app.ReadSequences(*sequencePath)
	if err != nil {
		log.Fatalf("Couldn't read Sequences from %q: %s", sequencePath, err)
	}

	// validate all models
	validationResult = app.ValidateSequences()
	if !validationResult.Valid() {
		log.Fatalf("Validation of all Sequences failed: \n%s",
			validationResult.Marshal())
	}
	// validate each sequences individually
	for _, sequence := range app.Sequences {
		validationResult := sequence.Validate(app.HMMs)
		if !validationResult.Valid() {
			log.Fatalf("Validation of Sequence %q failed: \n%s",
				sequence.ID, validationResult.Marshal())
		}
	}

	// evaluate for each sequence each model
	for _, sequence := range app.Sequences {
		fmt.Printf("=====================================" +
			"=================================\n")

		// the best model for the sequence
		best := Result{}

		fmt.Printf("[i] Evaluating Sequence %q\n", sequence.ID)
		for _, hmm := range app.HMMs {
			// test only against the target HMMs
			if !isInStringSlice(hmm.ID, sequence.HMMIDs) {
				continue
			}

			fmt.Printf("## using %q \n", hmm.ID)

			// calculate the likelihood of model given sequence
			likelihood := Evaluate(sequence, hmm)
			result := Result{
				ID:            hmm.ID,
				Likelihood:    likelihood,
				LogLikelihood: math.Log(likelihood),
			}

			if result.Better(best) {
				best = result
			}

			fmt.Printf("> L(O|M) = %."+*precision+"f \n", result.Likelihood)
			fmt.Printf("> log L(O|M) = %."+*precision+"f \n", result.LogLikelihood)
			fmt.Printf("---------------------------------\n")
		}

		fmt.Printf("## [RESULT] argmax(Mj) of P(%q | Mj) is\n", sequence.ID)
		fmt.Printf("> ID = %q \n", best.ID)
		fmt.Printf("> L(O|M) = %."+*precision+"f \n", best.Likelihood)
		fmt.Printf("> log L(O|M) = %."+*precision+"f\n", best.LogLikelihood)

		fmt.Printf("=====================================" +
			"=================================\n")
	}
}

// ReadHMMs reads all defined HMMs from the given path
// and stores it in the application
func (app *Application) ReadHMMs(path string) error {
	readerHmms, err := os.Open(path)
	if err != nil {
		return err
	}
	defer readerHmms.Close()

	// deserialize
	var hmms []HMM
	err = json.NewDecoder(readerHmms).Decode(&hmms)
	if err != nil {
		return err
	}

	// set N and M for each model
	for i := range hmms {
		hmms[i].N = len(hmms[i].S)
		hmms[i].M = len(hmms[i].V)
	}

	app.HMMs = hmms
	return nil
}

// ValidateHMMs verifies the validity of the read HMMs
func (app Application) ValidateHMMs() ValidationResult {
	validationResult := ValidationResult{}
	uniqueElements := map[string]bool{}

	if len(app.HMMs) == 0 {
		validationResult.Add("HMMs", "No HMMs given or invalid json file structure")
	}

	// verify the uniqueness of the hmm by inspecting ID
	for _, hmm := range app.HMMs {
		if uniqueElements[hmm.ID] == true {
			validationResult.Add("HMMs", "HMMs must have unique ID")
			break
		}
		uniqueElements[hmm.ID] = true
	}

	return validationResult
}

// ReadSequences reads all defined sequences of observations from the
// given path and stores it in the application
func (app *Application) ReadSequences(path string) error {
	sequenceReader, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer sequenceReader.Close()

	// deserialize
	var sequences []Sequence
	err = json.NewDecoder(sequenceReader).Decode(&sequences)
	if err != nil {
		return err
	}

	app.Sequences = sequences
	return nil
}

// ValidateSequences verifies the validity of the read sequences
func (app Application) ValidateSequences() ValidationResult {
	validationResult := ValidationResult{}
	uniqueElements := map[string]bool{}

	if len(app.Sequences) == 0 {
		validationResult.Add("Sequences", "No sequences given or invalid json file structure")
	}

	// verify the uniqueness of the hmm by inspecting ID
	for _, sequence := range app.Sequences {
		if uniqueElements[sequence.ID] == true {
			validationResult.Add("Sequences", "Must have unique ID")
			break
		}
		uniqueElements[sequence.ID] = true
	}

	return validationResult
}
