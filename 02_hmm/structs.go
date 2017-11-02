package main

import (
	"encoding/json"
)

// ValidationResult contains the errors within
// a validation process. An error is keyed with an identifier.
type ValidationResult map[string]string

// Add extends the current validation result by another entry
func (val ValidationResult) Add(key string, value string) {
	val[key] = value
}

// Include extends the current validation result by another.
func (val ValidationResult) Include(v ValidationResult) {
	for key, value := range v {
		val[key] = value
	}
}

// Valid checks whether the validation result contains no
// errors.
func (val ValidationResult) Valid() bool {
	return len(val) == 0
}

// Marshal serialized the validation result to JSON.
func (val ValidationResult) Marshal() string {
	b, err := json.MarshalIndent(val, "", "\t")
	if err != nil {
		b = []byte("")
	}

	return string(b)
}

// Result represents the result of the evaluation of
// sequence given individual HMMs. The ID is the ID of the HMM.
type Result struct {
	ID            string
	Likelihood    float64
	LogLikelihood float64
}

// Better determines by comparing it with another result whether
// the current result is better.
func (res Result) Better(x Result) bool {
	return res.Likelihood > x.Likelihood
}
