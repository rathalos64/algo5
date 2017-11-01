package main

import (
	"encoding/json"
)

type ValidationResult map[string]string

func (val ValidationResult) Add(key string, value string) {
	val[key] = value
}

func (val ValidationResult) Include(v ValidationResult) {
	for key, value := range v {
		val[key] = value
	}
}

func (val ValidationResult) Valid() bool {
	return len(val) == 0
}

func (val ValidationResult) Marshal() string {
	b, err := json.MarshalIndent(val, "", "\t")
	if err != nil {
		b = []byte("")
	}

	return string(b)
}

type Result struct {
	ID            string
	Likelihood    float64
	LogLikelihood float64
}

func (res Result) Better(x Result) bool {
	return res.Likelihood > x.Likelihood && res.LogLikelihood < x.LogLikelihood
}
