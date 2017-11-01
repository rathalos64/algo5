package main

type State string
type ObservationSymbol string

// HMM models a hidden markov model with all possible parameter as fields
//  S: represents the individual states {s1, s2, ... sN}
//  N: is the number of states
//
//  V: represents the individual observation symbols (all observations types known)
//  M: is the number of distinct observation symbols per state
//
// PI: represents the initial state distribution (initial start vector)
//  A: represents the state transition probability distribution
//  B: represents the observation symbol probability distribution (emission probabilities)
//
// In order to identify sequences by their observation symbol PI, A and B are keyed with the
// state / observation symbol.
type HMM struct {
	ID string `json:"id"`

	S []State `json:"states"`
	N int

	V []ObservationSymbol `json:"observations"`
	M int

	PI map[State]float64                       `json:"initial"`
	A  map[State]map[State]float64             `json:"transitions"`
	B  map[State]map[ObservationSymbol]float64 `json:"emissions"`
}

// Validate verifies the constraints of a given HMM; not only
// mathematical but functional ones
func (hmm HMM) Validate() ValidationResult {
	validationResult := ValidationResult{}

	if hmm.ID == "" {
		validationResult.Add("ID", "HMM must have an ID")
	}

	validationResult.Include(hmm.validateS())
	validationResult.Include(hmm.validateV())
	validationResult.Include(hmm.validatePI())
	validationResult.Include(hmm.validateA())
	validationResult.Include(hmm.validateB())

	return validationResult
}

func (hmm HMM) validateS() ValidationResult {
	validationResult := ValidationResult{}
	uniqueElements := map[string]bool{}

	// verify the uniqueness of given states
	for _, elem := range hmm.S {
		if uniqueElements[string(elem)] == true {
			validationResult.Add("S", "States must be unique")
			break
		}

		uniqueElements[string(elem)] = true
	}

	return validationResult
}

func (hmm HMM) validateV() ValidationResult {
	validationResult := ValidationResult{}
	uniqueElements := map[string]bool{}

	// verify the uniqueness of given observation symbols
	for _, elem := range hmm.V {
		if uniqueElements[string(elem)] == true {
			validationResult.Add("V", "Observation symbols must be unique")
			break
		}

		uniqueElements[string(elem)] = true
	}

	return validationResult
}

func (hmm HMM) validatePI() ValidationResult {
	validationResult := ValidationResult{}

	// verify size (1xN)
	if len(hmm.PI) != hmm.N {
		validationResult.Add("PI", "Invalid number of start probabilities; must be N or 1 for each state")
	}

	// verify vector identifier
	for key := range hmm.PI {
		if !isInStateSlice(key, hmm.S) {
			validationResult.Add("PI", "Given state identifier does not exist")
			break
		}
	}

	// verify probabilities
	for _, value := range hmm.PI {
		if value < 0 {
			validationResult.Add("PI", "Probabilities must be > 0")
			break
		}
	}

	// verify that the initial probabilities sum up to 1
	var probability float64
	for _, value := range hmm.PI {
		probability += value
	}

	if probability != 1.0 {
		validationResult.Add("PI", "Initial probabilities must sum up to 1.0")
	}

	return validationResult
}

func (hmm HMM) validateA() ValidationResult {
	validationResult := ValidationResult{}

	// verify matrix size (NxN)
	if len(hmm.A) == hmm.N {
		for row := range hmm.A {
			if cols := hmm.A[row]; len(cols) != hmm.N {
				validationResult.Add("A", "Invalid number of columns; must be N")
				break
			}
		}
	} else {
		validationResult.Add("A", "Invalid number of rows; must be N")
	}

	// verify matrix identifier
	for row := range hmm.A {
		if !isInStateSlice(row, hmm.S) {
			validationResult.Add("A", "Given state identifier of row does not exist")
			break
		}

		for col := range hmm.A[row] {
			if !isInStateSlice(col, hmm.S) {
				validationResult.Add("A", "Given state identifier of col does not exist")
				break
			}
		}
	}

	// verify probabilities
	for row := range hmm.A {
		for col := range hmm.A[row] {
			if hmm.A[row][col] < 0 {
				validationResult.Add("A", "Probabilities must be > 0")
				break
			}
		}
	}

	// verify sum of transition probabilities
	// All probabilities in each row must sum up to 1
	// Or each transition from a fixed state si to all other states sj (assuming that every state
	// can reach every other state) must sum up to 1
	for row := range hmm.A {
		var probability float64

		for col := range hmm.A[row] {
			probability += hmm.A[row][col]
		}

		if probability != 1.0 {
			validationResult.Add("A", "All state transition probabilities in each row must sum up to 1")
			break
		}
	}

	return validationResult
}

func (hmm HMM) validateB() ValidationResult {
	validationResult := ValidationResult{}

	// verify matrix size (NxM); number of states x number of observation symbols
	if len(hmm.B) == hmm.N {
		for row := range hmm.B {
			if cols := hmm.B[row]; len(cols) != hmm.M {
				validationResult.Add("B", "Invalid number of columns; must be M")
				break
			}
		}
	} else {
		validationResult.Add("B", "Invalid number of rows; must be N")
	}

	// verify matrix identifier
	for row := range hmm.B {
		if !isInStateSlice(row, hmm.S) {
			validationResult.Add("B", "Given state identifier of row does not exist")
			break
		}

		for col := range hmm.B[row] {
			if !isInObservationSlice(col, hmm.V) {
				validationResult.Add("B", "Given observation identifier of col does not exist")
				break
			}
		}
	}

	// verify probabilities
	for row := range hmm.B {
		for col := range hmm.B[row] {
			if hmm.B[row][col] < 0 {
				validationResult.Add("B", "Probabilities must be > 0")
				break
			}
		}
	}

	// verify sum of emission probabilities
	// All probabilities in each row must sum up to 1
	// Or for each state all observations must sum up to 1
	for row := range hmm.B {
		var probability float64

		for col := range hmm.B[row] {
			probability += hmm.B[row][col]
		}

		if probability != 1.0 {
			validationResult.Add("B", "All state emission probabilities in each row per state must sum up to 1")
			break
		}
	}

	return validationResult
}
