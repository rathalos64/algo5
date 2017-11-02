package main

// isInStateSlice checks whether a given state is
// in an array of states.
func isInStateSlice(s State, slice []State) bool {
	for _, elem := range slice {
		if elem == s {
			return true
		}
	}

	return false
}

// isInStateSlice checks whether a given observation symbol is in
// an array of observation symbols.
func isInObservationSlice(s ObservationSymbol, slice []ObservationSymbol) bool {
	for _, elem := range slice {
		if elem == s {
			return true
		}
	}

	return false
}

// isInStringSlice checks whether a given string x is in an
// array of strings.
func isInStringSlice(x string, elements []string) bool {
	for _, elem := range elements {
		if elem == x {
			return true
		}
	}

	return false
}

// hmmIDsToStringSlice extracts the ID of each given HMM
// and returns it as an array.
func hmmIDsToStringSlice(hmms []HMM) []string {
	ids := []string{}
	for _, hmm := range hmms {
		ids = append(ids, hmm.ID)
	}

	return ids
}
