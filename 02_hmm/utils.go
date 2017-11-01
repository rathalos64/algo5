package main

func isInStateSlice(s State, slice []State) bool {
	for _, elem := range slice {
		if elem == s {
			return true
		}
	}

	return false
}

func isInObservationSlice(s ObservationSymbol, slice []ObservationSymbol) bool {
	for _, elem := range slice {
		if elem == s {
			return true
		}
	}

	return false
}

func isInStringSlice(x string, elements []string) bool {
	for _, elem := range elements {
		if elem == x {
			return true
		}
	}

	return false
}

func hmmIDsToStringSlice(hmms []HMM) []string {
	ids := []string{}
	for _, hmm := range hmms {
		ids = append(ids, hmm.ID)
	}

	return ids
}
