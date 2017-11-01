package main

// Evaluate calculates the likelihood that a HMM generated the observed sequence.
// It uses the forward-algorithm consisting of three steps - initialization,
// induction and termination - to find the likelihood efficiently.
func Evaluate(seq Sequence, hmm HMM) float64 {
	T := len(seq.Observations)

	var probability float64
	for _, state := range hmm.S {
		probability += Induction(seq, hmm, state, T)
	}

	return probability
}

// Induction calculates for a given state s in a HMM the probability of
// reaching the state at point t - by calculating all previous paths to s at t - and
// observing the symbol of the sequence at point t
func Induction(seq Sequence, hmm HMM, s State, t int) float64 {
	var probability float64

	// tPrev means the previous point while every
	// array access with t - 1 means at point t with adjusting
	// to the array indexing order starting from 0
	tPrev := t - 1

	for _, fromState := range hmm.S {
		// the probability of the previous point; helper variable
		var pPrev float64

		// if the previous point is the first, calculate it using the initial probabilities;
		// otherwise, do another induction step at point tPrev = t - 1
		if tPrev > 1 {
			pPrev = Induction(seq, hmm, fromState, tPrev)
		} else {
			pPrev = hmm.PI[fromState] * hmm.B[fromState][seq.Observations[tPrev-1]]
		}

		// the previous probability at t - 1 given a previous state multiplied by the
		// probability of actually transition from the previous state into the actual one
		// at t
		probability += (pPrev * hmm.A[fromState][s])
	}

	// all probabilities of reaching state s at point t multiplied by the probability
	// of observing the symbol of the sequence at time t given state s
	return probability * hmm.B[s][seq.Observations[t-1]]
}
