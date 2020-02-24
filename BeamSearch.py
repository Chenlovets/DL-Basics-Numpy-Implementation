def GreedySearch(SymbolSets, y_probs):
	'''
	SymbolSets: This is the list containing all the symbols i.e. vocabulary (without blank)

	y_probs: Numpy array of (# of symbols+1, Seq_length, batch_size). Note that your batch size for part 1 would always remain 1, but if you plan to use your implementation for part 2 you need to incorporate batch_size.
	Return the forward probability of greedy path and corresponding compressed symbol sequence i.e. without blanks and repeated symbols.
	'''
	import numpy as np
	for i in range(y_probs.shape[2]):
		max_prob = np.prod(np.max(y_probs[:,:,i], axis=0))

	sequence = []
	for index in np.argmax(y_probs, axis=0):
		if len(sequence) == 0:
			if index != 0:
				sequence.append(SymbolSets[index-1])
		elif sequence[-1] != SymbolSets[index-1]:
			sequence.append(SymbolSets[index-1])

	compressed_sequence = ''
	for s in sequence:
		if s != '-':
			compressed_sequence += s
	return compressed_sequence, max_prob                

def BeamSearch(SymbolSets, y_probs, BeamWidth):
	'''
	SymbolSets: This is the list containing all the symbols i.e. vocabulary (without blank)
    
	y_probs: Numpy array of (# of symbols+1,Seq_length,batch_size). Note that your batch size for part 1 would always remain 1, but if you plan to use your implementation for part 2 you need to incorporate batch_size.
    
	BeamWidth: Width of the beam.

	The function should return the symbol sequence with the best path score (forward probability) 
	and a dictionary of all the final merged paths with their scores. 

	'''
	PathScore = {}
	BlankPathScore = {}
	blank = ''

	def InitializePaths(SymbolSets, y_probs, BeamWidth):

		InitialPathsWithFinalBlank = []
		InitialPathsWithFinalSymbol = []

		BlankPathScore[blank] = y_probs[0]
		InitialPathsWithFinalBlank.append(blank)

		for index, c in enumerate(SymbolSets):
			path = c
			PathScore[path] = y_probs[index+1]
			InitialPathsWithFinalSymbol.append(path)

		return Prune(InitialPathsWithFinalBlank, InitialPathsWithFinalSymbol, BlankPathScore, PathScore, BeamWidth)

	def Prune(PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore, BeamWidth):

		PrunedBlankPathScore = {}
		PrunedPathScore = {}
		scorelist = []

		# First gather all the relevant scores
		for p in PathsWithTerminalBlank:
			scorelist.append(BlankPathScore[p])
		for p in PathsWithTerminalSymbol:
			scorelist.append(PathScore[p])

		# Sort and find cutoff score that retains exactly BeamWidth paths
		scorelist.sort(reverse=True) 
		cutoff = scorelist[BeamWidth]
			
		PrunedPathsWithTerminalBlank = [] 
		for p in PathsWithTerminalBlank:
			if BlankPathScore[p] > cutoff:
				PrunedPathsWithTerminalBlank.append(p) # Set addition 
				PrunedBlankPathScore[p] = BlankPathScore[p]

		PrunedPathsWithTerminalSymbol = [] 
		for p in PathsWithTerminalSymbol:
			if PathScore[p] > cutoff: 
				PrunedPathsWithTerminalSymbol.append(p) # Set addition 
				PrunedPathScore[p] = PathScore[p]

		return PrunedPathsWithTerminalBlank, PrunedPathsWithTerminalSymbol, PrunedPathScore, PrunedBlankPathScore

	def ExtendWithBlank(PathsWithTerminalBlank, PathsWithTerminalSymbol, y):

		UpdatedPathsWithTerminalBlank = []
		UpdatedBlankPathScore = {}

		# First work on paths with terminal blanks，Repeating a blank doesn’t change the symbol sequence
		for path in PathsWithTerminalBlank:
			UpdatedPathsWithTerminalBlank.append(path)
			UpdatedBlankPathScore[path] = BlankPathScore[path] * y[0]
		# Then extend paths with terminal symbols by blanks
		for path in PathsWithTerminalSymbol:
			# If there is already an equivalent string in UpdatesPathsWithTerminalBlank, simply add the score. If not create a new entry
			if path in UpdatedPathsWithTerminalBlank:
				UpdatedBlankPathScore[path] += PathScore[path] * y[0]
			else:
				UpdatedPathsWithTerminalBlank.append(path)
				UpdatedBlankPathScore[path] = PathScore[path] * y[0]

		return UpdatedPathsWithTerminalBlank, UpdatedBlankPathScore

	def ExtendWithSymbol(PathsWithTerminalBlank, PathsWithTerminalSymbol, y, SymbolSets):

		UpdatedPathsWithTerminalSymbol = []
		UpdatedPathScore = {}

		# First work on paths with terminal symbols 
		for path in PathsWithTerminalSymbol:
			# Extend the path with every symbol other than blank
			for index, c in enumerate(SymbolSets):
				if c == path[-1]:
					newpath = path
				else:
					newpath = path + c
				if newpath not in UpdatedPathsWithTerminalSymbol:
					UpdatedPathsWithTerminalSymbol.append(newpath)
					UpdatedPathScore[newpath] = PathScore[path] * y[index + 1]
				else:
					UpdatedPathScore[newpath] += PathScore[path] * y[index + 1]


		# Then work on paths terminating in blanks
		for path in PathsWithTerminalBlank:
			for index, c in enumerate(SymbolSets):
				newpath = path + c
				# If there is already an equivalent string in UpdatesPathsWithTerminalSymbol, simply add the score. If not create a new entry
				if newpath in UpdatedPathsWithTerminalSymbol:
					UpdatedPathScore[newpath] += BlankPathScore[path] * y[index + 1]
				else:
					UpdatedPathsWithTerminalSymbol.append(newpath)
					UpdatedPathScore[newpath] = BlankPathScore[path] * y[index + 1]

		return UpdatedPathsWithTerminalSymbol, UpdatedPathScore

	def MergeIdenticalPaths(PathsWithTerminalBlank, PathsWithTerminalSymbol):

		FinalPathScore = {}

		# All paths with terminal symbols will remain
		MergedPaths = PathsWithTerminalSymbol 
		for index, p in enumerate(MergedPaths):
			FinalPathScore[p] = PathScore[p]

		# Paths with terminal blanks may contribute scores to existing identical paths
		for p in PathsWithTerminalBlank:
			if p in MergedPaths:
				FinalPathScore[p] += BlankPathScore[p]
			else:
				MergedPaths.append(p)
				FinalPathScore[p] = BlankPathScore[p]

		return MergedPaths, FinalPathScore

    # First time instant: Initialize paths with each of the symbols, including blank, using score at time t=1
	PathsWithTerminalBlank, PathsWithTerminalSymbol, PathScore, BlankPathScore = InitializePaths(SymbolSets, y_probs[:, 0, 0], BeamWidth)
    # Subsequent time steps
	for t in range(1, y_probs.shape[1]):
    	# First extend paths by a blank
		UpdatedPathsWithTerminalBlank, UpdatedBlankPathScore = ExtendWithBlank(PathsWithTerminalBlank, PathsWithTerminalSymbol, y_probs[:, t, 0])
    	# Next extend paths by a symbol
		UpdatedPathsWithTerminalSymbol, UpdatedPathScore = ExtendWithSymbol(PathsWithTerminalBlank, PathsWithTerminalSymbol, y_probs[:, t, 0], SymbolSets)
    	# Prune the collection down to the BeamWidth
		PathsWithTerminalBlank, PathsWithTerminalSymbol, PathScore, BlankPathScore = Prune(UpdatedPathsWithTerminalBlank, UpdatedPathsWithTerminalSymbol,
                                         UpdatedBlankPathScore, UpdatedPathScore, BeamWidth)

    # Merge identical paths differing only by the final blank
	MergedPaths, FinalPathScore = MergeIdenticalPaths(PathsWithTerminalBlank, PathsWithTerminalSymbol)
    
    # Pick best path
	BestPath = max(FinalPathScore, key=FinalPathScore.get)

	return BestPath, FinalPathScore







    