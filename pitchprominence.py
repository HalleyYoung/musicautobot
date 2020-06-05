def firstNote(xs, k):
	return (xs[0][0] % 12 == k, 2)

def lastNote(xs, k):
	return (xs[0][-1] % 12 == k, 2)

def nonPassingNeighboringTone(xs, k):
	pits = [i[0] for i in xs]
	return (len([q for q in range(1, len(pits)) if pits[q] % 12 == k and abs(pits[q] - pits[q - 1]) > 2 and abs(pits[q] - pits[q - 1]) > 2]) > 1, 2)

def nonPassingNeighboringChromaticTone(xs, k):
	pits = [i[0] for i in xs]
	return (len([q for q in range(1, len(pits)) if pits[q] % 12 == k and abs(pits[q] - pits[q - 1]) > 1 and abs(pits[q] - pits[q - 1]) > 1]) > 1, 2)

def sigTime1(xs, k):
	tot_len = sum([i[1] for i in xs])
	k_len = sum([i[1] for i in xs if i[0] % 12 == k])
	return (k_len/tot_len > 0.1, 1)

def sigTime2(xs, k):
	tot_len = sum([i[1] for i in xs])
	k_len = sum([i[1] for i in xs if i[0] % 12 == k])
	return (k_len/tot_len >= 0.25, 2)

def sigTime3(xs, k):
	tot_len = sum([i[1] for i in xs])
	k_len = sum([i[1] for i in xs if i[0] % 12 == k])
	return (k_len/tot_len > 0.25, 2)

def inMeasure(xs, k):
	return (k % 12 in [i[0] % 12 for i in xs], 1)