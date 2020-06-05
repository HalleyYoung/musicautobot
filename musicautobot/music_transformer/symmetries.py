def hasSamePitches(x,y):
	return [i[0] for i in x] == [i[0] for i in y]

def hasSamePitchesPrefix(x,y):
	return len(x) >= 3 and [i[0] for i in x[:3]] == [i[0] for i in y[:3]]

def hasAddOnePitchSymmetry(x,y):
	if abs(len(x) - len(y)) != 1:
		return False
	return any([[q[0] for q in x[:i]] + [q[0] for q in x[i+1:]] == y for i in range(len(x))]) or any([[q[0] for q in y[:i]] + [q[0] for q in y[i+1:]] == x for i in range(len(y))])

def hasSamePitchesSuffix(x,y):
	return len(x) >= 3 and [i[0] for i in x[-3:]] == [i[0] for i in y[-3:]]

def hasIntervalSymmetry(x,y):
	if len(x) != len(y):
		return False
	ints_x = [x[i][0] - x[i - 1][0] for i in range(1, len(x))]
	ints_y = [y[i][0] - y[i - 1][0] for i in range(1, len(y))]
	return len(ints_x) >= 3 and len([abs(ints_x[k] - ints_y[k]) <= 1 for k in range(len(ints_x))]) >= len(ints_x) - 2

def hasIntervalPrefix(x,y):
	ints_x = [x[i][0] - x[i - 1][0] for i in range(1, len(x))]
	ints_y = [y[i][0] - y[i - 1][0] for i in range(1, len(y))]
	return len(ints_x) >= 3 and len(ints_y) >= 3 and all([abs(ints_x[k] - ints_y[k]) <= 1 for k in range(3)])


def hasIntervalSuffix(x,y):
	ints_y = [y[i][0] - y[i - 1][0] for i in range(1, len(y))]
	ints_x = [x[i][0] - x[i - 1][0] for i in range(1, len(x))]
	return len(ints_x) >= 3 and len(ints_y) >= 3 and all([abs(ints_x[-k] - ints_y[-k]) <= 1 for k in range(1,4)])


def hasChangeOnePitchSymmetry(x,y):
	if len(x) != len(y):
		return False
	return any([[q[0] for q in x[:i]] + [q[0] for q in x[i+1:]] == [q[0] for q in y[:i]] + [q[0] for q in y[i+1:]] for i in range(len(x))])


def hasNonTrivialPrefix(x,y):
	if len(x) > 4 and len(y) > 4 and x[:3] == y[:3]:
		return True
	return False

def hasNonTrivialSuffix(x,y):
	if len(x) > 4 and len(y) > 4 and x[-3:] == y[-3:]:
		return True
	return False

def hasSameRhythm(x,y):
	return [q[1] for q in x] == [q[1] for q in y]

def hasAddOneRhythmSymmetry(x,y):
	if abs(len(x) - len(y)) != 1:
		return False
	onsets_x = [sum([q[1] for q in x[:i]]) for i in range(len(x))]
	onsets_y = [sum([q[1] for q in y[:i]]) for i in range(len(y))]
	return any([onsets_x == onsets_y[:i] + onsets_y[i + 1:] for i in range(len(y))]) or any([onsets_y == onsets_x[:i] + onsets_x[i + 1:] for i in range(len(x))])

def hasSubsetRhythm(x,y):
	onsets_x = [sum([q[1] for q in x[:i]]) for i in range(len(x))]
	onsets_y = [sum([q[1] for q in y[:i]]) for i in range(len(y))]
	return all([x in onsets_y for x in onsets_x]) or all([y in onsets_x for y in onsets_y])

def hasSameContour(x,y):
	cont_x_pit = [len(set([k for k in [q[0] for q in x] if k < x[i][0]])) for i in range(len(x))]
	cont_x_dur = [len(set([k for k in [q[1] for q in x] if k < x[i][1]]))  for i in range(len(x))]
	cont_y_pit = [len(set([k for k in [q[0] for q in y] if k < y[i][0]]))  for i in range(len(y))]
	cont_y_dur = [len(set([k for k in [q[1] for q in y] if k < y[i][1]]))  for i in range(len(y))]
	return (cont_x_pit == cont_y_pit) and (cont_x_dur == cont_y_dur)

def hasSameContourPrefix(x,y):
	cont_x_pit = [len(set([k for k in [q[0] for q in x] if k < x[i][0]])) for i in range(len(x))]
	cont_x_dur = [len(set([k for k in [q[1] for q in x] if k < x[i][1]]))  for i in range(len(x))]
	cont_y_pit = [len(set([k for k in [q[0] for q in y] if k < y[i][0]]))  for i in range(len(y))]
	cont_y_dur = [len(set([k for k in [q[1] for q in y] if k < y[i][1]]))  for i in range(len(y))]
	return len(cont_x_pit) >= 3 and (cont_x_pit[:3] == cont_y_pit[:3]) and (cont_x_dur[:3] == cont_y_dur[:3])

def hasSameContourSuffix(x,y):
	cont_x_pit = [len(set([k for k in [q[0] for q in x] if k < x[i][0]])) for i in range(len(x))]
	cont_x_dur = [len(set([k for k in [q[1] for q in x] if k < x[i][1]]))  for i in range(len(x))]
	cont_y_pit = [len(set([k for k in [q[0] for q in y] if k < y[i][0]]))  for i in range(len(y))]
	cont_y_dur = [len(set([k for k in [q[1] for q in y] if k < y[i][1]]))  for i in range(len(y))]
	return len(cont_x_pit) >= 3 and (cont_x_pit[-3:] == cont_y_pit[-3:]) and (cont_x_dur[-3:] == cont_y_dur[-3:])

