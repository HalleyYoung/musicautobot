import random
import numpy as np

def isValidPitch(prev_notes, ref_measures, ref_measure, references, k):
    if len(prev_notes) >= 2 and k % 12 == prev_notes[-1] % 12 and k % 12  == prev_notes[-2] % 12:
        return False
    ref_measure = [i - 20 if i >  88 else i for i in ref_measure]

    for i in range(1,4):
        if len(prev_notes) >= i and len(ref_measure) <= i:
            return k % 12 in [i % 12 for i in ref_measure]
    #print(ref_measure)
    next_ref = ref_measure[len(prev_notes)] if len(ref_measure) > len(prev_notes) else ref_measure[random.choice([-3,-2,-1])]
    if references["hasSamePitches"]:
        if not abs(k % 12) in [i % 12 for i in ref_measure]:
            if k % 12 not in [i % 12 for i in ref_measures]:
                return False
            return random.uniform(0,1) < 0.4
            #print("fail 1")

    if references["hasSamePitchesPrefix"]:
        if len(prev_notes) <= 3:
            if not k % 12 in [w % 12 for w in ref_measure[:4]]:
                #print("fail 2")
                if k % 12 not in [i % 12 for i in ref_measures]:
                    return False
                return random.uniform(0,1) < 0.4
    if references["hasAddOnePitchSymmetry"] or references["hasChangeOnePitchSymmetry"]:
        if (not next_ref % 12 == k % 12):
            if k % 12 not in [i % 12 for i in ref_measures]:
                return False
            return random.uniform(0,1) < 0.4
            #print("fail 3")
    return True
    if len(prev_notes) > 1:
        pass
        if references["hasIntervalSymmetry"] or references["hasIntervalPrefix"] or references["hasIntervalSuffix"]:
            kth_int_in_reference = next_ref - ref_measure[min(len(ref_measure) - 1, len(prev_notes) - 1)]
            if not abs((k - prev_notes[-1]) - kth_int_in_reference) <= 1:
                if references["hasIntervalSymmetry"]:
                    return False
                if references["hasIntervalPrefix"]:
                    if len(prev_notes) < 3:
                        return False
                if references["hasIntervalPrefix"]:
                     if len(prev_notes) > 6:
                        return False
    if (references["hasSameContour"] or references["hasSameContourPrefix"] or references["hasSameContourSuffix"]) and len(prev_notes) > 0:
        return True
        ref_measure_contour = [len([q for q in ref_measure if ref_measure[i] <= q]) for i in range(len(ref_measure))]
        k_contour = len([q for q in prev_notes if k <= q])
        if len(ref_measure_contour) > len(prev_notes):
            if k_contour != ref_measure_contour[len(prev_notes)]:
                #print("fail 5")
                if references["hasSameContour"]:
                    return False
                elif references["hasSameContourPrefix"]:
                    if len(prev_notes) < 3:
                        return False
                if references["hasSameContourSuffix"]:
                    if len(prev_notes) >= len(ref_measure) - 3:
                        return False
    return True



def isValidDur(prev_notes, ref_measure, references, k):
    #print("8ish 1")
    onsets = [sum(prev_notes[:i + 1]) for i in range(len(prev_notes))] if len(prev_notes) > 0 else [0]
    #print((onsets, prev_notes))
    ref_onsets = [sum(ref_measure[:i]) for i in ref_measure]
    if len(onsets) > 0:
        #if onsets[-1] > 0:
        #    print(k, onsets[-1] + k % 2 == 1, (onsets[-1] + k) in [sum(ref_measure[:i]) for i in range(len(ref_measure))])
        if (onsets[-1] + k) % 2 == 1:
            if not (onsets[-1] + k) in [sum(ref_measure[:i]) for i in range(len(ref_measure))]:
                #if k == 8 and  len(ref_measure) > 2 and ref_measure[2] == 8 and len(prev_notes) == 2:
                #    print("in 8 bad 1" + str(ref_measure) + str(prev_notes))
                return False
    

    cur_onset = sum(prev_notes)
    dur_at_onset_ind = [i for i in range(len(ref_measure)) if sum(ref_measure[:i]) <= cur_onset][-1]
    onset_at_next_onset = sum(ref_measure[:dur_at_onset_ind + 1])
    dur_at_onset = ref_measure[dur_at_onset_ind]

    if (k) == 1 and not any([sum(ref_measure[:i]) == sum(prev_notes) and ref_measure[i] == k for i in range(len(ref_measure))]):
        return False
    #print((prev_notes, ref_measure, onset_at_next_onset, sum(prev_notes) + k, k))
    if sum(prev_notes) + k > onset_at_next_onset:
        #print("greater " + str((prev_notes, ref_measure, k)))
        #    print("in 8 bad 2 " + str(ref_measure) + str(prev_notes))
        return False

    #print(ref_measure[len(prev_notes)])
    if False:#references["hasNonTrivialPrefix"] or references["hasNonTrivialSuffix"] or True:
        if len(prev_notes) < 3:
            if (ref_measure[dur_at_onset_ind] != k + 1) and not (sum(prev_notes) == 15):
                return False
             #or random.uniform(0,1) < 0.1
        else:
            pass
    if False:#references["hasNonTrivialSuffix"]:
        if len(prev_notes) > 5:
            if (not ref_measure[dur_at_onset_ind] == k + 1) and not (sum(prev_notes) == 15):
                return False #or random.uniform(0,1) < 0.1
        else:
            pass
    if references["hasSameRhythm"] or references["hasAddOneRhythmSymmetry"] or references["hasSubsetRhythm"] or references["hasNonTrivialPrefix"] or references["hasNonTrivialSuffix"] or random.uniform(0,1) < 0.8:
        #print("getting here 3")
        a = ref_measure[dur_at_onset_ind] == k or (ref_measure[dur_at_onset_ind] >= k and sum(prev_notes) + k % 2 != 1 and random.uniform(0,1) < 0.1) \
        or ((sum(prev_notes) + k) in ref_onsets and random.uniform(0,1) < 0.2)
        #if k == 8 and  len(ref_measure) > 2 and ref_measure[2] == 8:
        #    print("in 8 " + str(a))
        return a
        if a:
            #print("k is " + str(k))
            return True
        # or (sum(prev_notes) >= 11 and random.uniform(0,1) < 0.05) or (sum([i for i in prev_notes]) == 15)
        #rhys_so_far = sum(ref_measure) + k
        #onset_on_in_ref = rhys_so_far in [sum(ref_measure[:i]) for i in range(len(ref_measure))]
        #return onset_on_in_ref
    return True