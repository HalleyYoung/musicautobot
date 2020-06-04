import random
import numpy as np

def isValidPitch(prev_notes, ref_measure, references, k):
    ref_measure = [i - 20 if i >  88 else i for i in ref_measure]
    #print(ref_measure)
    next_ref = ref_measure[len(prev_notes)] if len(ref_measure) > len(prev_notes) else ref_measure[random.choice([-3,-2,-1])]
    if references["hasSamePitches"]:
        if not abs(k % 12) in [i % 12 for i in ref_measure]:
            #print("fail 1")

            return False
    if references["hasSamePitchesPrefix"]:
        if len(prev_notes) <= 3:
            if not k % 12 in [w % 12 for w in ref_measure[:4]]:
                #print("fail 2")

                return False
    if references["hasAddOnePitchSymmetry"] or references["hasChangeOnePitchSymmetry"]:
        if (not next_ref % 12 == k % 12) and random.uniform(0,1) < 0.9:
            #print("fail 3")
            return False
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
    if len(prev_notes) >= len(ref_measure):
        return True
    #print(ref_measure[len(prev_notes)])
    if references["hasNonTrivialPrefix"] or references["hasNonTrivialSuffix"] or True:
        if len(prev_notes) < 3:
            if (ref_measure[len(prev_notes)] != k + 1) and not (sum(prev_notes) == 15):
                return False
             #or random.uniform(0,1) < 0.1
        else:
            pass
    if references["hasNonTrivialSuffix"]:
        if len(prev_notes) > 5:
            if (not ref_measure[len(prev_notes)] == k + 1) and not (sum(prev_notes) == 15):
                return False #or random.uniform(0,1) < 0.1
        else:
            pass
    if references["hasSameRhythm"] or references["hasAddOneRhythmSymmetry"] or references["hasSubsetRhythm"] or random.uniform(0,1) < 0.8:
        a = ref_measure[len(prev_notes)] == k + 1
        if a:
            #print("k is " + str(k))
            return True
        # or (sum(prev_notes) >= 11 and random.uniform(0,1) < 0.05) or (sum([i for i in prev_notes]) == 15)
        rhys_so_far = sum(ref_measure) + k
        onset_on_in_ref = rhys_so_far in [sum(ref_measure[:i]) for i in range(len(ref_measure))]
        return onset_on_in_ref
    return True