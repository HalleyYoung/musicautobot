import os
import pickle
import random
from music21 import *

from musicautobot.numpy_encode import *
from musicautobot.utils.file_processing import process_all, process_file
from musicautobot.config import *
from musicautobot.music_transformer import *

# Location of your midi files
midi_path =  Path('data/midi/referencemids')

# Location of saved datset
data_path = Path('data/numpy')

midiprobs = pickle.load(open("midiprobs.pcl", "rb"))
all_inds = list(range(len(midiprobs)))
all_inds.sort(reverse=True, key = lambda k: midiprobs[k])


# Data
data = MusicDataBunch.empty(data_path)
vocab = data.vocab

pretrained_path = Path("data/MusicTransformer.pth")

learn = music_model_learner(data, pretrained_path=pretrained_path)

features = pickle.load(open("reference_features.pcl", "rb"))
ref_measures = pickle.load(open("allrefmeasures.pcl", "rb"))
ref_profile = pickle.load(open("prevprofiles.pcl", "rb"))


midi_files = get_files(midi_path, recurse=True, extensions='.mid')
random.shuffle(midi_files)


for z in range(1005, 1015):#all_inds[10:]:
	s = stream.Score()
	onset = 0.0
	seed_item = MusicItem.from_file(Path("data/midi/referencemidsC/" + str(z) + ".mid"), data.vocab)
	pred = learn.predict(seed_item, n_words=400, temperatures=(0.2,0.4), min_bars=12, top_k=24, top_p=0.7, references=features[z], ref_measures=ref_measures[z], pitch_profile = ref_profile[z])

	prev_note = 60
	for val in pred:
		a = val.to_text().split()
		xxseps = [0] + [i for i in range(len(a)) if a[i].startswith("xx")]
		between_seps = [a[xxseps[i]:xxseps[i + 1]] for i in range(len(xxseps) - 1)]

		print(str(len(between_seps)) + " between")
		pits = []
		durs = []
		for q in between_seps:
			pits.append([int(z[1:]) for z in q if z.startswith("n")][0])
			durs.append([(int(z[1:]) + 1)/4 for z in q if z.startswith("d")][0])


		#pits = [int(a[i + 1][1:]) for i in xxseps]
		#print((pits))
		#durs = [(int(a[i+2][1:]) + 1)/4 for i in xxseps]
		print((durs))

		if len(pits) != len(durs):
			print(a)

		
		for i in range(len(pits)):
			pc = pits[i] % 12
			closest_pc = min(range(48 + pc, pc+84, 12), key = lambda j: abs(j - prev_note))
			n = note.Note(closest_pc)
			n.quarterLength = (durs[i])
			s.insert(onset, n)
			onset += durs[i]
			prev_note = closest_pc
	s.write(fp="our-model/test" + str(z) + ".mid", fmt="midi")
