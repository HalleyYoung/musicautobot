import os
import pickle
from musicautobot.numpy_encode import *
from musicautobot.utils.file_processing import process_all, process_file
from musicautobot.config import *
from musicautobot.music_transformer import *

midi_path = Path('data/midi/nlb')
data_path = Path('data/numpy')
data_save_name = 'musicitem_data_save.pkl'
pretrained_path = Path("MusicTransformer.pth")

midi_files = get_files(midi_path, '.mid', recurse=True)
data = pickle.load(open(data_save_name, "rb"))#MusicDataBunch.from_files(midi_files, data_path, processors=[Midi2ItemProcessor()], bs=4, bptt=128, encode_position=False)
#pickle.dump(data, open(data_save_name, "wb"))

learn = music_model_learner(data, arch=TransformerXL, config=default_config(), pretrained_path=pretrained_path)


for i in range(100):
	print(i)
	learn.fit_one_cycle(1)
	learn.save(Path("models/out" + str(i) + ".pth"))
