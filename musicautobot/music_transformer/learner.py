from fastai.basics import *
from fastai.text.learner import LanguageLearner, get_language_model, _model_meta
from .model import *
from .transform import MusicItem
from ..numpy_encode import SAMPLE_FREQ
from ..utils.top_k_top_p import top_k_top_p
from ..utils.midifile import is_empty_midi
from .is_valid import isValidPitch, isValidDur
from .pitchprofiletransformer import TransformerModel
import symmetries
import pitchprominence
from inspect import getmembers, isfunction
import torch
import copy
from scipy import spatial
from time import time

_model_meta[MusicTransformerXL] = _model_meta[TransformerXL] # copy over fastai's model metadata

model_pitch_profile = TransformerModel()
model_pitch_profile.load_state_dict(torch.load("transformernn/pitchtransformer.pth"))

seq_len = 20
symmetries_functions_list = [o for o in getmembers(symmetries) if isfunction(o[1])]

def getScore(vocab, poss_bar, references, ref_measure, prof):
    print("in getscore")
    pits = [i - vocab.note_range[0] for i in poss_bar if i in range(*vocab.note_range)]
    pcs = [i % 12 for i in pits]
    durs = [i - vocab.dur_range[0] + 1 for i in poss_bar if i in range(*vocab.dur_range)]
    pits_durs = [(pits[i], durs[i]) for i in range(len(durs))]
    pits_durs_profile = np.zeros(12)
    for pc in range(12):
        pits_durs_profile[pc] = sum([i[1] for i in pits_durs if i[0] % 12 == pc])
    symmetries_has = [(x[0], x[1](ref_measure, pits_durs)) for x in symmetries_functions_list]
    len_good = len([(sym_name, sym_value) for (sym_name, sym_value) in symmetries_has if sym_value and references[sym_name]])
    return spatial.distance.cosine(pits_durs_profile, prof) + 10*len_good - max(0, abs(3 - len(set(pcs))))



def getPitchProfile(prev_profile, k):
    a = model_pitch_profile(prev_profile.view(1,-1,12).float())
    b = a[0][-1][k % 12]
    return b

def concat(xss):
    new = []
    for xs in xss:
        new.extend(xs)
    return new

def music_model_learner(data:DataBunch, arch=MusicTransformerXL, config:dict=None, drop_mult:float=1.,
                        pretrained_path:PathOrStr=None, **learn_kwargs) -> 'LanguageLearner':
    "Create a `Learner` with a language model from `data` and `arch`."
    meta = _model_meta[arch]

    if pretrained_path: 
        state = torch.load(pretrained_path, map_location='cpu')
        if config is None: config = state['config']

    model = get_language_model(arch, len(data.vocab.itos), config=config, drop_mult=drop_mult)
    learn = MusicLearner(data, model, split_func=meta['split_lm'], **learn_kwargs)

    if pretrained_path: 
        get_model(model).load_state_dict(state['model'], strict=False)
        if not hasattr(learn, 'opt'): learn.create_opt(defaults.lr, learn.wd)
        try:    learn.opt.load_state_dict(state['opt'])
        except: pass
        del state
        gc.collect()

    return learn

# Predictions
from fastai import basic_train # for predictions
class MusicLearner(LanguageLearner):
    def save(self, file:PathLikeOrBinaryStream=None, with_opt:bool=True, config=None):
        "Save model and optimizer state (if `with_opt`) with `file` to `self.model_dir`. `file` can be file-like (file or buffer)"
        out_path = super().save(file, return_path=True, with_opt=with_opt)
        if config and out_path:
            state = torch.load(out_path)
            state['config'] = config
            torch.save(state, out_path)
            del state
            gc.collect()
        return out_path

    def beam_search(self, xb:Tensor, n_words:int, top_k:int=10, beam_sz:int=10, temperature:float=1.,
                    ):
        "Return the `n_words` that come after `text` using beam search."
        self.model.reset()
        self.model.eval()
        xb_length = xb.shape[-1]
        if xb.shape[0] > 1: xb = xb[0][None]
        yb = torch.ones_like(xb)

        nodes = None
        xb = xb.repeat(top_k, 1)
        nodes = xb.clone()
        scores = xb.new_zeros(1).float()
        with torch.no_grad():
            for k in progress_bar(range(n_words), leave=False):
                out = F.log_softmax(self.model(xb)[0][:,-1], dim=-1)
                values, indices = out.topk(top_k, dim=-1)
                scores = (-values + scores[:,None]).view(-1)
                indices_idx = torch.arange(0,nodes.size(0))[:,None].expand(nodes.size(0), top_k).contiguous().view(-1)
                sort_idx = scores.argsort()[:beam_sz]
                scores = scores[sort_idx]
                nodes = torch.cat([nodes[:,None].expand(nodes.size(0),top_k,nodes.size(1)),
                                indices[:,:,None].expand(nodes.size(0),top_k,1),], dim=2)
                nodes = nodes.view(-1, nodes.size(2))[sort_idx]
                self.model[0].select_hidden(indices_idx[sort_idx])
                xb = nodes[:,-1][:,None]
        if temperature != 1.: scores.div_(temperature)
        node_idx = torch.multinomial(torch.exp(-scores), 1).item()
        return [i.item() for i in nodes[node_idx][xb_length:] ]
    def predict3(self, item, reference_measures, references, prev_pitch_profile, prev_notes):
        self.model.reset()
        new_idx = []
        vocab = self.data.vocab
        x, pos = item.to_tensor(), item.get_pos_tensor()
        last_pos = pos[-1] if len(pos) else 0
        y = torch.tensor([0])

        start_pos = last_pos

        sep_count = 0
        bar_len = SAMPLE_FREQ * 4 # assuming 4/4 time
        vocab = self.data.vocab

        repeat_count = 0
        if hasattr(self.model[0], 'encode_position'):
            encode_position = self.model[0].encode_position
        else: encode_position = False


    def predict2(self, item:MusicItem, n_words:int=128,
                     temperatures:float=(1.0,1.0), min_bars=4,
                     top_k=30, top_p=0.6):
        self.model.reset()
        new_idx = []
        vocab = self.data.vocab
        x = x[-112:]
        x, pos = item.to_tensor(), item.get_pos_tensor()
        last_pos = pos[-1] if len(pos) else 0
        y = torch.tensor([0])

        start_pos = last_pos

        sep_count = 0
        bar_len = SAMPLE_FREQ * 4 # assuming 4/4 time
        vocab = self.data.vocab

        repeat_count = 0
        if hasattr(self.model[0], 'encode_position'):
            encode_position = self.model[0].encode_position
        else: encode_position = False

        for i in progress_bar(range(n_words), leave=True):
            with torch.no_grad():
                if encode_position:
                    batch = { 'x': x[None], 'pos': pos[None] }
                    logits = self.model(batch)[0][-1][-1]
                else:
                    logits = self.model(x[None])[0][-1][-1]

            prev_idx = poss_bar if len(new_idx) else vocab.pad_idx

            # Temperature
            # Use first temperatures value if last prediction was duration
            temperature = temperatures[0] if vocab.is_duration_or_pad(prev_idx) else temperatures[1]
            repeat_penalty = max(0, np.log((repeat_count+1)/4)/5) * temperature
            temperature += repeat_penalty
            if temperature != 1.: logits = logits / temperature
                

            # Filter
            # bar = 16 beats
            filter_value = -float('Inf')
            if ((last_pos - start_pos) // 16) <= min_bars: logits[vocab.bos_idx] = filter_value

            logits = filter_invalid_indexes(logits, prev_idx, vocab, filter_value=filter_value)
            logits = top_k_top_p(logits, top_k=top_k, top_p=top_p, filter_value=filter_value)
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx = torch.multinomial(probs, 1).item()

            # Update repeat count
            num_choices = len(probs.nonzero().view(-1))
            if num_choices <= 2: repeat_count += 1
            else: repeat_count = repeat_count // 2

            if prev_idx==vocab.sep_idx: 
                duration = idx - vocab.dur_range[0]
                last_pos = last_pos + duration

                bars_pred = (last_pos - start_pos) // 16
                abs_bar = last_pos // 16
                # if (bars % 8 == 0) and (bars_pred > min_bars): break
                if (i / n_words > 0.80) and (abs_bar % 4 == 0): break


            if idx==vocab.bos_idx: 
                print('Predicted BOS token. Returning prediction...')
                break

            x = x.new_tensor([idx])
            pos = pos.new_tensor([last_pos])

        pred = vocab.to_music_item(np.array(new_idx))
        full = item.append(pred)
        return pred, full

    def predict(self, item:MusicItem, n_words:int=128,
                     temperatures:float=(1.0,1.0), min_bars=4,
                     top_k=30, top_p=0.6, ref_measures=[], references=[], pitch_profile = []):
        "Return the `n_words` that come after `text`."
        self.model.reset()
        new_idx = []
        vocab = self.data.vocab
        x, pos = item.to_tensor(), item.get_pos_tensor()
        x = x[-112:]
        pos = pos[-112:]
        last_pos = pos[-1] if len(pos) else 0
        y = torch.tensor([0])

        start_pos = last_pos

        sep_count = 0
        bar_len = SAMPLE_FREQ * 4 # assuming 4/4 time
        vocab = self.data.vocab

        repeat_count = 0
        if hasattr(self.model[0], 'encode_position'):
            encode_position = self.model[0].encode_position
        else: encode_position = False
        prev_idx = -1
        for j_ in range(len(references)):
            prof = model_pitch_profile(pitch_profile.view(1,-1,12).float()).detach()[0,-1,:].numpy()

            cur_bar = None
            cur_bar_score = -10000
            print("j_ " + str(j_))
            prev_pits = []
            prev_durs = []
            prev_logits = []
            for qq in range(5):
                tot_len_so_far = 0

                print("q_ " + str(qq))
                poss_bar = []
                x_ = copy.copy(x)#[-112:]
                for i_ in range(60):
                    x_ = x_[-112:]
                    start_time = time()
                    print("i_" + str(i_))
                    with torch.no_grad():
                        if encode_position:
                            batch = { 'x': x_[None], 'pos': pos[None] }
                            logits = self.model(batch)[0][-1][-1]
                        else:
                            logits = self.model(x_[None])[0][-1][-1]
                    #if prev_logits != []:
                    #    print(torch.sum(prev_logits == logits))
                    prev_logits = logits

                    prev_prev_idx = prev_idx
                    prev_idx = poss_bar[-1] if len(poss_bar) else vocab.sep_idx

                    #print((prev_idx == vocab.sep_idx, prev_idx in range(*vocab.dur_range), prev_idx in range(*vocab.note_range)))

                    # Temperature
                    # Use first temperatures value if last prediction was duration
                    temperature = temperatures[0] if vocab.is_duration_or_pad(prev_idx) else temperatures[1]
                    repeat_penalty = max(0, np.log((repeat_count+1)/4)/5) * temperature
                    temperature += repeat_penalty
                    if temperature != 1.: logits = logits / temperature
                        

                    # Filter
                    # bar = 16 beats
                    filter_value = -float('Inf')
                    if ((last_pos - start_pos) // 16) <= min_bars: logits[vocab.bos_idx] = filter_value


                    print("pre-time is " + str(time() - start_time))
                    mid_time = time()
                    logits = filter_invalid_indexes(logits, prev_idx, vocab, filter_value=filter_value)
                    logits = filter_invalid_indices(vocab, logits, prev_idx, prev_pits, prev_durs, ref_measures[j_], references[j_], prof, tot_len_so_far)
                    print("mid time is " + str(time() - mid_time))
                    #logits = top_k_top_p(logits, top_k=top_k, top_p=top_p, filter_value=filter_value)
                    mid_time2 = time()
                    # Sample
                    probs = F.softmax(logits, dim=-1)

                    idx = torch.multinomial(probs, 1).item()

                    cat_idx = 1 if idx == vocab.sep_idx else 2 if idx in range(*vocab.dur_range) else 0 if idx in range(*vocab.note_range) else -1
                    cat_prev = 1 if prev_idx == vocab.sep_idx else 2 if prev_idx in range(*vocab.dur_range) else 0 if prev_idx in range(*vocab.note_range) else -1
                    assert(cat_idx != cat_prev)

                    # Update repeat count
                    num_choices = len(probs.nonzero().view(-1))
                    if num_choices <= 2: repeat_count += 1
                    else: repeat_count = repeat_count // 2

                    tot_len = getLength(poss_bar, vocab)
                    #print((poss_bar, tot_len))
                    if tot_len == 16:
                        #print("innnnnnnnn" + str(prev_durs))
                        poss_bar.append(vocab.sep_idx)
                        score_bar = getScore(vocab, poss_bar, references[j_], ref_measures[j_], prof)
                        if score_bar > cur_bar_score:
                            print("score greater")
                            cur_bar = poss_bar
                            cur_bar_score = score_bar
                        break
                    elif tot_len > 16:
                        #print("innnnn2 " + str(prev_durs))
                        poss_bar[-1] = 16 - getLength(poss_bar[:-1], vocab) + vocab.dur_range[0] - 1
                        #print(getLength(new_idx, vocab))
                        poss_bar.append(vocab.sep_idx)
                        score_bar = getScore(vocab, poss_bar, references[j_], ref_measures[j_], prof)
                        if score_bar > cur_bar_score:
                            print("score greater")
                            cur_bar = poss_bar
                            cur_bar_score = score_bar
                        break
                    """ 
                        duration = idx - vocab.dur_range[0]
                        last_pos = last_pos + duration

                    

                        bars_pred = (last_pos - start_pos) // 16
                        abs_bar = last_pos // 16
                        # if (bars % 8 == 0) and (bars_pred > min_bars): break
                        if (i / n_words > 0.80) and (abs_bar % 4 == 0): break
                    """




                    poss_bar.append(idx)
                    if idx in range(*vocab.note_range):
                        prev_pits.append(idx - vocab.note_range[0])
                    elif idx in range(*vocab.dur_range):
                        prev_durs.append(idx - vocab.dur_range[0] + 1)
                        tot_len_so_far += prev_durs[-1]
                    x_ = torch.cat((x_, torch.from_numpy(np.array([idx]))), 0)
                    pos = pos.new_tensor([last_pos])
                    print("post time " + str(time() - mid_time2))

            new_idx.append(cur_bar)
            x = torch.cat((x, torch.from_numpy(np.array(cur_bar))), 0)


            cur_pits = [i - vocab.note_range[0] for i in cur_bar if i in range(*vocab.note_range)]
            cur_durs = [i - vocab.dur_range[0] + 1 for i in cur_bar if i in range(*vocab.dur_range)]
            pits_durs = [(cur_pits[i], cur_durs[i]) for i in range(len(cur_durs))]
            cur_bar_pp = np.zeros(12)
            for pc in range(12):
                cur_bar_pp[pc] = sum([k[1] for k in pits_durs if k[0] % 12 == pc])
            cur_bar_pp /= np.sum(cur_bar_pp)

            pitch_profile = torch.cat((pitch_profile, torch.from_numpy(cur_bar_pp).view(1,12)), 0)

        #print([len(q) for q in new_idx])
        #print(np.array(concat(new_idx)).shape)
        pred = [vocab.to_music_item(np.array(q)) for q in new_idx]
        #full = item.append(pred)
        return pred
    
# High level prediction functions from midi file
def predict_from_midi(learn, midi=None, n_words=400, 
                      temperatures=(1.0,1.0), top_k=30, top_p=0.6, seed_len=None, **kwargs):
    vocab = learn.data.vocab
    seed = MusicItem.from_file(midi, vocab) if not is_empty_midi(midi) else MusicItem.empty(vocab)
    if seed_len is not None: seed = seed.trim_to_beat(seed_len)

    pred, full = learn.predict(seed, n_words=n_words, temperatures=temperatures, top_k=top_k, top_p=top_p, **kwargs)
    return full

def filter_invalid_indexes(res, prev_idx, vocab, filter_value=-float('Inf')):
    if vocab.is_duration_or_pad(prev_idx) or prev_idx == vocab.sep_idx:
        res[list(range(*vocab.dur_range))] = filter_value
    else:
        res[list(range(*vocab.note_range))] = filter_value
    if prev_idx == vocab.sep_idx:
        res[prev_idx] = filter_value
    return res

def getLength(xs, vocab):
    return sum([i - vocab.dur_range[0] + 1 for i in xs if i in range(*vocab.dur_range)])

def filter_invalid_indices(vocab, res, prev_idx, prev_pits, prev_durs, ref_measure, references, pitch_profile, tot_len_so_far = 0):
    #print(prev_idx in range(*vocab.dur_range))
    #print(prev_idx in range(*vocab.note_range))


    #print(prev_idx)
    if prev_idx in range(*vocab.dur_range):
        for k in range(len(res)):
            if k != vocab.sep_idx:
                res[k] = -float("Inf")
    elif prev_idx == vocab.sep_idx:
        #print(res[vocab.note_range[0]:vocab.note_range[1]])
        for k in range(len(res)):
            if k in range(*vocab.note_range):
                if res[k] != -float("Inf"):
                    #print(prev_pits)
                    if not isValidPitch(prev_pits, [i[0] for i in ref_measure], references, k - vocab.note_range[0]):
                        res[k] -= 20
                    else:
                        res[k] += 20
                res[k] += 20*pitch_profile[k % 12]             
            else:
                res[k] = -float("Inf")
    elif prev_idx in range(*vocab.note_range):
        all_inf = True

        for k in range(len(res)):
            if k in range(*vocab.dur_range):
                if res[k] != -float("Inf"):
                    if not isValidDur(prev_durs, [int(i[1]*4) for i in ref_measure], references, k - vocab.dur_range[0]):
                        res[k] =-float("Inf")#0.00001*res[k]
                    else:
                        #print("in else") 
                        all_inf = False
                        #print([int(i[1]*4) for i in ref_measure])
                        pass
                if tot_len_so_far + k - vocab.dur_range[0] > 16:
                    res[k] = -float("Inf")
            else:
                res[k] = -float("Inf")
        if all_inf:
            print("innnnnn all_inf")
            res[vocab.dur_range[0]] = 0.01
            
        #print(res[vocab.note_range[0]:vocab.note_range[1]])
    else:
        print("previdx " + str(prev_idx))
        #0.1*res[k]
    #print(res[vocab.dur_range[0]:vocab.dur_range[1]])
    """
    else:
        for i in range(len(res)):
            if i != vocab.sep_idx:
                res[i] = -float("Inf")
    """
    return res


