import torch
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from fuzzywuzzy import fuzz
import json, re, random, os
from copy import deepcopy
from tqdm import tqdm
from pytorch_pretrained_bert import BertTokenizer
flatten = lambda l: [item for sublist in l for item in sublist]
nlp = spacy.load('en_core_web_sm')

def remove_stop(input):
    # input: list of str
    output = [word for word in input if word not in STOP_WORDS]
    return output


def remove_th(input):
    pattern = re.compile(r'[0-9]+th($|\s)')
    output = [word.strip('th') if re.match(pattern, word) else word for word in input]
    return output


def get_ngram(tokens):
    ngram = []
    for i in range(1, len(tokens) + 1):
        for s in range(len(tokens) - i + 1):
            ngram.append((" ".join(tokens[s: s + i]), s, i + s))
    return ngram

def kvret2kvret_star(config):
    origin_file_set = ['kvret/train',
                       'kvret/dev',
                       'kvret/test']
    file_set = ['kvret_star/train',
                'kvret_star/dev',
                'kvret_star/test']
    def sessions_write(sessions,f_w):
        for session in sessions:
            f_w.writelines(session)
        f_w.write('\n')
    for origin_file_name,file_name in zip(origin_file_set,file_set):
        f_r = open('data/' + origin_file_name + '.iob', 'r', encoding='utf-8').readlines()
        f_w = open('data/' + file_name + '.iob', 'w', encoding='utf-8')
        sessions=[]
        current_session = []
        for line in f_r:
            if line=='\n':
                sessions.append(current_session)
                current_session = []
                if len(sessions) > 1:
                    sessions_write(sessions,f_w)
                    sessions=[]
                elif random.Random(24022019).randint(1,9) < 6:
                    continue
                else:
                    sessions_write(sessions, f_w)
                    sessions = []
            else:
                current_session.append(line)
def json2iob_kvret():

    file_set = ['train',
                'dev',
                'test']
    for file_name in file_set:
        f_r = open('data/kvret/' + file_name + '.json', 'r', encoding='utf-8')
        f_w = open('data/kvret/' + file_name + '.iob', 'w', encoding='utf-8')

        json_data = json.load(f_r)


        for dialogue in json_data:
            session_intent = dialogue['scenario']['task']['intent']
            driver = ''
            for turn in dialogue['dialogue']:
                speaker = turn['turn']
                if speaker == 'driver':
                    driver = turn['data']['utterance']
                    continue
                else:
                    if '|||' not in driver and driver!='':
                        end = turn['data']['end_dialogue']
                        intent = 'thanks' if end else session_intent
                        slots = turn['data']['slots']

                        driver_seg = [token.text for token in nlp.tokenizer(driver) if not token.is_space]
                        driver_seg_lower = [token.lower() for token in driver_seg]
                        driver_seg_lower_rs = [token.lower().strip('s') for token in driver_seg]
                        driver_seg_lower_rs_th = remove_th(driver_seg_lower_rs)
                        driver_len = len(driver_seg_lower)
                        driver_iob = ['O' for i in driver_seg_lower]
                        for slot, value in slots.items():
                            flag_find = False
                            flag_exist = False
                            value_seg_lower = [token.text.lower().strip('s') for token in
                                               nlp.tokenizer(value.strip().replace('.', ''))]
                            value_seg_lower = remove_th(value_seg_lower)
                            value_len = len(value_seg_lower)

                            # match exactly
                            for i in range(driver_len):
                                if (i + value_len <= driver_len) and (driver_seg_lower_rs_th[i:i + value_len] == value_seg_lower):
                                    driver_iob[i] = 'B-' + slot
                                    for j in range(1, value_len):
                                        driver_iob[i + j] = 'I-' + slot
                                    flag_find = True
                                    break
                                if driver_seg_lower_rs_th[i] in value_seg_lower and driver_seg_lower[i] not in STOP_WORDS:
                                    flag_exist = True

                            if flag_exist and not flag_find:
                                # remove stop word in slot_value
                                n_gram_candidate = get_ngram(driver_seg_lower_rs_th)
                                n_gram_candidate = sorted(n_gram_candidate, key=lambda x: (fuzz.token_sort_ratio(x[0], value_seg_lower),-len(x[0].split())),
                                                          reverse=True)

                                top = n_gram_candidate[0]
                                for i in range(top[1], top[2]):
                                    if i == top[1]:
                                        driver_iob[i] = 'B-' + slot
                                    else:
                                        driver_iob[i] = 'I-' + slot
                                #print('{}\t{}'.format(value,' '.join(driver_seg[top[1]:top[2]])))
                        driver = ' '.join(driver_seg) + '|||' + ' '.join(driver_iob) + '|||' + intent
                        f_w.write(driver + '\n')
                    assistant = turn['data']['utterance']
                    f_w.write(assistant + '\n')
            f_w.write('\n')
        f_w.close()
def json2iob_m2m():
    file_set = ['sim-M/dev',
                'sim-M/train',
                'sim-M/test',
                'sim-R/dev',
                'sim-R/train',
                'sim-R/test'
                ]
    for file_name in file_set:
        f_r = open('data/m2m/' + file_name + '.json', 'r', encoding='utf-8')
        f_w = open('data/m2m/' + file_name + '.iob', 'w', encoding='utf-8')

        json_data = json.load(f_r)
        for dialogue in json_data:
            user_intent = ''
            for i, turn in enumerate(dialogue['turns']):
                if i == 0:
                    user_intent = turn['user_intents'][0]
                user_act = [act['type'] for act in turn['user_acts']]
                user_tokens = turn['user_utterance']['tokens']
                slots = turn['user_utterance']['slots']
                user_iob = ['O' for token in user_tokens]
                for slot in slots:
                    start = slot['start']
                    end = slot['exclusive_end']
                    slot_name = slot['slot']
                    user_iob[start] = 'B-'+slot_name
                    for j in range(start+1,end):
                        user_iob[j] = 'I-'+slot_name
                if i != 0:
                    sys = turn['system_utterance']['text']
                    f_w.write(sys + '\n')
                user = ' '.join(user_tokens)+'|||'+' '.join(user_iob)+'|||'+user_intent+'|||'+' '.join(user_act)
                f_w.write(user + '\n')

            f_w.write('\n')
        f_w.close()

    for file_name in ['train.iob','test.iob','dev.iob']:
        os.system('cat data/m2m/sim-R/'+ file_name+' data/m2m/sim-M/'+file_name+' > data/m2m/'+file_name)

def build_vocab(path, user_only=False, bert=False):
    print('building dictionary first...')
    data = open(path,"r",encoding="utf-8").readlines()
    p_data = []
    bot = []
    for d in data:
        if d=="\n":
            bot=[]
            continue
        dd = d.replace("\n","").split("|||")
        if len(dd)==1:
            if user_only:
                pass
            else:
                bot = dd[0].split()
        else:
            user = dd[0].split()
            tag = dd[1].split()
            intent = dd[2]
            p_data.append([bot,user,tag,intent])
    bots, currents, slots, intents = list(zip(*p_data))
    vocab = list(set(flatten(currents+bots)))
    vocab_freq = {}
    for word in flatten(currents+bots):
        if vocab_freq.get(word)==None:
            vocab_freq[word]=1
        else:
            vocab_freq[word]+=1
    vocab = [v for v, f in vocab_freq.items() if f > 1]
    slot_vocab = list(set(flatten(slots)))
    intent_vocab = list(set(intents))

    for rand_vocab in [vocab,slot_vocab,intent_vocab]:
        rand_vocab.sort()
    word2index={"[PAD]" : 0, "[UNK]" : 1, "[NULL]" : 2, "<s>" : 3, "</s>" : 4}
    for vo in vocab:
        if word2index.get(vo)==None:
            word2index[vo] = len(word2index)

    slot2index={"[PAD]" : 0}
    for vo in slot_vocab:
        if slot2index.get(vo)==None:
            slot2index[vo] = len(slot2index)

    intent2index={}
    for vo in intent_vocab:
        if intent2index.get(vo)==None:
            intent2index[vo] = len(intent2index)
    if bert:
        tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')
        word2index=tokenizer.vocab
    return [word2index,slot2index,intent2index]


def prepare_dataset(path,config,built_vocab,user_only=False):
    slm = config.slm_weight>0
    data = open(path,"r",encoding="utf-8").readlines()
    p_data = []
    c_data = []
    history=[["[NULL]"]]
    for d in data:
        if d=="\n":
            if slm:
                temp = deepcopy(history)
                for i in range(1,len(history)):
                    c_data.append([temp[:i],temp[i:]])
            history=[["[NULL]"]]
            continue
        dd = d.replace("\n","").split("|||")
        if len(dd)==1:
            if user_only:
                pass
            else:
                bot = dd[0].split()
                history.append(bot)
        else:
            user = dd[0].split()
            tag = dd[1].split()
            head = [1]*len(user)
            intent = dd[2]
            temp = deepcopy(history)
            p_data.append([temp,user,tag,intent,head])
            history.append(user)

    word2index, slot2index, intent2index = built_vocab

    for t in tqdm(p_data):
        for i,history in enumerate(t[0]):
            t[0][i] = prepare_sequence(history, word2index).view(1, -1)

        t[1] = prepare_sequence(t[1], word2index).view(1, -1)
        t[2] = prepare_sequence(t[2], slot2index).view(1, -1)
        t[3] = torch.LongTensor([intent2index[t[3]]]).view(1,-1)
        t[4] = torch.LongTensor([t[4]]).view(1, -1)
    if slm:
        for t in tqdm(c_data):
            for i, history in enumerate(t[0]):
                t[0][i] = prepare_sequence(history, word2index).view(1, -1)
            for i, candidate in enumerate(t[1]):
                t[1][i] = prepare_sequence(candidate, word2index).view(1, -1)
            t.append(torch.LongTensor([1]+[0 for i in range(i-1)]).view(1, -1))
    else:
        c_data = p_data

    return p_data,c_data


def prepare_sequence(seq, to_index):
    idxs = list(map(lambda w: to_index[w] if to_index.get(w) is not None else to_index["[UNK]"], seq))
    return torch.LongTensor(idxs)
def index2seq(index, to_index):
    to_seq = {j:i for i,j in to_index.items()}
    seq = list(map(lambda w: to_seq[w], index))
    return seq

def prepare_dataset_bert(path,config,built_vocab,user_only=False):
    tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')
    slm = config.slm_weight>0
    data = open(path,"r",encoding="utf-8").readlines()
    p_data = []
    c_data = []
    history=[["[CLS]"]]
    for d in data:
        if d=="\n":
            if slm:
                temp = deepcopy(history)
                for i in range(1,len(history)):
                    c_data.append([temp[:i],temp[i:]])
            history=[["[CLS]"]]
            continue
        dd = d.replace("\n","").split("|||")
        if len(dd)==1:
            if user_only:
                pass
            else:
                #bot = dd[0].split()
                bot = ["[CLS]"]+tokenizer.tokenize(dd[0])+['[SEP]']
                history.append(bot)
        else:
            #user = dd[0].split()
            words = ["[CLS]"]+dd[0].split()+ ["[SEP]"]
            tags = ["[PAD]"] +dd[1].split()+["[PAD]"]
            user = []
            tag = []
            head = []
            for w, t in zip(words, tags):
                tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
                is_head = [1]+[0]* (len(tokens) - 1)
                t = [t] + ["[PAD]"] * (len(tokens) - 1)
                head.extend(is_head)
                user.extend(tokens)
                tag.extend(t)
            assert len(user)==len(head)==len(tag)
            intent = dd[2]
            temp = deepcopy(history)
            p_data.append([temp,user,tag,intent,head])
            history.append(user)

    word2index, slot2index, intent2index = built_vocab

    for t in tqdm(p_data):
        for i, history in enumerate(t[0]):
            # t[0][i] = prepare_sequence(history, word2index).view(1, -1)
            t[0][i] = torch.LongTensor(prepare_sequence(history,word2index)).view(1, -1)
        # t[1] = prepare_sequence(t[1], word2index).view(1, -1)
        t[1] = torch.LongTensor(prepare_sequence(t[1],word2index)).view(1, -1)
        t[2] = prepare_sequence(t[2], slot2index).view(1, -1)
        t[3] = torch.LongTensor([intent2index[t[3]]]).view(1,-1)
        t[4] = torch.LongTensor([t[4]]).view(1, -1)
    if slm:
        for t in tqdm(c_data):
            for i, history in enumerate(t[0]):
                t[0][i] = torch.LongTensor(prepare_sequence(history,word2index)).view(1, -1)
            for i, candidate in enumerate(t[1]):
                t[1][i] = torch.LongTensor(prepare_sequence(candidate,word2index)).view(1, -1)
            t.append(torch.LongTensor([1]+[0 for i in range(i-1)]).view(1, -1))
    else:
        c_data = p_data

    return p_data,c_data


def data_loader(train_data,batch_size,shuffle=False):
    if shuffle: random.Random(24022019).shuffle(train_data)
    sindex = 0
    eindex = batch_size
    while eindex < len(train_data):
        batch = train_data[sindex: eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield batch

    if eindex >= len(train_data):
        batch = train_data[sindex:]
        yield batch


def pad_to_batch_slm(batch, w_to_ix):  # for bAbI dataset
    history, candidate, label = list(zip(*batch))

    max_history = max([len(h) for h in history])
    max_len = max([h.size(1) for h in flatten(history)])
    max_candidate = max([len(h) for h in candidate])
    max_len_candidate = max([h.size(1) for h in flatten(candidate)])

    historys, candidates, labels = [], [], []
    for i in range(len(batch)):
        history_p_t = []
        for j in range(len(history[i])):
            if history[i][j].size(1) < max_len:
                history_p_t.append(torch.cat([history[i][j], torch.LongTensor(
                    [w_to_ix['[PAD]']] * (max_len - history[i][j].size(1))).view(1, -1)], 1))
            else:
                history_p_t.append(history[i][j])

        while len(history_p_t) < max_history:
            history_p_t.append(torch.LongTensor([w_to_ix['[PAD]']] * max_len).view(1, -1))

        history_p_t = torch.cat(history_p_t)
        historys.append(history_p_t)

        candidate_p_t = []
        for j in range(len(candidate[i])):
            if candidate[i][j].size(1) < max_len_candidate:
                candidate_p_t.append(torch.cat([candidate[i][j], torch.LongTensor(
                    [w_to_ix['[PAD]']] * (max_len_candidate - candidate[i][j].size(1))).view(1, -1)], 1))
            else:
                candidate_p_t.append(candidate[i][j])

        while len(candidate_p_t) < max_candidate:
            candidate_p_t.append(torch.LongTensor([w_to_ix['[PAD]']] * max_len_candidate).view(1, -1))

        candidate_p_t = torch.cat(candidate_p_t)
        candidates.append(candidate_p_t)

        if label[i].size(1) < max_candidate:
            labels.append(torch.cat(
                [label[i], torch.LongTensor([0] * (max_candidate - label[i].size(1))).view(1, -1)], 1))
        else:
            labels.append(label[i])

    labels = torch.cat(labels)

    return historys, candidates, labels


def pad_to_batch(batch, w_to_ix,s_to_ix): # for bAbI dataset
    history,current,slot,intent,head = list(zip(*batch))
    max_history = max([len(h) for h in history])
    max_len = max([h.size(1) for h in flatten(history)])
    max_current = max([c.size(1) for c in current])
    max_head = max([h.size(1) for h in head])
    max_slot = max([s.size(1) for s in slot])
    assert max_slot==max_current==max_head

    historys, currents, slots,heads = [], [], [], []
    for i in range(len(batch)):
        history_p_t = []
        for j in range(len(history[i])):
            if history[i][j].size(1) < max_len:
                history_p_t.append(torch.cat([history[i][j], torch.LongTensor([w_to_ix['[PAD]']] * (max_len - history[i][j].size(1))).view(1, -1)], 1))
            else:
                history_p_t.append(history[i][j])

        while len(history_p_t) < max_history:
            history_p_t.append(torch.LongTensor([w_to_ix['[PAD]']] * max_len).view(1, -1))

        history_p_t = torch.cat(history_p_t)
        historys.append(history_p_t)

        if current[i].size(1) < max_current:
            currents.append(torch.cat([current[i], torch.LongTensor(
                [w_to_ix['[PAD]']] * (max_current - current[i].size(1))).view(1, -1)], 1))
            heads.append(torch.cat([head[i], torch.LongTensor(
                [0] * (max_current - current[i].size(1))).view(1, -1)], 1))
            slots.append(torch.cat([slot[i], torch.LongTensor(
                [s_to_ix['[PAD]']] * (max_current - slot[i].size(1))).view(1, -1)],1))
        else:
            heads.append(head[i])
            currents.append(current[i])
            slots.append(slot[i])



    currents = torch.cat(currents)
    slots = torch.cat(slots)
    intents = torch.cat(intent)
    heads = torch.cat(heads)

    return historys, currents, slots, intents, heads


def pad_to_history(history, x_to_ix): # this is for inference

    max_x = max([len(s) for s in history])
    x_p = []
    for i in range(len(history)):
        h = prepare_sequence(history[i],x_to_ix).unsqueeze(0)
        if len(history[i]) < max_x:
            x_p.append(torch.cat([h,torch.LongTensor([x_to_ix['[PAD]']] * (max_x - h.size(1))).view(1, -1)], 1))
        else:
            x_p.append(h)

    history = torch.cat(x_p)
    return [history]
