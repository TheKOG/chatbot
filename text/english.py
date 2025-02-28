import pickle
import os
import re
from g2p_en import G2p
from string import punctuation

from text import symbols

current_file_path = os.path.dirname(__file__)
CMU_DICT_PATH = os.path.join(current_file_path, 'cmudict.rep')
CACHE_PATH = os.path.join(current_file_path, 'cmudict_cache.pickle')
_g2p = G2p()

arpa = {'AH0', 'S', 'AH1', 'EY2', 'AE2', 'EH0', 'OW2', 'UH0', 'NG', 'B', 'G', 'AY0', 'M', 'AA0', 'F', 'AO0', 'ER2', 'UH1', 'IY1', 'AH2', 'DH', 'IY0', 'EY1', 'IH0', 'K', 'N', 'W', 'IY2', 'T', 'AA1', 'ER1', 'EH2', 'OY0', 'UH2', 'UW1', 'Z', 'AW2', 'AW1', 'V', 'UW2', 'AA2', 'ER', 'AW0', 'UW0', 'R', 'OW1', 'EH1', 'ZH', 'AE0', 'IH2', 'IH', 'Y', 'JH', 'P', 'AY1', 'EY0', 'OY2', 'TH', 'HH', 'D', 'ER0', 'CH', 'AO1', 'AE1', 'AO2', 'OY1', 'AY2', 'IH1', 'OW0', 'L', 'SH'}


def post_replace_ph(ph):
    rep_map = {
        '：': ',',
        '；': ',',
        '，': ',',
        '。': '.',
        '！': '!',
        '？': '?',
        '\n': '.',
        "·": ",",
        '、': ",",
        '...': '…',
        'v': "V"
    }
    if ph in rep_map.keys():
        ph = rep_map[ph]
    if ph in symbols:
        return ph
    if ph not in symbols:
        ph = 'UNK'
    return ph

def read_dict():
    g2p_dict = {}
    start_line = 49
    with open(CMU_DICT_PATH) as f:
        line = f.readline()
        line_index = 1
        while line:
            if line_index >= start_line:
                line = line.strip()
                word_split = line.split('  ')
                word = word_split[0]

                syllable_split = word_split[1].split(' - ')
                g2p_dict[word] = []
                for syllable in syllable_split:
                    phone_split = syllable.split(' ')
                    g2p_dict[word].append(phone_split)

            line_index = line_index + 1
            line = f.readline()

    return g2p_dict


def cache_dict(g2p_dict, file_path):
    with open(file_path, 'wb') as pickle_file:
        pickle.dump(g2p_dict, pickle_file)


def get_dict():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, 'rb') as pickle_file:
            g2p_dict = pickle.load(pickle_file)
    else:
        g2p_dict = read_dict()
        cache_dict(g2p_dict, CACHE_PATH)

    return g2p_dict

eng_dict = get_dict()

def refine_ph(phn):
    tone = 0
    if re.search(r'\d$', phn):
        tone = int(phn[-1]) + 1
        phn = phn[:-1]
    return phn.lower(), tone

def refine_syllables(syllables):
    tones = []
    phonemes = []
    for phn_list in syllables:
        for i in range(len(phn_list)):
            phn = phn_list[i]
            phn, tone = refine_ph(phn)
            phonemes.append(phn)
            tones.append(tone)
    return phonemes, tones


def text_normalize(text):
    # todo: eng text normalize
    return text

def g2p(text):

    phones = []
    tones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.upper() in eng_dict:
            phns, tns = refine_syllables(eng_dict[w.upper()])
            phones += phns
            tones += tns
        else:
            phone_list = list(filter(lambda p: p != " ", _g2p(w)))
            for ph in phone_list:
                if ph in arpa:
                    ph, tn = refine_ph(ph)
                    phones.append(ph)
                    tones.append(tn)
                else:
                    phones.append(ph)
                    tones.append(0)
    # todo: implement word2ph
    word2ph = [1 for i in phones]

    phones = [post_replace_ph(i) for i in phones]
    return phones, tones, word2ph

if __name__ == "__main__":
    # print(get_dict())
    # print(eng_word_to_phoneme("hello"))
    print(g2p("In this paper, we propose 1 DSPGAN, a GAN-based universal vocoder."))
    # all_phones = set()
    # for k, syllables in eng_dict.items():
    #     for group in syllables:
    #         for ph in group:
    #             all_phones.add(ph)
    # print(all_phones)