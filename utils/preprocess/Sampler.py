import random
from copy import deepcopy

import numpy as np
import torch
from utils.preprocess.loader import Loader

class Sampler(object):

    def __init__(self, args, langs: list,tokenizer):

        self.args=args
        self.tokenizer = tokenizer
        if self.args.word_dict_dir is not None:
            self.__word_dict,self.__languages=self.__get_word_dict(self.args.word_dict_dir,langs)
        self.__data=[]
        self.__index= 0
        self.__cosda_rate=args.cosda_rate
        self.max_seq_length = self.args.max_seq_length


    def __next__(self):
        sentence = self.__data[self.__index]
        self.__index += 1
        return sentence

    def get_dict(self):
        return deepcopy(self.__word_dict)

    def load_data(self,data_dir, language: str, split: str, shuffle=False):
        self.examples, self.intents, self.slot2idx =Loader.read_examples_from_mATIS(data_dir,language,split,self.__cosda_rate)
        if shuffle:
            self.examples=list(set(self.examples))

    def __len__(self):
        return len(self.examples)


    def __random_sample(self):
        return self.examples[random.randint(0, len(self.examples)-1)]

    # get positive sample
    def positive_sample(self,tokens):
        assert (self.__word_dict is not None) and (self.__languages is not None) and (self.__cosda_rate is not None)
        return self.convert_token_with_CoSDA(
            tokens=tokens,
            word_dicts=self.__word_dict,
            langs=self.__languages,
            random_rate=self.__cosda_rate)

    def __get_word_dict(self,word_dict_dir,langs: list):
        word_dicts = {lang: {} for lang in langs}
        for lang in langs:
            with open(word_dict_dir + lang + ".txt", "r",encoding='utf-8') as f:
                for line in f.readlines():
                    if not line:
                        continue
                    source, target = line.split()
                    if source.strip() == target.strip():
                        continue
                    if source not in word_dicts[lang]:
                        word_dicts[lang][source] = [target]
                    else:
                        word_dicts[lang][source].append(target)
        return word_dicts, langs


    def convert_examples_to_features(self):
        features = []
        for (ex_index, example) in enumerate(self.examples):
            intent_id = self.intents.index(example.intent)
            original=self.convert_tokens_to_ids_with_padding(example.words,example.labels)
            pos=self.convert_tokens_to_ids_with_padding(self.positive_sample(example.words),example.labels)
            features.append({'original':original,'positive':pos,'intent_id':intent_id})
        return features

    def get_slot_list(self,pred, gold_slots, input_ids, input_mask, idx2slot):
        masked_t_or_f = []
        for batch in input_mask:
            masked_t_or_f.append([])
            for m in batch:
                if m == 0:
                    masked_t_or_f[-1].append(False)
                else:
                    masked_t_or_f[-1].append(True)
        pred_list = []
        real_list = []
        token_list = []
        for i, ((pb, lb), mb) in enumerate(zip(zip(pred, gold_slots), masked_t_or_f)):
            masked_pred = pb[torch.from_numpy(np.array(mb))]
            masked_labels = lb[torch.from_numpy(np.array(mb))]
            masked_ids = input_ids[i][torch.from_numpy(np.array(mb))]
            tokens = self.tokenizer.convert_ids_to_tokens(masked_ids)
            this_pred = []
            this_real = []
            this_tokens = []
            for t, p, l in zip(tokens, masked_pred, masked_labels):
                if int(l) == -100:
                    if len(this_tokens) > 0 and t != "<unk>":
                        this_tokens[-1] += t
                    continue
                l = idx2slot[int(l)]
                p = idx2slot[int(p)]
                this_pred.append(p)
                this_real.append(l)
                this_tokens.append(t)
            pred_list.append(deepcopy(this_pred))
            real_list.append(deepcopy(this_real))
            token_list.append(deepcopy(this_tokens))
        return token_list, pred_list, real_list

    def convert_tokens_to_ids_with_padding(self,
            words,
            labels
    ):
        pad_label_id = -100
        label_ids = []
        input_ids =[]
        input_ids.extend(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize('[CLS]')))
        label_ids = [pad_label_id] + label_ids
        for word, label in zip(words, labels):
            word_tokens = self.tokenizer.tokenize(word)
            label_ids.extend([self.slot2idx[label]] + [pad_label_id] * (len(word_tokens) - 1))
            input_ids.extend(self.tokenizer.convert_tokens_to_ids(word_tokens))
        special_tokens_count = 2
        if len(input_ids) > self.max_seq_length - special_tokens_count:
            input_ids = input_ids[: (self.max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (self.max_seq_length - special_tokens_count)]
        pad_token_id = self.tokenizer.pad_token_id
        pad_token_segment_id = self.tokenizer.pad_token_type_id
        segment_ids = [0] * (len(input_ids))
        input_mask = [1] * len(input_ids)
        padding_length = self.max_seq_length - len(input_ids)
        input_ids += [pad_token_id] * padding_length
        input_mask += [0] * padding_length
        segment_ids += [pad_token_segment_id] * padding_length

        label_ids += [pad_label_id] * padding_length
        assert len(input_ids) == self.max_seq_length and len(input_mask) == self.max_seq_length and len(
            segment_ids) == self.max_seq_length and len(label_ids) == self.max_seq_length
        return {'input_ids':input_ids, 'attention_mask':input_mask, 'token_type_ids':segment_ids,'label_ids': label_ids}

    def get_word_dict(self,langs: list):
        word_dict_dir = "../../MUSE_dict/"
        word_dicts = {lang: {} for lang in langs}
        for lang in langs:
            with open(word_dict_dir + lang + ".txt", "r") as f:
                for line in f.readlines():
                    if not line:
                        # print("gg")
                        continue
                    source, target = line.split()
                    if source.strip() == target.strip():
                        continue
                    if source not in word_dicts[lang]:
                        word_dicts[lang][source] = [target]
                    else:
                        word_dicts[lang][source].append(target)
        return word_dicts, langs

    def convert_4_1_token(self,token, word_dicts, langs):
        this_lang = random.choice(langs)
        raw_token = token.replace("▁", "")
        time = 0
        while time < 10 and raw_token not in word_dicts[this_lang]:
            this_lang = random.choice(langs)
            time += 1

        if raw_token in word_dicts[this_lang]:
            myrandom = random.Random(self.args.sample_seed)
            token=myrandom.choice(word_dicts[this_lang][raw_token])
            return token
        else:
            return token

    # code switch
    def convert_token_with_CoSDA(self,tokens, word_dicts, langs, random_rate=0.2):
        result = []
        for token in tokens:
            raw_token = token.replace("▁", "")
            if random.random() <= random_rate:
                result.append(self.convert_4_1_token(raw_token, word_dicts, langs))
            else:
                result.append(token)
        return deepcopy(result)

