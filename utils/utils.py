# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """


import logging
import json
import numpy as np
from copy import deepcopy
import csv
logger = logging.getLogger(__name__)

def index_of_string(document, value):
    n1 = len(document)
    n2 = len(value)
    result_list = []
    for i in range(n1 - n2 + 1):
        if document[i:i + n2] == value:
            result_list.append([i, i + n2])
    return deepcopy(result_list)

import torch
def sent_acc(output, labels):
    pred = torch.argmax(output,dim=1)
    correct = 0
    total = 0
    for p,r in zip(pred,labels):
        if p == r:
            correct += 1
        total += 1
    return correct, total

def get_slot_list(pred, gold_slots, input_ids, input_mask, idx2slot, tokenizer):
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
        tokens = tokenizer.convert_ids_to_tokens(masked_ids)
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
    return token_list,pred_list,real_list
