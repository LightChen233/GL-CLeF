"""
@Author		:           Lee, Qin
@StartTime	:           2018/08/13
@Filename	:           metric.py
@Software	:           Pycharm
@Framework  :           Pytorch
@LastModify	:           2019/05/07
"""

import numpy as np
from collections import Counter


class Evaluator(object):

    @staticmethod
    def print_bad_case(pred_slot, real_slot, pred_intent, real_intent,
                       pred_list,
                       real_list,
                       pred_intent_list, real_intent_list,
                       tokens, output_dir):
        """
        Compute the accuracy based on the whole predictions of
        given sentence, including slot and intent.
        """
        f = open(output_dir, "a", encoding="utf-8")
        for p_slot, r_slot, p_intent, r_intent, p_list, r_list, p_i, r_i, token in zip(pred_slot, real_slot,
                                                                                       pred_intent, real_intent,
                                                                                       pred_list,
                                                                                       real_list,
                                                                                       pred_intent_list,
                                                                                       real_intent_list,
                                                                                       tokens):
            tmp_p = []
            _p_list = []
            _r_list = []
            tmp_r = []
            tmp_token = []
            for p, r, t, pl, rl in zip(p_slot.numpy(), r_slot.numpy(), token, p_list, r_list):
                if r != -100:
                    tmp_p.append(p)
                    tmp_r.append(r)
                    _p_list.append(pl)
                    _r_list.append(rl)
                    tmp_token.append(t)
                else:
                    if len(tmp_token) != 0:
                        tmp_token[-1] += t

            # print(p_slot,r_slot)
            if np.all(tmp_p == tmp_r) and p_intent == r_intent:
                pass
            else:
                f.write(p_i + "\t" + r_i + "\n")
                for p, r, t in zip(p_list, r_list, token):
                    f.write("%s\t%s\t%s\n" % (t, p, r))
                f.write("\n")

    @staticmethod
    def semantic_acc(pred_slot, real_slot, pred_intent, real_intent):
        """
        Compute the accuracy based on the whole predictions of
        given sentence, including slot and intent.
        """
        total_count, correct_count = 0.0, 0.0
        for p_slot, r_slot, p_intent, r_intent in zip(pred_slot, real_slot, pred_intent, real_intent):
            tmp_p = []
            tmp_r = []
            for p, r in zip(p_slot.numpy(), r_slot.numpy()):
                if r != -100:
                    tmp_p.append(p)
                    tmp_r.append(r)
            # print(p_slot,r_slot)
            if np.all(tmp_p == tmp_r) and p_intent == r_intent:
                correct_count += 1.0
            total_count += 1.0

        return correct_count, total_count

    @staticmethod
    def accuracy(pred_list, real_list):
        """
        Get accuracy measured by predictions and ground-trues.
        """

        pred_array = np.array(list(Evaluator.expand_list(pred_list)))
        real_array = np.array(list(Evaluator.expand_list(real_list)))
        return (pred_array == real_array).sum() * 1.0 / len(pred_array)

    @staticmethod
    def f1_score(pred_list, real_list):
        """
        Get F1 score measured by predictions and ground-trues.
        """

        tp, fp, fn = 0.0, 0.0, 0.0
        for i in range(len(pred_list)):
            seg = set()
            result = [elem.strip() for elem in pred_list[i]]
            target = [elem.strip() for elem in real_list[i]]

            j = 0
            while j < len(target):
                cur = target[j]
                if cur[0] == 'B':
                    k = j + 1
                    while k < len(target):
                        str_ = target[k]
                        if not (str_[0] == 'I' and cur[1:] == str_[1:]):
                            break
                        k = k + 1
                    seg.add((cur, j, k - 1))
                    j = k - 1
                j = j + 1

            tp_ = 0
            j = 0
            while j < len(result):
                cur = result[j]
                if cur[0] == 'B':
                    k = j + 1
                    while k < len(result):
                        str_ = result[k]
                        if not (str_[0] == 'I' and cur[1:] == str_[1:]):
                            break
                        k = k + 1
                    if (cur, j, k - 1) in seg:
                        tp_ += 1
                    else:
                        fp += 1
                    j = k - 1
                j = j + 1

            fn += len(seg) - tp_
            tp += tp_

        p = tp / (tp + fp) if tp + fp != 0 else 0
        r = tp / (tp + fn) if tp + fn != 0 else 0
        return 2 * p * r / (p + r) if p + r != 0 else 0

    """
    Max frequency prediction. 
    """

    @staticmethod
    def max_freq_predict(sample):
        predict = []
        for items in sample:
            predict.append(Counter(items).most_common(1)[0][0])
        return predict

    @staticmethod
    def exp_decay_predict(sample, decay_rate=0.8):
        predict = []
        for items in sample:
            item_dict = {}
            curr_weight = 1.0
            for item in items[::-1]:
                item_dict[item] = item_dict.get(item, 0) + curr_weight
                curr_weight *= decay_rate
            predict.append(sorted(item_dict.items(), key=lambda x_: x_[1])[-1][0])
        return predict

    @staticmethod
    def expand_list(nested_list):
        for item in nested_list:
            if isinstance(item, (list, tuple)):
                for sub_item in Evaluator.expand_list(item):
                    yield sub_item
            else:
                yield item

    @staticmethod
    def nested_list(items, seq_lens):
        num_items = len(items)
        trans_items = [[] for _ in range(0, num_items)]

        count = 0
        for jdx in range(0, len(seq_lens)):
            for idx in range(0, num_items):
                trans_items[idx].append(items[idx][count:count + seq_lens[jdx]])
            count += seq_lens[jdx]

        return trans_items
