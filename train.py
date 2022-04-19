import os
import random

import fitlog
import numpy as np

from tqdm import tqdm
from transformers import BertTokenizer
from transformers import AdamW
from copy import deepcopy
from argparse import ArgumentParser

from utils.preprocess.Sampler import Sampler
from metric import Evaluator
import torch

from utils.tool import Batch
from models.contrastive_learning import ContrastiveLearning as CLModel

from utils.utils import sent_acc


class Classifier(object):

    def __init__(self, args):
        self.args = args

        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)

        self.saved_model_name = f"{self.args.lang}_{self.args.saved_model_name}_batch{self.args.batch_size}_lr{self.args.lr}_seqlen{self.args.max_seq_length}_seed{self.args.seed} "
        self.saved_encoder_name = f"en_{self.args.saved_model_name}_batch{self.args.batch_size}_lr{self.args.lr}_seqlen{self.args.max_seq_length}_seed{self.args.seed}"

        cosda_lang=[]
        if self.args.lang == "en":
            cosda_lang = ["de", "es", "fr", "hi", "ja", "pt", "tr", "zh"]
        elif self.args.lang == "zh":
            cosda_lang = ["hi", "ja", "tr", "zh"]
        tokenizer=BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
        self.train_sampler = Sampler(args, cosda_lang,tokenizer)
        self.dev_sampler = Sampler(args, cosda_lang,tokenizer)
        self.test_sampler = Sampler(args, cosda_lang,tokenizer)
        self.dev_languages = ["EN", "DE", "ES", "FR", "HI", "JA", "PT", "TR", "ZH"]
        self.test_languages = ["EN", "DE", "ES", "FR", "HI", "JA", "PT", "TR", "ZH"]

        self.train_sampler.load_data(self.args.data_dir, "EN", "train",shuffle=True)
        self.train_features = self.train_sampler.convert_examples_to_features()

        self.train_steps = self.args.epochs * len(self.train_features)
        self.model = CLModel(args=self.args,
                             num_slots=len(self.train_sampler.slot2idx),
                             num_intents=len(self.train_sampler.intents)
                             )

        self.dev_features_dic = {}
        self.test_features_dic = {}
        for lang in self.test_languages:
            self.dev_sampler.load_data(self.args.data_dir, lang, "dev")
            self.test_sampler.load_data(self.args.data_dir, lang, "test")
            print(
                f"load {len(self.dev_sampler)} {lang} dev examples, load {len(self.test_sampler)} {lang} test examples")
            dev_feature = self.dev_sampler.convert_examples_to_features()
            test_feature = self.test_sampler.convert_examples_to_features()
            self.dev_features_dic[lang] = dev_feature
            self.test_features_dic[lang] = test_feature

        self.idx2slot = {v: k for k, v in self.train_sampler.slot2idx.items()}



    def evaluate(self, lang, split="dev", output=False):
        self.model.eval()
        ev = Evaluator()
        intent_total, intent_correct = 0.0, 0.0
        slot_acc_total, slot_acc_right = 0.0, 0.0
        all_tokens = []
        all_pred_slots = []
        all_real_slots = []
        num = 0
        if output:
            if not os.path.exists("./result/"):
                os.makedirs("./result/")
        with torch.no_grad():
            if split=="dev":
                eval_data=Batch.to_list(self.dev_features_dic[lang], self.args.batch_size)
            elif split=="test":
                eval_data = Batch.to_list(self.test_features_dic[lang], self.args.batch_size)
            iterator = tqdm(eval_data)
            for batch in iterator:
                num += 1
                intent_list = []
                for data in batch:
                    intent_list.append(data['intent_id'])
                intent_id = torch.tensor(intent_list, dtype=torch.long)
                input_ids, attention_mask, token_type_ids, slot_labels = self.model.get_ids(batch, 'original')
                out = self.model.forward(batch,evaluate=True)
                intent_output=out.intent_logits
                slot_output=out.slot_logits
                intent_output = intent_output.cpu()
                slot_output = slot_output.cpu()
                label_ids = slot_labels.cpu()
                input_ids = input_ids.cpu()
                input_mask = attention_mask.cpu()
                correct, total = sent_acc(intent_output, intent_id)
                intent_correct += correct
                intent_total += total
                correct_count, total_count = ev.semantic_acc(
                    torch.argmax(slot_output, dim=2),
                    label_ids,
                    torch.argmax(intent_output, dim=1),
                    intent_id)
                slot_acc_total += total_count
                slot_acc_right += correct_count

                token_list, pred_list, real_list = self.test_sampler.get_slot_list(
                    torch.argmax(slot_output, dim=2),
                    label_ids, input_ids, input_mask, self.idx2slot)

                if output:
                    pred_intent_list = [self.train_sampler.intents[int(i_batch)] for i_batch in
                                        torch.argmax(intent_output, dim=1)]
                    real_intent_list = [self.train_sampler.intents[int(i_batch)] for i_batch in intent_id]
                    ev.print_bad_case(
                        torch.argmax(slot_output, dim=2),
                        label_ids,
                        torch.argmax(intent_output, dim=1),
                        intent_id,
                        pred_list,
                        real_list,
                        pred_intent_list,
                        real_intent_list,
                        token_list,
                        self.args.output_dir)

                all_tokens.extend(token_list)
                all_pred_slots.extend(pred_list)
                all_real_slots.extend(real_list)
                del intent_output, slot_output
        assert len(all_pred_slots) == len(all_real_slots) and len(all_pred_slots) == len(all_tokens)
        f1 = ev.f1_score(all_pred_slots,all_real_slots)
        accuracy = ev.accuracy(all_pred_slots, all_real_slots)
        overall_acc = slot_acc_right / slot_acc_total
        return (intent_correct / intent_total), f1, accuracy, overall_acc

    def test(self, state_dict=None):
        print("====== start test ======")
        if state_dict is None:
            if self.args.load_model_path is not None:
                checkpoint_dir = self.args.load_model_path
                model_CKPT = torch.load(checkpoint_dir)
            else:
                model_CKPT = torch.load(self.args.saved_model_dir + self.saved_model_name)
            self.model.load_state_dict(model_CKPT['state_dict'], False)
        else:
            self.model.load_state_dict(state_dict)
        if self.args.load_weights and not self.args.train:
            print(f"predict from {self.args.load_model_path}")
        else:
            print(f"predict from {self.saved_model_name}")
        self.record_by_step(step=self.args.epochs,split="test",output=True)
        fitlog.finish()

    def set_optimizer(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta', 'LayerNorm.bias', 'LayerNorm.weight']
        params = {'in_pretrained': {'decay': [], 'no_decay': []}, 'out_pretrained': {'decay': [], 'no_decay': []}}
        encoders = ["roberta", "bert", "encoder"]
        for n, p in param_optimizer:
            is_in_pretrained = 'in_pretrained' if any(enc in n for enc in encoders) else 'out_pretrained'
            is_no_decay = 'no_decay' if any(nd in n for nd in no_decay) else 'decay'
            params[is_in_pretrained][is_no_decay].append(p)

        lr = self.args.lr
        weight_decay = 0.0

        grouped_parameters = [
            {'params': params['in_pretrained']['decay'], 'weight_decay': weight_decay, 'lr': lr},
            {'params': params['in_pretrained']['no_decay'], 'weight_decay': 0.0, 'lr': lr},
            {'params': params['out_pretrained']['decay'], 'weight_decay': weight_decay, 'lr': lr},
            {'params': params['out_pretrained']['no_decay'], 'weight_decay': weight_decay, 'lr': lr},
        ]
        return AdamW(grouped_parameters,
                          lr=self.args.lr,
                          correct_bias=False)

    def run_batches(self):
        optimizer=self.set_optimizer()
        all_loss = 0
        all_size = 0
        iteration = 0
        iterator = tqdm(Batch.to_list(self.train_features, self.args.batch_size))
        for step, batch in enumerate(iterator):
            optimizer.zero_grad()
            self.model.train()
            out = self.model.forward(batch)
            loss =out.loss
            loss.backward()
            optimizer.step()
            all_loss += loss.item()
            iteration += 1
            all_size += len(batch)
        return all_loss / all_size, iteration


    def train(self):
        checkpoint_dir = self.args.saved_model_dir + self.saved_model_name
        min_loss_unchanged_num = 0
        best_overall = -1.0
        best_state_dict = None

        for i in range(self.args.epochs):
            loss,iteration=self.run_batches()
            fitlog.add_loss(loss, name="Loss", step=i)
            # Dev
            logs = self.record_by_step(i, split="dev", output=False)
            overall_acc=logs['overall acc']
            if overall_acc > best_overall:
                logs=self.record_by_step(step=i, split="test", output=False)
                best_overall = overall_acc
                fitlog.add_best_metric({"test": logs})
                min_loss_unchanged_num = 0
                best_state_dict = deepcopy(self.model.state_dict())
                torch.save({'state_dict': best_state_dict}, checkpoint_dir)
            else:
                min_loss_unchanged_num += 1
                if min_loss_unchanged_num >= self.args.patience:
                    print("best overall acc:", best_overall)
                    torch.save({'state_dict': best_state_dict}, checkpoint_dir)
                    return best_state_dict
        if best_state_dict is None:
            best_state_dict = deepcopy(self.model.state_dict())
        torch.save({'state_dict': best_state_dict}, checkpoint_dir)
        return best_state_dict

    def record_by_step(self,step,split="dev",output=False):
        i_accs = []
        s_f1s = []
        s_accs = []
        o_accs = []
        for key_lang in self.dev_languages:
            intent_acc, slot_f1, slot_acc, overall_acc = self.evaluate(key_lang, split=split, output=output)
            fitlog.add_metric({"{}_{}".format(split,key_lang): {"intent_acc": intent_acc, "slot_f1": slot_f1,
                                                           "slot_acc": slot_acc, "overall_acc": overall_acc}}, step=step)
            i_accs.append(intent_acc)
            s_f1s.append(slot_f1)
            s_accs.append(slot_acc)
            o_accs.append(overall_acc)
        intent_acc = sum(i_accs) / len(i_accs)
        slot_f1 = sum(s_f1s) / len(s_f1s)
        slot_acc = sum(s_accs) / len(s_accs)
        overall_acc = sum(o_accs) / len(o_accs)
        logs = {
            "intent acc": intent_acc,
            "slot f1": slot_f1,
            "slot acc": slot_acc,
            "overall acc": overall_acc
        }
        fitlog.add_metric({"{}_avg".format(split): logs}, step=step)
        return logs

if __name__ == "__main__":
    fitlog.set_log_dir('./logs/')

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            return False

    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=11111)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--negative_num', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--dropout_prob', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=20)

    parser.add_argument('--output_dir', type=str, default="./out/out.log")
    parser.add_argument('--saved_model_dir', type=str, default="saved_model/")
    parser.add_argument('--data_dir', type=str, default="./MultiATIS++/data")

    parser.add_argument('--load_model_path', type=str, default="")
    parser.add_argument('--load_weights', type=str2bool, default=False)

    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--cosda_rate', type=float, default=0.55)
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--sample_seed', type=int, default=128)
    parser.add_argument('--saved_model_name', type=str, default='slu_')
    parser.add_argument('--lang', type=str, default="en")
    parser.add_argument('--word_dict_dir', type=str, default="./MUSE_dict/")

    parser.add_argument('--temperature', type=float, default=2)
    parser.add_argument('--lambda1', type=float, default=0.01)
    parser.add_argument('--lambda2', type=float, default=0.005)
    parser.add_argument('--lambda3', type=float, default=0.01)
    parser.add_argument('--gpu', type=str2bool, default=True)

    args = parser.parse_args()
    fitlog.add_hyper(args)
    print(f"args: {args}")
    cls = Classifier(args)
    if args.train:
        best_state_dict = cls.train()
    else:
        best_state_dict=None
    cls.test(best_state_dict)
