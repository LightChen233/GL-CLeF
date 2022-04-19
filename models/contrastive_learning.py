import torch
import torch.nn as nn
from transformers import BertConfig
from utils.myQueue import Queue

from models.base_model import BaseModel, OUT


class ContrastiveLearning(torch.nn.Module):
    def __init__(self, args, num_intents, num_slots):
        super().__init__()
        self.args = args
        self.num_intents = num_intents
        self.num_slots = num_slots
        self.device = torch.device('cuda') if args.gpu else torch.device('cpu')
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        self.lambda3 = args.lambda3
        self.temperature = args.temperature
        self.negative_num = args.negative_num
        self.slot_pad_idx = -100
        self.batch_size = args.batch_size

        self.max_seq_length = args.max_seq_length
        self.model = BaseModel(args, num_intents, num_slots)
        if self.args.load_weights and self.args.train:
            self.load_model()
        self.model.to(self.device)
        self.config = BertConfig.from_pretrained('bert-base-multilingual-uncased')
        self.hidden_size = self.config.hidden_size
        self.cls_queue = Queue(self.hidden_size, 1, maxsize=args.negative_num, batch_size=self.batch_size)
        self.embed_queue = Queue(self.hidden_size, self.max_seq_length, maxsize=args.negative_num,
                                 batch_size=self.batch_size)

    def load_model(self):
        checkpoint_dir = self.args.load_model_path
        if self.args.load_weights:
            model_CKPT = torch.load(checkpoint_dir)
            self.model.load_state_dict(model_CKPT['state_dict'], False)

    def forward(self, batch, evaluate=False):
        # base model
        out = self.model.forward(batch, evaluate=evaluate)

        embedded = out.embedded
        cls = out.cls
        intent_logits = out.intent_logits
        slot_logits = out.slot_logits
        total_loss = out.loss
        if not evaluate:
            # contrastive learning
            out = self.model.forward(batch, 'original')
            origin_embedded = out.embedded
            origin_cls = out.cls

            out = self.model.forward(batch, 'positive')
            pos_embedded = out.embedded
            pos_cls = out.cls

            if self.cls_queue.size > 0:
                negative_embedded = self.embed_queue.negative_encode(len(batch))
                negative_cls = self.cls_queue.negative_encode(len(batch))
                local_intent_loss = self.contrastive_local_intent_loss(origin_cls, pos_cls, negative_cls)
                local_slot_loss = self.contrastive_local_slot_loss(origin_embedded, pos_embedded, negative_embedded)
                global_loss = self.contrastive_global_loss(origin_cls, origin_embedded, negative_embedded)
                global_loss += self.contrastive_global_loss(origin_cls, pos_embedded, negative_embedded)
                total_loss += self.lambda1 * local_intent_loss + self.lambda2 * local_slot_loss + self.lambda3 * global_loss / 2

            self.cls_queue.enqueue_batch_tensor(origin_cls.detach())
            self.cls_queue.enqueue_batch_tensor(pos_cls.detach())
            self.embed_queue.enqueue_batch_tensor(origin_embedded.detach())
            self.embed_queue.enqueue_batch_tensor(pos_embedded.detach())
        return OUT(embedded, cls, intent_logits, slot_logits, total_loss)

    def get_ids(self, batch, datatype):
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []
        slot_labels_list = []
        for data in batch:
            input_ids_list.append(data[datatype]['input_ids'])
            attention_mask_list.append(data[datatype]['attention_mask'])
            token_type_ids_list.append(data[datatype]['token_type_ids'])
            slot_labels_list.append(data[datatype]['label_ids'])
        input_ids = torch.tensor(input_ids_list, dtype=torch.long).to(self.device)
        attention_mask = torch.tensor(attention_mask_list, dtype=torch.long).to(self.device)
        token_type_ids = torch.tensor(attention_mask_list, dtype=torch.long).to(self.device)
        slot_labels = torch.tensor(slot_labels_list, dtype=torch.long).to(self.device)
        return input_ids, attention_mask, token_type_ids, slot_labels

    def contrastive_local_intent_loss(self, origin, positive, negative):
        # batch_size
        N = origin.shape[0]
        # feature_size
        C = origin.shape[1]
        # negative_size
        K = self.cls_queue.size
        l_pos = torch.bmm(origin.unsqueeze(1), positive.unsqueeze(2)).reshape(N, 1)
        l_neg = torch.bmm(origin.unsqueeze(1), negative.squeeze(2)).reshape(N, K)

        logits = torch.cat((l_pos, l_neg), dim=1)
        labels = torch.zeros(N, dtype=torch.long).to(self.device)
        if self.args.gpu:
            criteria = nn.CrossEntropyLoss().cuda()
        else:
            criteria = nn.CrossEntropyLoss()

        loss = criteria(torch.div(logits, self.temperature), labels)
        return loss

    def contrastive_global_loss(self, origin, positive, negative):
        # batch_size
        N = origin.shape[0]
        # feature_size
        C = origin.shape[1]
        # sequence_length
        L = positive.shape[1]
        # negative_size
        K = self.cls_queue.size
        l_pos = torch.bmm(origin.unsqueeze(1), positive.permute(0, 2, 1)).reshape(N * L, 1)
        l_neg = torch.bmm(origin.unsqueeze(1), negative.reshape(N, C, L * K)).reshape(N * L, K)

        logits = torch.cat((l_pos, l_neg), dim=1)
        labels = torch.zeros(N * L, dtype=torch.long).to(self.device)
        if self.args.gpu:
            criteria = nn.CrossEntropyLoss().cuda()
        else:
            criteria = nn.CrossEntropyLoss()
        loss = criteria(torch.div(logits, self.temperature), labels)
        return loss/128

    def contrastive_local_slot_loss(self, origin, positive, negative):
        # batch_size
        N = origin.shape[0]
        # feature_size
        C = origin.shape[2]
        # sequence_length
        L = origin.shape[1]
        # negative_size
        K = self.cls_queue.size
        l_pos = torch.bmm(origin, positive.permute(0, 2, 1)).reshape(N * L * L, 1)
        l_neg = torch.bmm(origin, negative.reshape(N, C, L * K)).reshape(N * L * L, K)

        logits = torch.cat((l_pos, l_neg), dim=1)
        labels = torch.zeros(N * L * L, dtype=torch.long).to(self.device)
        if self.args.gpu:
            criteria = nn.CrossEntropyLoss().cuda()
        else:
            criteria = nn.CrossEntropyLoss()
        loss = criteria(torch.div(logits, self.temperature), labels)
        return loss/(128*128)