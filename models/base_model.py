import torch
import torch.nn as nn
from transformers import BertModel, BertConfig


class BaseModel(torch.nn.Module):
    def __init__(self, args, num_intents, num_slots):
        super().__init__()
        self.args = args
        self.num_intents = num_intents
        self.num_slots = num_slots
        self.device = torch.device('cuda') if args.gpu else torch.device('cpu')
        self.slot_pad_idx = -100
        self.model = BertModel.from_pretrained('bert-base-multilingual-uncased')
        self.config = BertConfig.from_pretrained('bert-base-multilingual-uncased')
        self.model.to(self.device)
        if self.args.gpu:
            self.intent_classifier = nn.Linear(self.config.hidden_size, num_intents).cuda()
            self.slot_classifier = nn.Linear(self.config.hidden_size, num_slots).cuda()
            self.intent_criterion = nn.CrossEntropyLoss().cuda()
            self.slot_criterion = nn.CrossEntropyLoss(ignore_index=self.slot_pad_idx).cuda()
            self.intent_layer = nn.Linear(num_intents, self.config.hidden_size).cuda()
        else:
            self.intent_classifier = nn.Linear(self.config.hidden_size, num_intents)
            self.slot_classifier = nn.Linear(self.config.hidden_size, num_slots)
            self.intent_criterion = nn.CrossEntropyLoss()
            self.slot_criterion = nn.CrossEntropyLoss(ignore_index=self.slot_pad_idx)
            self.intent_layer = nn.Linear(num_intents, self.config.hidden_size)

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
        token_type_ids = torch.tensor(token_type_ids_list, dtype=torch.long).to(self.device)
        slot_labels = torch.tensor(slot_labels_list, dtype=torch.long).to(self.device)
        return input_ids, attention_mask, token_type_ids, slot_labels

    def get_base_loss(self, batch, datatype, intent_label_ids):
        embedded, cls, attention_mask, slot_labels_ids = self.embedding(batch, datatype)
        intent_logits = self.intent_classifier(cls)
        slot_logits = self.slot_classifier(embedded)

        total_loss = 0
        # 1. Intent Softmax
        if intent_label_ids is not None:
            if self.num_intents == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1))
            else:
                intent_loss_fct = nn.CrossEntropyLoss(ignore_index=self.slot_pad_idx)
                intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intents), intent_label_ids.view(-1))
            total_loss += intent_loss

        # 2. Slot Softmax
        if slot_labels_ids is not None:
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = slot_logits.view(-1, self.num_slots)[active_loss]
                active_labels = slot_labels_ids.view(-1)[active_loss]  # active_length
                slot_loss = self.slot_criterion(active_logits, active_labels)
            else:
                slot_loss = self.slot_criterion(slot_logits.view(-1, self.num_slots), slot_labels_ids.view(-1))
            total_loss += slot_loss
        return embedded, cls, intent_logits, slot_logits, total_loss

    def embedding(self, batch, datatype):
        input_ids, attention_mask, token_type_ids, slot_labels = self.get_ids(batch, datatype)
        out = self.model.forward(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return out.last_hidden_state, out.pooler_output, attention_mask, slot_labels

    def forward(self, batch, datatype=None, evaluate=False):
        intent_list = []
        for data in batch:
            intent_list.append(data['intent_id'])
        intent_label_ids = torch.tensor(intent_list, dtype=torch.long).to(self.device)
        if datatype is not None:
            embedded, cls, _, _ = self.embedding(batch, datatype)
            return OUT(embedded, cls, None, None, None)
        if not evaluate:
            pos_embedded, pos_cls, intent_logits, slot_logits, total_loss = self.get_base_loss(batch, 'positive',
                                                                                               intent_label_ids)
            return OUT(pos_embedded, pos_cls, intent_logits, slot_logits, total_loss)
        else:
            origin_embedded, origin_cls, intent_logits, slot_logits, total_loss = self.get_base_loss(batch, 'original',
                                                                                                     intent_label_ids)
            return OUT(origin_embedded, origin_cls, intent_logits, slot_logits, total_loss)


class OUT:
    def __init__(self, embedded, cls, intent_logits, slot_logits, total_loss):
        self.embedded = embedded
        self.cls = cls
        self.intent_logits = intent_logits
        self.slot_logits = slot_logits
        self.loss = total_loss
