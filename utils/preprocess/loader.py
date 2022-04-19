from copy import deepcopy
import json
import csv
class Loader(object):
    def __init__(self):
        self.examples=[]
        self.intents=[]
        self.slot2idx=[]

    @staticmethod
    def split_intent(intent: str):
        wrong_map = {
            'airfare': 'atis_airfare',
            'airline': 'atis_airline',
            'flight': 'atis_flight',
            'flight_no': 'atis_flight_no'
        }
        if " " in intent:
            results = intent.strip().split(' ')
        elif "#" in intent:
            results = intent.strip().split('#')
        else:
            if intent in wrong_map:
                results = [wrong_map[intent]]
            else:
                return [intent]
        for i in range(len(results)):
            if results[i] in wrong_map:
                results[i] = wrong_map[results[i]]
        return results
    @staticmethod
    def read_examples_from_mATIS(data_dir, language: str, split: str, cosda_dict=None):
        with open(f"{data_dir}/intent_vocab.json", "r", encoding='utf-8') as intent_vocab:
            intent_json = json.load(intent_vocab)
            intents = intent_json["idx_to_token"]
        with open(f"{data_dir}/slot_vocab.json", "r", encoding='utf-8') as slot_vocab:
            slot_json = json.load(slot_vocab)
            slots = slot_json["idx_to_token"]
            slot2idx = slot_json["token_to_idx"]

        examples = []

        csv.register_dialect('tsv_dialect', delimiter='\t', quoting=csv.QUOTE_ALL)
        with open(f"{data_dir}/{split}_{language}.tsv", "r", encoding='utf-8') as wf:
            reader = csv.DictReader(wf, fieldnames=["id", "utterance", "slot_labels", "intent"], dialect='tsv_dialect')
            for n, row in enumerate(reader):
                if n == 0:
                    continue
                data = dict(row)
                result_union = Loader.split_sentence_tags(data["utterance"], data["slot_labels"], slots)
                if not result_union:
                    continue
                multi_intent = Loader.split_intent(data["intent"])
                words = deepcopy(result_union[0])


                examples.append(
                    InputExample(guid=deepcopy(data["id"]),
                                 words=deepcopy(words),
                                 labels=deepcopy(result_union[1]),
                                 intent=deepcopy(multi_intent[0])))
        csv.unregister_dialect('tsv_dialect')
        return examples, intents, slot2idx

    @staticmethod
    def split_sentence_tags(sentence: str, tags: str, slots: list):
        utterance_list = sentence.strip().split()
        slots_list = tags.strip().split()
        while "" in utterance_list:
            utterance_list.remove("")
        while "" in slots_list:
            slots_list.remove("")
        for slot in slots_list:
            if slot not in slots:
                print(f"Wrong slot name: {slot}")
                return None
        if len(utterance_list) == len(slots_list):
            return (utterance_list, slots_list)
        else:
            return None

class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels, intent):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels
        self.intent = intent