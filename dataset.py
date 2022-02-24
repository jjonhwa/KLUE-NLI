from transformers.utils.data import Dataset

import torch

def label_to_num(label):
    label_dict = {"entailment": 0, "contradiction": 1, "neutral": 2, "answer": 3}
    num_label = []

    for v in label:
        num_label.append(label_dict[v])
    
    return num_label

def num_to_label(label):
    label_dict = {0: "entailment", 1: "contradiction", 2: "neutral"}
    str_label = []

    for i, v in enumerate(label):
        str_label.append([i,label_dict[v]])
    
    return str_label
    
class NLI_Dataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length, is_training: bool = True):
        self.dataset = dataset # pandas.DataFrame dataset
        self.premise = self.dataset['premise']
        self.hypothesis = self.dataset['hypothesis']

        self.label = label_to_num(self.dataset['label'].values)
        self.label = torch.tensor(self.label)

        self.max_length = max_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        premise = self.premise[idx]
        hypothesis = self.hypothesis[idx]
        label = self.label[idx]

        concat_sentence = premise + "[SEP]" + hypothesis

        encoded_dict = self.tokenizer(
            concat_sentence,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
            return_token_type_ids=False
        )

        encoded_dict['input_ids'] = encoded_dict['input_ids'].squeeze(0)
        encoded_dict['attention_mask'] = encoded_dict['attention_mask'].squeeze(0)

        encoded_dict['label'] = label
        return encoded_dict

class NLI_Explain_Dataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length, is_training: bool = True):
        self.dataset = dataset # pandas.DataFrame dataset
        self.premise = self.dataset['premise']
        self.hypothesis = self.dataset['hypothesis']
        if is_training:
            self.train_label = label_to_num(self.dataset['label'].values)
        if not is_training:
            self.train_label = self.dataset['label'].values
        self.label = torch.tensor(self.train_label)

        self.max_length = max_length
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        premise = self.premise[idx]
        hypothesis = self.hypothesis[idx]
        label = self.label[idx]

        if premise.endswith("."):
            premise = premise[:-1]
        if hypothesis.endswith("."):
            hypothesis = hypothesis[:-1]

        premise_input_ids = self.tokenizer.encode(premise, add_special_tokens=False)
        hypothesis_input_ids = self.tokenizer.encode(hypothesis, add_special_tokens=False)
        input_ids = premise_input_ids + [2] + hypothesis_input_ids

        if len(input_ids) > self.max_length - 2:
            input_ids = input_ids[:self.max_length - 2]
        
        # convert list to tensor
        length = torch.LongTensor([len(input_ids) + 2])
        input_ids = torch.LongTensor([0] + input_ids + [2])
        label = torch.LongTensor([label])
        return input_ids, label, length