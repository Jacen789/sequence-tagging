import re
import os
import json

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

here = os.path.dirname(os.path.abspath(__file__))


def read_data(input_file):
    tokens_list = []
    tags_list = []
    tagset = set()
    with open(input_file, 'r', encoding='utf-8') as f_in:
        tokens = []
        tags = []
        for line in f_in:
            line = line.strip()
            if line:
                token, tag = re.split(r'\s+', line, maxsplit=1)
                tokens.append(token)
                tags.append(tag)
                tagset.add(tag)
            else:
                if tokens and tags:
                    tokens_list.append(tokens)
                    tags_list.append(tags)
                tokens = []
                tags = []
    tagset = list(tagset)
    tagset.sort()
    return tokens_list, tags_list, tagset


def save_tagset(tagset, output_file):
    with open(output_file, 'w', encoding='utf-8') as f_out:
        f_out.write('\n'.join(tagset))


def get_tag2idx(file):
    with open(file, 'r', encoding='utf-8') as f_in:
        tagset = re.split(r'\s+', f_in.read().strip())
    return dict((tag, idx) for idx, tag in enumerate(tagset))


def get_idx2tag(file):
    with open(file, 'r', encoding='utf-8') as f_in:
        tagset = re.split(r'\s+', f_in.read().strip())
    return dict((idx, tag) for idx, tag in enumerate(tagset))


def save_checkpoint(checkpoint_dict, file):
    with open(file, 'w', encoding='utf-8') as f_out:
        json.dump(checkpoint_dict, f_out, ensure_ascii=False, indent=2)


def load_checkpoint(file):
    with open(file, 'r', encoding='utf-8') as f_in:
        checkpoint_dict = json.load(f_in)
    return checkpoint_dict


class NerDataset(Dataset):
    def __init__(self, data_file_path, tagset_path, pretrained_model_path=None, max_len=128, is_train=False):
        self.data_file_path = data_file_path
        self.tagset_path = tagset_path
        self.pretrained_model_path = pretrained_model_path or 'bert-base-chinese'
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_path)
        self.max_len = max_len
        self.is_train = is_train

        if is_train:
            self.tokens_list, self.tags_list, self.tagset = read_data(data_file_path)
            save_tagset(self.tagset, self.tagset_path)
        else:
            self.tokens_list, self.tags_list, _ = read_data(data_file_path)
        self.tag2idx = get_tag2idx(self.tagset_path)

    def __len__(self):
        return len(self.tags_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_tokens = self.tokens_list[idx]
        sample_tags = self.tags_list[idx]
        encoded = self.tokenizer.encode_plus(sample_tokens, max_length=self.max_len, pad_to_max_length=True)
        sample_token_ids = encoded['input_ids']
        sample_token_type_ids = encoded['token_type_ids']
        sample_attention_mask = encoded['attention_mask']
        sample_tags = sample_tags[:self.max_len - 2]
        sample_tags = ['O'] + sample_tags + ['O'] * (self.max_len - len(sample_tags) - 1)
        sample_tag_ids = [self.tag2idx[tag] for tag in sample_tags]

        sample = {
            'token_ids': torch.tensor(sample_token_ids),
            'token_type_ids': torch.tensor(sample_token_type_ids),
            'attention_mask': torch.tensor(sample_attention_mask),
            'tag_ids': torch.tensor(sample_tag_ids)
        }
        return sample
