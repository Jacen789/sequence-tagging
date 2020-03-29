import os
import torch
from transformers import BertModel, BertTokenizer

here = os.path.dirname(os.path.abspath(__file__))


class Bert(object):
    def __init__(self, pretrained_model_name_or_path=None, use_gpu=True):
        if pretrained_model_name_or_path is None:
            pretrained_model_name_or_path = os.path.join(here, 'pretrained_models', 'bert-base-chinese')
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model = BertModel.from_pretrained(pretrained_model_name_or_path)
        self.device = torch.device('cuda') if use_gpu and torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)

    def encode(self, texts, is_tokenized=False):
        if is_tokenized:
            texts = [(text, None) for text in texts]
        encoded = self.tokenizer.batch_encode_plus(texts, pad_to_max_length=True, return_tensors='pt')
        input_ids = encoded['input_ids'].to(self.device)
        token_type_ids = encoded['token_type_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            last_hidden_states = outputs[0]
        return last_hidden_states


def main():
    bert = Bert()
    texts = ['你好吗？', '你好！']
    print(texts)
    vecs = bert.encode(texts)
    print(vecs)

    print('-' * 79)
    texts = [list('你好吗？'), list('你好！')]
    print(texts)
    vecs = bert.encode(texts, is_tokenized=True)
    print(vecs)


if __name__ == '__main__':
    main()
