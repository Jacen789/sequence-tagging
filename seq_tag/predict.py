import os
import torch

from transformers import BertTokenizer

from .data_utils import get_idx2tag
from .model import BertBilstmCrf
from .metric import get_entities

here = os.path.dirname(os.path.abspath(__file__))


def predict(hparams):
    device = hparams.device
    seed = hparams.seed
    torch.manual_seed(seed)

    pretrained_model_path = hparams.pretrained_model_path
    tagset_file = hparams.tagset_file
    model_file = hparams.model_file

    idx2tag = get_idx2tag(tagset_file)
    hparams.tagset_size = len(idx2tag)
    model = BertBilstmCrf(hparams).to(device)
    model.load_state_dict(torch.load(model_file))
    model.eval()

    tokenizer = BertTokenizer.from_pretrained(pretrained_model_path)
    while True:
        text = input("输入中文句子：")

        tokens = tokenizer.tokenize(text)
        new_tokens = ['[CLS]'] + tokens + ['SEP']
        encoded = tokenizer.batch_encode_plus([(tokens, None)], return_tensors='pt')
        input_ids = encoded['input_ids'].to(device)
        token_type_ids = encoded['token_type_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        with torch.no_grad():
            tag_ids = model.decode(input_ids, token_type_ids, attention_mask)[0]
        tags = [idx2tag[tag_id] for tag_id in tag_ids]
        print(list(zip(new_tokens, tags)))
        chunks = get_entities(tags)
        for chunk_type, chunk_start, chunk_end in chunks:
            print(chunk_type, ''.join(new_tokens[chunk_start: chunk_end + 1]))
