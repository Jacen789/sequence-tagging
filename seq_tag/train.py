import os
import time
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .data_utils import NerDataset, get_idx2tag, load_checkpoint, save_checkpoint
from .model import BertBilstmCrf
from . import metric

here = os.path.dirname(os.path.abspath(__file__))


def train(hparams):
    device = hparams.device
    seed = hparams.seed
    torch.manual_seed(seed)

    pretrained_model_path = hparams.pretrained_model_path
    train_file = hparams.train_file
    validation_file = hparams.validation_file
    log_dir = hparams.log_dir
    tagset_file = hparams.tagset_file
    model_file = hparams.model_file
    checkpoint_file = hparams.checkpoint_file

    max_len = hparams.max_len
    train_batch_size = hparams.train_batch_size
    validation_batch_size = hparams.validation_batch_size
    epochs = hparams.epochs

    learning_rate = hparams.learning_rate
    weight_decay = hparams.weight_decay

    # train_dataset
    train_dataset = NerDataset(train_file, tagset_path=tagset_file,
                               pretrained_model_path=pretrained_model_path,
                               max_len=max_len, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    # model
    idx2tag = get_idx2tag(tagset_file)
    hparams.tagset_size = len(idx2tag)
    model = BertBilstmCrf(hparams).to(device)

    # load checkpoint if one exists
    if os.path.exists(checkpoint_file):
        checkpoint_dict = load_checkpoint(checkpoint_file)
        best_f1 = checkpoint_dict['best_f1']
        epoch_offset = checkpoint_dict['best_epoch'] + 1
        model.load_state_dict(torch.load(model_file))
    else:
        checkpoint_dict = {}
        best_f1 = 0.0
        epoch_offset = 0

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    running_loss = 0.0
    writer = SummaryWriter(os.path.join(log_dir, time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())))

    for epoch in range(epoch_offset, epochs):
        print("Epoch: {}".format(epoch))
        model.train()
        for i_batch, sample_batched in enumerate(tqdm(train_loader, desc='Training')):
            token_ids = sample_batched['token_ids'].to(device)
            token_type_ids = sample_batched['token_type_ids'].to(device)
            attention_mask = sample_batched['attention_mask'].to(device)
            tag_ids = sample_batched['tag_ids'].to(device)
            model.zero_grad()
            loss = model(token_ids, tag_ids, token_type_ids, attention_mask)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i_batch % 10 == 9:
                writer.add_scalar('Training/training loss', running_loss / 10, epoch * len(train_loader) + i_batch)
                running_loss = 0.0

        if validation_file:
            validation_dataset = NerDataset(validation_file, tagset_path=tagset_file,
                                            pretrained_model_path=pretrained_model_path,
                                            max_len=max_len, is_train=False)
            val_loader = DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=False)
            model.eval()
            with torch.no_grad():
                tags_true_list = []
                tags_pred_list = []
                for val_i_batch, val_sample_batched in enumerate(tqdm(val_loader, desc='Validation')):
                    token_ids = val_sample_batched['token_ids'].to(device)
                    token_type_ids = val_sample_batched['token_type_ids'].to(device)
                    attention_mask = val_sample_batched['attention_mask'].to(device)
                    tag_ids = val_sample_batched['tag_ids'].tolist()
                    pred_tag_ids = model.decode(input_ids=token_ids, token_type_ids=token_type_ids,
                                                attention_mask=attention_mask)

                    seq_ends = attention_mask.sum(dim=1)
                    true_tag_ids = [_tag_ids[:seq_ends[i]] for i, _tag_ids in enumerate(tag_ids)]
                    batched_tags_true = [[idx2tag[tag_id] for tag_id in _tag_ids] for _tag_ids in true_tag_ids]
                    batched_tags_pred = [[idx2tag[tag_id] for tag_id in _tag_ids] for _tag_ids in pred_tag_ids]
                    tags_true_list.extend(batched_tags_true)
                    tags_pred_list.extend(batched_tags_pred)

                print(metric.classification_report(tags_true_list, tags_pred_list))
                f1 = metric.f1_score(tags_true_list, tags_pred_list)
                precision = metric.precision_score(tags_true_list, tags_pred_list)
                recall = metric.recall_score(tags_true_list, tags_pred_list)
                accuracy = metric.accuracy_score(tags_true_list, tags_pred_list)
                writer.add_scalar('Validation/f1', f1, epoch)
                writer.add_scalar('Validation/precision', precision, epoch)
                writer.add_scalar('Validation/recall', recall, epoch)
                writer.add_scalar('Validation/accuracy', accuracy, epoch)

                if checkpoint_dict.get('epoch_f1'):
                    checkpoint_dict['epoch_f1'][epoch] = f1
                else:
                    checkpoint_dict['epoch_f1'] = {epoch: f1}
                if f1 > best_f1:
                    best_f1 = f1
                    checkpoint_dict['best_f1'] = best_f1
                    checkpoint_dict['best_epoch'] = epoch
                    torch.save(model.state_dict(), model_file)
                save_checkpoint(checkpoint_dict, checkpoint_file)

    writer.close()
