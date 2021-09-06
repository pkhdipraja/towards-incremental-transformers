import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

import model.transformer as transformer
import model.linear_transformer as linear_transformer
from model.model_utils import add_null_tokens, LabelSmoothing
from collections import OrderedDict
from seqeval.metrics import f1_score
from torch.optim.lr_scheduler import MultiStepLR


class TransformerEncoderLabelling(pl.LightningModule):
    """
    Standard Transformer encoder as baseline for sequence
    labelling task.
    """
    def __init__(self, cfgs, token2idx, label2idx, 
                 pretrained_emb=None, position_enc=True):
        super(TransformerEncoderLabelling, self).__init__()
        self.cfgs = cfgs
        self.token_size = len(token2idx)
        self.label_size = len(label2idx)
        self.label2idx = label2idx
        self.idx2label = {value: key for key, value in label2idx.items()}
        self.encoder = transformer.EncoderLabelling(cfgs, self.token_size,
                                                    self.label_size,
                                                    pretrained_emb, position_enc
                                                    )
        self.out_proj = nn.Linear(cfgs.HIDDEN_SIZE, self.label_size)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, pred_mask=None):
        logits = self.out_proj(self.encoder(x, pred_mask))
        return logits

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        if self.cfgs.DATASET == 'srl-nw-wsj':
            pred_mask = (targets == self.label2idx['B-V']).type(torch.long)
            logits = self.forward(inputs, pred_mask=pred_mask)
        else:
            logits = self.forward(inputs)

        active_labels_mask = (targets != 0)
        active_logits = logits[active_labels_mask]
        active_targets = targets[active_labels_mask]

        loss = self.loss(active_logits, active_targets)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        if self.cfgs.DATASET == 'srl-nw-wsj':
            pred_mask = (targets == self.label2idx['B-V']).type(torch.long)
            logits = self.forward(inputs, pred_mask=pred_mask)
        else:
            logits = self.forward(inputs)

        active_labels_mask = (targets != 0)
        active_logits = logits[active_labels_mask]
        active_targets = targets[active_labels_mask]

        loss = self.loss(active_logits, active_targets)

        # Compute non-incremental accuracy for this batch
        # where the index is not masked.
        pred = torch.argmax(active_logits, dim=1)
        val_acc = (active_targets == pred)

        output = OrderedDict({'val_loss': loss,
                              'val_acc': val_acc,
                              'val_pred': pred,
                              'val_label': active_targets})
        return output

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        if self.cfgs.DATASET == 'srl-nw-wsj':
            pred_mask = (targets == self.label2idx['B-V']).type(torch.long)
            logits = self.forward(inputs, pred_mask=pred_mask)
        else:
            logits = self.forward(inputs)

        active_labels_mask = (targets != 0)
        active_logits = logits[active_labels_mask]
        active_targets = targets[active_labels_mask]

        loss = self.loss(active_logits, active_targets)

        pred = torch.argmax(active_logits, dim=1)
        test_acc = (active_targets == pred)

        output = OrderedDict({'test_loss': loss,
                              'test_acc': test_acc,
                              'test_pred': pred,
                              'test_label': active_targets})
        return output

    def validation_epoch_end(self, outputs):
        val_acc = None
        val_f1 = None
        val_loss_mean = torch.stack([out['val_loss'] for out in outputs]).mean()

        if self.cfgs.DATASET not in self.cfgs.BIO_SCHEME:
            val_acc = torch.cat([out['val_acc'] for out in outputs]).view(-1)
            val_acc = torch.sum(val_acc).item()/(len(val_acc) * 1.0)
        else:
            # Compute F-score
            preds = []
            labels = []
            for out in outputs:
                preds.append(
                    [self.idx2label[pred.item()] for pred in out['val_pred']]
                )
                labels.append(
                    [self.idx2label[pred.item()] for pred in out['val_label']]
                )
            val_f1 = f1_score(labels, preds)

        values = {'val_loss_mean': val_loss_mean,
                  'val_acc': val_acc,
                  'val_f1': val_f1}
        self.log_dict(values)

    def test_epoch_end(self, outputs):
        test_acc = None
        test_f1 = None
        test_loss_mean = torch.stack([out['test_loss'] for out in outputs]).mean()

        if self.cfgs.DATASET not in self.cfgs.BIO_SCHEME:
            test_acc = torch.cat([out['test_acc'] for out in outputs]).view(-1)
            test_acc = torch.sum(test_acc).item()/(len(test_acc) * 1.0)
        else:
            # Compute F-score
            preds = []
            labels = []
            for out in outputs:
                preds.append(
                    [self.idx2label[pred.item()] for pred in out['test_pred']]
                )
                labels.append(
                    [self.idx2label[pred.item()] for pred in out['test_label']]
                )

            test_f1 = f1_score(labels, preds)

        values = {'test_loss_mean': test_loss_mean,
                  'test_acc': test_acc,
                  'test_f1': test_f1}
        self.log_dict(values)

    def configure_optimizers(self):
        optim = getattr(torch.optim, self.cfgs.OPT)
        optimizer = optim(self.parameters(), lr=self.cfgs.LR,
                          **self.cfgs.OPT_PARAMS)
        scheduler = {
            'scheduler': MultiStepLR(optimizer, milestones=self.cfgs.LR_DECAY_LIST, gamma=self.cfgs.LR_DECAY_RATE),
            'name': 'LR scheduler'
        }
        return [optimizer], [scheduler]

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, closure, on_tpu=False,
                       using_native_amp=False, using_lbfgs=False):
        if current_epoch < self.cfgs.WARMUP_EPOCH:
            warmup_lr = (current_epoch+1)/self.cfgs.WARMUP_EPOCH * self.cfgs.LR
            lr = min(warmup_lr, self.cfgs.LR)
            for pg in optimizer.param_groups:
                pg['lr'] = lr
        optimizer.step(closure=closure)


class TransformerEncoderClassification(pl.LightningModule):
    """
    Standard Transformer encoder as baseline for sequence
    classification task.
    """
    def __init__(self, cfgs, token2idx, label2idx,
                 pretrained_emb=None, position_enc=True):
        super(TransformerEncoderClassification, self).__init__()
        self.cfgs = cfgs
        self.token_size = len(token2idx)
        self.label_size = len(label2idx)
        self.idx2label = {value: key for key, value in label2idx.items()}
        self.encoder = transformer.EncoderClassification(cfgs, self.token_size,
                                                         self.label_size,
                                                         pretrained_emb, position_enc
                                                         )
        self.out_proj = nn.Linear(cfgs.HIDDEN_SIZE, self.label_size)
        self.loss = LabelSmoothing(smoothing=0.1)

    def forward(self, x):
        logits = self.out_proj(self.encoder(x))
        return logits

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        targets = targets.squeeze(-1)
        logits = self.forward(inputs)

        loss = self.loss(logits, targets)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        targets = targets.squeeze(-1)
        logits = self.forward(inputs)

        loss = self.loss(logits, targets)

        # Compute non-incremental accuracy for this batch
        # where the index is not masked.
        pred = torch.argmax(logits, dim=1)
        val_acc = (targets == pred)

        output = OrderedDict({'val_loss': loss, 'val_acc': val_acc})
        return output

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        targets = targets.squeeze(-1)
        logits = self.forward(inputs)

        loss = self.loss(logits, targets)

        pred = torch.argmax(logits, dim=1)
        test_acc = (targets == pred)

        output = OrderedDict({'test_loss': loss, 'test_acc': test_acc})
        return output

    def validation_epoch_end(self, outputs):
        val_acc = torch.cat([out['val_acc'] for out in outputs]).view(-1)
        val_acc = torch.sum(val_acc).item()/(len(val_acc) * 1.0)

        val_loss_mean = torch.stack([out['val_loss'] for out in outputs]).mean()

        values = {'val_loss_mean': val_loss_mean, 'val_acc': val_acc}
        self.log_dict(values)

    def test_epoch_end(self, outputs):
        test_acc = torch.cat([out['test_acc'] for out in outputs]).view(-1)
        test_acc = torch.sum(test_acc).item()/(len(test_acc) * 1.0)

        test_loss_mean = torch.stack([out['test_loss'] for out in outputs]).mean()

        values = {'test_loss_mean': test_loss_mean, 'test_acc': test_acc}
        self.log_dict(values)

    def configure_optimizers(self):
        optim = getattr(torch.optim, self.cfgs.OPT)
        optimizer = optim(self.parameters(), lr=self.cfgs.LR,
                          **self.cfgs.OPT_PARAMS)
        scheduler = {
            'scheduler': MultiStepLR(optimizer, milestones=self.cfgs.LR_DECAY_LIST, gamma=self.cfgs.LR_DECAY_RATE),
            'name': 'LR scheduler'
        }
        return [optimizer], [scheduler]

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, closure, on_tpu=False,
                       using_native_amp=False, using_lbfgs=False):
        if current_epoch < self.cfgs.WARMUP_EPOCH:
            warmup_lr = (current_epoch+1)/self.cfgs.WARMUP_EPOCH * self.cfgs.LR
            lr = min(warmup_lr, self.cfgs.LR)
            for pg in optimizer.param_groups:
                pg['lr'] = lr
        optimizer.step(closure=closure)


class LinearCausalEncoderLabelling(pl.LightningModule):
    """
    Linear Transformers with causal masking for sequence labelling task.
    """
    def __init__(self, cfgs, token2idx, label2idx,
                 pretrained_emb=None, position_enc=True):
        super(LinearCausalEncoderLabelling, self).__init__()
        self.cfgs = cfgs
        self.token_size = len(token2idx)
        self.label_size = len(label2idx)
        self.token2idx = token2idx
        self.label2idx = label2idx
        self.idx2label = {value: key for key, value in label2idx.items()}
        self.encoder = linear_transformer.LinearCausalEncoderLabelling(cfgs, self.token_size,
                                                                       self.label_size,
                                                                       pretrained_emb, position_enc
                                                                       )
        self.out_proj = nn.Linear(cfgs.HIDDEN_SIZE, self.label_size)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, pred_mask=None):
        logits = self.out_proj(self.encoder(x, pred_mask))
        return logits

    def training_step(self, batch, batch_idx):
        inputs, targets = batch

        # Shift label by d tokens and add d NULL tokens to input where d is delay.
        if self.cfgs.DELAY > 0:
            inputs = add_null_tokens(inputs,
                                     self.cfgs.DELAY,
                                     self.token2idx['NULL'])
            targets = torch.roll(targets, self.cfgs.DELAY, 1)

        if self.cfgs.DATASET == 'srl-nw-wsj':
            pred_mask = (targets == self.label2idx['B-V']).type(torch.long)
            logits = self.forward(inputs, pred_mask=pred_mask)
        else:
            logits = self.forward(inputs)

        active_labels_mask = (targets != 0)
        active_logits = logits[active_labels_mask]
        active_targets = targets[active_labels_mask]

        loss = self.loss(active_logits, active_targets)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch

        # Shift label by d tokens and add d NULL tokens to input where d is delay.
        if self.cfgs.DELAY > 0:
            inputs = add_null_tokens(inputs,
                                     self.cfgs.DELAY,
                                     self.token2idx['NULL'])
            targets = torch.roll(targets, self.cfgs.DELAY, 1)

        if self.cfgs.DATASET == 'srl-nw-wsj':
            pred_mask = (targets == self.label2idx['B-V']).type(torch.long)
            logits = self.forward(inputs, pred_mask=pred_mask)
        else:
            logits = self.forward(inputs)

        active_labels_mask = (targets != 0)
        active_logits = logits[active_labels_mask]
        active_targets = targets[active_labels_mask]

        loss = self.loss(active_logits, active_targets)

        # Compute non-incremental accuracy for this batch
        # where the index is not masked.
        pred = torch.argmax(active_logits, dim=1)
        val_acc = (active_targets == pred)

        output = OrderedDict({'val_loss': loss,
                              'val_acc': val_acc,
                              'val_pred': pred,
                              'val_label': active_targets})
        return output

    def test_step(self, batch, batch_idx):
        # This assume batch size = 1
        inputs, targets = batch

        # Shift label by d tokens and add d NULL tokens to input where d is delay.
        if self.cfgs.DELAY > 0:
            inputs = add_null_tokens(inputs,
                                     self.cfgs.DELAY,
                                     self.token2idx['NULL'])
            targets = torch.roll(targets, self.cfgs.DELAY, 1)

        if self.cfgs.DATASET == 'srl-nw-wsj':
            pred_mask = (targets == self.label2idx['B-V']).type(torch.long)
            logits = self.forward(inputs, pred_mask=pred_mask)
        else:
            logits = self.forward(inputs)

        active_labels_mask = (targets != 0)
        active_logits = logits[active_labels_mask]
        active_targets = targets[active_labels_mask]

        loss = self.loss(active_logits, active_targets)

        pred = torch.argmax(active_logits, dim=1)
        test_acc = (active_targets == pred)

        output = OrderedDict({'test_loss': loss,
                              'test_acc': test_acc,
                              'test_pred': pred,
                              'test_label': active_targets})
        return output

    def validation_epoch_end(self, outputs):
        val_acc = None
        val_f1 = None
        val_loss_mean = torch.stack([out['val_loss'] for out in outputs]).mean()

        if self.cfgs.DATASET not in self.cfgs.BIO_SCHEME:
            val_acc = torch.cat([out['val_acc'] for out in outputs]).view(-1)
            val_acc = torch.sum(val_acc).item()/(len(val_acc) * 1.0)
        else:
            # Compute F-score
            preds = []
            labels = []
            for out in outputs:
                preds.append(
                    [self.idx2label[pred.item()] for pred in out['val_pred']]
                )
                labels.append(
                    [self.idx2label[pred.item()] for pred in out['val_label']]
                )
            val_f1 = f1_score(labels, preds)

        values = {'val_loss_mean': val_loss_mean,
                  'val_acc': val_acc,
                  'val_f1': val_f1}
        self.log_dict(values)

    def test_epoch_end(self, outputs):
        test_acc = None
        test_f1 = None
        test_loss_mean = torch.stack([out['test_loss'] for out in outputs]).mean()

        if self.cfgs.DATASET not in self.cfgs.BIO_SCHEME:
            test_acc = torch.cat([out['test_acc'] for out in outputs]).view(-1)
            test_acc = torch.sum(test_acc).item()/(len(test_acc) * 1.0)
        else:
            # Compute F-score
            preds = []
            labels = []
            for out in outputs:
                preds.append(
                    [self.idx2label[pred.item()] for pred in out['test_pred']]
                )
                labels.append(
                    [self.idx2label[pred.item()] for pred in out['test_label']]
                )

            test_f1 = f1_score(labels, preds)

        values = {'test_loss_mean': test_loss_mean,
                  'test_acc': test_acc,
                  'test_f1': test_f1}
        self.log_dict(values)

    def configure_optimizers(self):
        optim = getattr(torch.optim, self.cfgs.OPT)
        optimizer = optim(self.parameters(), lr=self.cfgs.LR,
                          **self.cfgs.OPT_PARAMS)
        scheduler = {
            'scheduler': MultiStepLR(optimizer, milestones=self.cfgs.LR_DECAY_LIST, gamma=self.cfgs.LR_DECAY_RATE),
            'name': 'LR scheduler'
        }
        return [optimizer], [scheduler]

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, closure, on_tpu=False,
                       using_native_amp=False, using_lbfgs=False):
        if current_epoch < self.cfgs.WARMUP_EPOCH:
            warmup_lr = (current_epoch+1)/self.cfgs.WARMUP_EPOCH * self.cfgs.LR
            lr = min(warmup_lr, self.cfgs.LR)
            for pg in optimizer.param_groups:
                pg['lr'] = lr 
        optimizer.step(closure=closure)


class LinearEncoderLabelling(pl.LightningModule):
    """
    Linear Transformers for sequence labelling task.
    Attend to all tokens.
    """
    def __init__(self, cfgs, token2idx, label2idx,
                 pretrained_emb=None, position_enc=True):
        super(LinearEncoderLabelling, self).__init__()
        self.cfgs = cfgs
        self.token_size = len(token2idx)
        self.label_size = len(label2idx)
        self.label2idx = label2idx
        self.idx2label = {value: key for key, value in label2idx.items()}
        self.encoder = linear_transformer.LinearEncoderLabelling(cfgs, self.token_size,
                                                                 self.label_size,
                                                                 pretrained_emb, position_enc
                                                                 )
        self.out_proj = nn.Linear(cfgs.HIDDEN_SIZE, self.label_size)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, pred_mask=None):
        logits = self.out_proj(self.encoder(x, pred_mask))
        return logits

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        if self.cfgs.DATASET == 'srl-nw-wsj':
            pred_mask = (targets == self.label2idx['B-V']).type(torch.long)
            logits = self.forward(inputs, pred_mask=pred_mask)
        else:
            logits = self.forward(inputs)

        active_labels_mask = (targets != 0)
        active_logits = logits[active_labels_mask]
        active_targets = targets[active_labels_mask]

        loss = self.loss(active_logits, active_targets)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        if self.cfgs.DATASET == 'srl-nw-wsj':
            pred_mask = (targets == self.label2idx['B-V']).type(torch.long)
            logits = self.forward(inputs, pred_mask=pred_mask)
        else:
            logits = self.forward(inputs)

        active_labels_mask = (targets != 0)
        active_logits = logits[active_labels_mask]
        active_targets = targets[active_labels_mask]

        loss = self.loss(active_logits, active_targets)

        # Compute non-incremental accuracy for this batch
        # where the index is not masked.
        pred = torch.argmax(active_logits, dim=1)
        val_acc = (active_targets == pred)

        output = OrderedDict({'val_loss': loss,
                              'val_acc': val_acc,
                              'val_pred': pred,
                              'val_label': active_targets})
        return output

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        if self.cfgs.DATASET == 'srl-nw-wsj':
            pred_mask = (targets == self.label2idx['B-V']).type(torch.long)
            logits = self.forward(inputs, pred_mask=pred_mask)
        else:
            logits = self.forward(inputs)

        active_labels_mask = (targets != 0)
        active_logits = logits[active_labels_mask]
        active_targets = targets[active_labels_mask]

        loss = self.loss(active_logits, active_targets)

        pred = torch.argmax(active_logits, dim=1)
        test_acc = (active_targets == pred)

        output = OrderedDict({'test_loss': loss,
                              'test_acc': test_acc,
                              'test_pred': pred,
                              'test_label': active_targets})
        return output

    def validation_epoch_end(self, outputs):
        val_acc = None
        val_f1 = None
        val_loss_mean = torch.stack([out['val_loss'] for out in outputs]).mean()

        if self.cfgs.DATASET not in self.cfgs.BIO_SCHEME:
            val_acc = torch.cat([out['val_acc'] for out in outputs]).view(-1)
            val_acc = torch.sum(val_acc).item()/(len(val_acc) * 1.0)
        else:
            # Compute F-score
            preds = []
            labels = []
            for out in outputs:
                preds.append(
                    [self.idx2label[pred.item()] for pred in out['val_pred']]
                )
                labels.append(
                    [self.idx2label[pred.item()] for pred in out['val_label']]
                )
            val_f1 = f1_score(labels, preds)

        values = {'val_loss_mean': val_loss_mean,
                  'val_acc': val_acc,
                  'val_f1': val_f1}
        self.log_dict(values)

    def test_epoch_end(self, outputs):
        test_acc = None
        test_f1 = None
        test_loss_mean = torch.stack([out['test_loss'] for out in outputs]).mean()

        if self.cfgs.DATASET not in self.cfgs.BIO_SCHEME:
            test_acc = torch.cat([out['test_acc'] for out in outputs]).view(-1)
            test_acc = torch.sum(test_acc).item()/(len(test_acc) * 1.0)
        else:
            # Compute F-score
            preds = []
            labels = []
            for out in outputs:
                preds.append(
                    [self.idx2label[pred.item()] for pred in out['test_pred']]
                )
                labels.append(
                    [self.idx2label[pred.item()] for pred in out['test_label']]
                )

            test_f1 = f1_score(labels, preds)

        values = {'test_loss_mean': test_loss_mean,
                  'test_acc': test_acc,
                  'test_f1': test_f1}
        self.log_dict(values)

    def configure_optimizers(self):
        optim = getattr(torch.optim, self.cfgs.OPT)
        optimizer = optim(self.parameters(), lr=self.cfgs.LR,
                          **self.cfgs.OPT_PARAMS)
        scheduler = {
            'scheduler': MultiStepLR(optimizer, milestones=self.cfgs.LR_DECAY_LIST, gamma=self.cfgs.LR_DECAY_RATE),
            'name': 'LR scheduler'
        }
        return [optimizer], [scheduler]

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, closure, on_tpu=False,
                       using_native_amp=False, using_lbfgs=False):
        if current_epoch < self.cfgs.WARMUP_EPOCH:
            warmup_lr = (current_epoch+1)/self.cfgs.WARMUP_EPOCH * self.cfgs.LR
            lr = min(warmup_lr, self.cfgs.LR)
            for pg in optimizer.param_groups:
                pg['lr'] = lr
        optimizer.step(closure=closure)


class RecurrentEncoderLabelling(pl.LightningModule):
    """
    Linear Transformers as RNN for sequence labelling task.
    """
    def __init__(self, cfgs, token2idx, label2idx,
                 pretrained_emb=None, position_enc=True):
        super(RecurrentEncoderLabelling, self).__init__()
        self.cfgs = cfgs
        self.token_size = len(token2idx)
        self.label_size = len(label2idx)
        self.idx2label = {value: key for key, value in label2idx.items()}
        self.encoder = linear_transformer.RecurrentEncoderLabelling(cfgs, self.token_size,
                                                                 self.label_size,
                                                                 pretrained_emb, position_enc
                                                                 )
        self.out_proj = nn.Linear(cfgs.HIDDEN_SIZE, self.label_size)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, i=0, memory=None):
        x, memory = self.encoder(x, i, memory)
        x = self.out_proj(x)
        return x, memory

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        memory = None
        logits = []
        for i in range(self.cfgs.MAX_TOKEN):
            out_i, memory = self.forward(inputs[:, i:i+1], i=i, memory=memory)
            logits.append(out_i)
        logits = torch.stack(logits, dim=1)

        active_labels_mask = (targets != 0)
        active_logits = logits[active_labels_mask]
        active_targets = targets[active_labels_mask]

        loss = self.loss(active_logits, active_targets)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        memory = None
        logits = []
        for i in range(self.cfgs.MAX_TOKEN):
            out_i, memory = self.forward(inputs[:, i:i+1], i=i, memory=memory)
            logits.append(out_i)
        logits = torch.stack(logits, dim=1)

        active_labels_mask = (targets != 0)
        active_logits = logits[active_labels_mask]
        active_targets = targets[active_labels_mask]

        loss = self.loss(active_logits, active_targets)

        # Compute non-incremental accuracy for this batch
        # where the index is not masked.
        pred = torch.argmax(active_logits, dim=1)
        val_acc = (active_targets == pred)

        output = OrderedDict({'val_loss': loss,
                              'val_acc': val_acc,
                              'val_pred': pred,
                              'val_label': active_targets})
        return output

    def test_step(self, batch, batch_idx):
        # This assume batch size = 1
        inputs, targets = batch
        active_labels_mask = (targets != 0)
        active_inputs = inputs[active_labels_mask].view(1, -1)

        seq_len = active_inputs.size(1)
        memory = None
        logits = []
        for i in range(seq_len):
            out_i, memory = self.forward(active_inputs[:, i:i+1],
                                         i=i,
                                         memory=memory,
                                         )
            logits.append(out_i)

        active_logits = torch.stack(logits, dim=1).squeeze(0)
        active_targets = targets[active_labels_mask]

        loss = self.loss(active_logits, active_targets)

        pred = torch.argmax(active_logits, dim=1)
        test_acc = (active_targets == pred)

        output = OrderedDict({'test_loss': loss,
                              'test_acc': test_acc,
                              'test_pred': pred,
                              'test_label': active_targets})
        return output

    def validation_epoch_end(self, outputs):
        val_acc = None
        val_f1 = None
        val_loss_mean = torch.stack([out['val_loss'] for out in outputs]).mean()

        if self.cfgs.DATASET not in self.cfgs.BIO_SCHEME:
            val_acc = torch.cat([out['val_acc'] for out in outputs]).view(-1)
            val_acc = torch.sum(val_acc).item()/(len(val_acc) * 1.0)
        else:
            # Compute F-score
            preds = []
            labels = []
            for out in outputs:
                preds.append(
                    [self.idx2label[pred.item()] for pred in out['val_pred']]
                )
                labels.append(
                    [self.idx2label[pred.item()] for pred in out['val_label']]
                )
            val_f1 = f1_score(labels, preds)

        values = {'val_loss_mean': val_loss_mean,
                  'val_acc': val_acc,
                  'val_f1': val_f1}
        self.log_dict(values)

    def test_epoch_end(self, outputs):
        test_acc = None
        test_f1 = None
        test_loss_mean = torch.stack([out['test_loss'] for out in outputs]).mean()

        if self.cfgs.DATASET not in self.cfgs.BIO_SCHEME:
            test_acc = torch.cat([out['test_acc'] for out in outputs]).view(-1)
            test_acc = torch.sum(test_acc).item()/(len(test_acc) * 1.0)
        else:
            # Compute F-score
            preds = []
            labels = []
            for out in outputs:
                preds.append(
                    [self.idx2label[pred.item()] for pred in out['test_pred']]
                )
                labels.append(
                    [self.idx2label[pred.item()] for pred in out['test_label']]
                )

            test_f1 = f1_score(labels, preds)

        values = {'test_loss_mean': test_loss_mean,
                  'test_acc': test_acc,
                  'test_f1': test_f1}
        self.log_dict(values)

    def configure_optimizers(self):
        optim = getattr(torch.optim, self.cfgs.OPT)
        optimizer = optim(self.parameters(), lr=self.cfgs.LR,
                          **self.cfgs.OPT_PARAMS)
        scheduler = {
            'scheduler': MultiStepLR(optimizer, milestones=self.cfgs.LR_DECAY_LIST, gamma=self.cfgs.LR_DECAY_RATE),
            'name': 'LR scheduler'
        }
        return [optimizer], [scheduler]

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, closure, on_tpu=False,
                       using_native_amp=False, using_lbfgs=False):
        if current_epoch < self.cfgs.WARMUP_EPOCH:
            warmup_lr = (current_epoch+1)/self.cfgs.WARMUP_EPOCH * self.cfgs.LR
            lr = min(warmup_lr, self.cfgs.LR)
            for pg in optimizer.param_groups:
                pg['lr'] = lr
        optimizer.step(closure=closure)


class LinearEncoderClassification(pl.LightningModule):
    """
    Linear Transformers for sequence classification task.
    Attend to all tokens.
    """
    def __init__(self, cfgs, token2idx, label2idx,
                 pretrained_emb=None, position_enc=True):
        super(LinearEncoderClassification, self).__init__()
        self.cfgs = cfgs
        self.token_size = len(token2idx)
        self.label_size = len(label2idx)
        self.idx2label = {value: key for key, value in label2idx.items()}
        self.encoder = linear_transformer.LinearEncoderClassification(cfgs, self.token_size,
                                                                      self.label_size,
                                                                      pretrained_emb, position_enc
                                                                      )
        self.out_proj = nn.Linear(cfgs.HIDDEN_SIZE, self.label_size)
        self.loss = LabelSmoothing(smoothing=0.1)

    def forward(self, x):
        logits = self.out_proj(self.encoder(x))
        return logits

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        targets = targets.squeeze(-1)
        logits = self.forward(inputs)

        loss = self.loss(logits, targets)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        targets = targets.squeeze(-1)
        logits = self.forward(inputs)

        loss = self.loss(logits, targets)

        # Compute non-incremental accuracy for this batch
        # where the index is not masked.
        pred = torch.argmax(logits, dim=1)
        val_acc = (targets == pred)

        output = OrderedDict({'val_loss': loss, 'val_acc': val_acc})
        return output

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        targets = targets.squeeze(-1)
        logits = self.forward(inputs)

        loss = self.loss(logits, targets)

        pred = torch.argmax(logits, dim=1)
        test_acc = (targets == pred)

        output = OrderedDict({'test_loss': loss, 'test_acc': test_acc})
        return output

    def validation_epoch_end(self, outputs):
        val_acc = torch.cat([out['val_acc'] for out in outputs]).view(-1)
        val_acc = torch.sum(val_acc).item()/(len(val_acc) * 1.0)

        val_loss_mean = torch.stack([out['val_loss'] for out in outputs]).mean()

        values = {'val_loss_mean': val_loss_mean, 'val_acc': val_acc}
        self.log_dict(values)

    def test_epoch_end(self, outputs):
        test_acc = torch.cat([out['test_acc'] for out in outputs]).view(-1)
        test_acc = torch.sum(test_acc).item()/(len(test_acc) * 1.0)

        test_loss_mean = torch.stack([out['test_loss'] for out in outputs]).mean()

        values = {'test_loss_mean': test_loss_mean, 'test_acc': test_acc}
        self.log_dict(values)

    def configure_optimizers(self):
        optim = getattr(torch.optim, self.cfgs.OPT)
        optimizer = optim(self.parameters(), lr=self.cfgs.LR,
                          **self.cfgs.OPT_PARAMS)
        scheduler = {
            'scheduler': MultiStepLR(optimizer, milestones=self.cfgs.LR_DECAY_LIST, gamma=self.cfgs.LR_DECAY_RATE),
            'name': 'LR scheduler'
        }
        return [optimizer], [scheduler]

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, closure, on_tpu=False,
                       using_native_amp=False, using_lbfgs=False):
        if current_epoch < self.cfgs.WARMUP_EPOCH:
            warmup_lr = (current_epoch+1)/self.cfgs.WARMUP_EPOCH * self.cfgs.LR
            lr = min(warmup_lr, self.cfgs.LR)
            for pg in optimizer.param_groups:
                pg['lr'] = lr
        optimizer.step(closure=closure)


class LinearCausalEncoderClassification(pl.LightningModule):
    """
    Linear Transformers with causal masking for sequence classification task.
    """
    def __init__(self, cfgs, token2idx, label2idx,
                 pretrained_emb=None, position_enc=True):
        super(LinearCausalEncoderClassification, self).__init__()
        self.cfgs = cfgs
        self.token_size = len(token2idx)
        self.label_size = len(label2idx)
        self.token2idx = token2idx
        self.idx2label = {value: key for key, value in label2idx.items()}
        self.encoder = linear_transformer.LinearCausalEncoderClassification(cfgs, self.token_size,
                                                                            self.label_size,
                                                                            pretrained_emb, position_enc
                                                                            )
        self.out_proj = nn.Linear(cfgs.HIDDEN_SIZE, self.label_size)
        self.loss = LabelSmoothing(smoothing=0.1)

    def forward(self, x, valid=False):
        logits, mean_mask = self.encoder(x, valid)
        logits = self.out_proj(logits)
        return logits, mean_mask

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        targets = targets.expand(targets.size(0), self.cfgs.MAX_TOKEN)

        if self.cfgs.DELAY > 0:
            targets_mask = (torch.sum(
                torch.abs(inputs.unsqueeze(2)), dim=-1
                ) == 0).unsqueeze(2)
            targets = targets.masked_fill(targets_mask.squeeze(), 0)

            inputs = add_null_tokens(inputs,
                                     self.cfgs.DELAY,
                                     self.token2idx['NULL'])
            active_targets_mask = (targets != 0)

            logits, _ = self.forward(inputs)
            active_targets = targets[active_targets_mask]
        else:
            logits, mean_mask = self.forward(inputs)
            active_targets = targets[mean_mask]
        loss = self.loss(logits, active_targets)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        targets = targets.squeeze(-1)

        if self.cfgs.DELAY > 0:
            inputs = add_null_tokens(inputs,
                                     self.cfgs.DELAY,
                                     self.token2idx['NULL'])

        logits, _ = self.forward(inputs, valid=True)

        loss = self.loss(logits, targets)

        # Compute non-incremental accuracy for this batch
        # where the index is not masked.
        pred = torch.argmax(logits, dim=1)
        val_acc = (targets == pred)

        output = OrderedDict({'val_loss': loss, 'val_acc': val_acc})
        return output

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        targets = targets.squeeze(-1)

        if self.cfgs.DELAY > 0:
            inputs = add_null_tokens(inputs,
                                     self.cfgs.DELAY,
                                     self.token2idx['NULL'])

        logits, _ = self.forward(inputs, valid=True)

        loss = self.loss(logits, targets)

        pred = torch.argmax(logits, dim=1)
        test_acc = (targets == pred)

        output = OrderedDict({'test_loss': loss, 'test_acc': test_acc})
        return output

    def validation_epoch_end(self, outputs):
        val_acc = torch.cat([out['val_acc'] for out in outputs]).view(-1)
        val_acc = torch.sum(val_acc).item()/(len(val_acc) * 1.0)

        val_loss_mean = torch.stack([out['val_loss'] for out in outputs]).mean()

        values = {'val_loss_mean': val_loss_mean, 'val_acc': val_acc}
        self.log_dict(values)

    def test_epoch_end(self, outputs):
        test_acc = torch.cat([out['test_acc'] for out in outputs]).view(-1)
        test_acc = torch.sum(test_acc).item()/(len(test_acc) * 1.0)

        test_loss_mean = torch.stack([out['test_loss'] for out in outputs]).mean()

        values = {'test_loss_mean': test_loss_mean, 'test_acc': test_acc}
        self.log_dict(values)

    def configure_optimizers(self):
        optim = getattr(torch.optim, self.cfgs.OPT)
        optimizer = optim(self.parameters(), lr=self.cfgs.LR,
                          **self.cfgs.OPT_PARAMS)
        scheduler = {
            'scheduler': MultiStepLR(optimizer, milestones=self.cfgs.LR_DECAY_LIST, gamma=self.cfgs.LR_DECAY_RATE),
            'name': 'LR scheduler'
        }
        return [optimizer], [scheduler]

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, closure, on_tpu=False,
                       using_native_amp=False, using_lbfgs=False):
        if current_epoch < self.cfgs.WARMUP_EPOCH:
            warmup_lr = (current_epoch+1)/self.cfgs.WARMUP_EPOCH * self.cfgs.LR
            lr = min(warmup_lr, self.cfgs.LR)
            for pg in optimizer.param_groups:
                pg['lr'] = lr
        optimizer.step(closure=closure)


class RecurrentEncoderClassification(pl.LightningModule):
    """
    Linear Transformers as RNN for sequence classification task.
    """
    def __init__(self, cfgs, token2idx, label2idx,
                 pretrained_emb=None, position_enc=True):
        super(RecurrentEncoderClassification, self).__init__()
        self.cfgs = cfgs
        self.token_size = len(token2idx)
        self.label_size = len(label2idx)
        self.idx2label = {value: key for key, value in label2idx.items()}
        self.encoder = linear_transformer.RecurrentEncoderClassification(cfgs, self.token_size,
                                                                         self.label_size,
                                                                         pretrained_emb, position_enc
                                                                         )
        self.out_proj = nn.Linear(cfgs.HIDDEN_SIZE, self.label_size)
        self.loss = LabelSmoothing(smoothing=0.1)

    def forward(self, x, i=0, memory=None):
        x, memory = self.encoder(x, i, memory)
        return x, memory

    def training_step(self, batch, batch_idx):
        pass  # See utils/utils.py

    def validation_step(self, batch, batch_idx):
        # This assume batch size = 1
        inputs, targets = batch
        targets = targets.squeeze(-1)

        active_inputs_mask = (inputs != 0)
        active_inputs = inputs[active_inputs_mask].view(1, -1)

        seq_len = active_inputs.size(1)
        memory = None
        outs = []
        for i in range(seq_len):
            out_i, memory = self.forward(active_inputs[:, i:i+1],
                                         i=i,
                                         memory=memory,
                                         )
            outs.append(out_i)

        active_outs = torch.stack(outs, dim=1).squeeze(0)
        active_outs = torch.sum(active_outs, dim=0, keepdim=True)

        # logits = self.forward(active_outs, i=seq_len, valid=True)
        logits = self.out_proj(active_outs/seq_len)

        # Compute non-incremental accuracy for this batch
        # where the index is not masked.
        loss = self.loss(logits, targets)
        pred = torch.argmax(logits, dim=1)
        val_acc = (targets == pred)

        output = OrderedDict({'val_loss': loss, 'val_acc': val_acc})
        return output

    def test_step(self, batch, batch_idx):
        # This assume batch size = 1
        inputs, targets = batch
        targets = targets.squeeze(-1)

        active_inputs_mask = (inputs != 0)
        active_inputs = inputs[active_inputs_mask].view(1, -1)

        seq_len = active_inputs.size(1)
        memory = None
        outs = []
        for i in range(seq_len):
            out_i, memory = self.forward(active_inputs[:, i:i+1],
                                         i=i,
                                         memory=memory,
                                         )
            outs.append(out_i)

        active_outs = torch.stack(outs, dim=1).squeeze(0)
        active_outs = torch.sum(active_outs, dim=0, keepdim=True)

        logits = self.out_proj(active_outs/seq_len)

        loss = self.loss(logits, targets)
        pred = torch.argmax(logits, dim=1)
        test_acc = (targets == pred)

        output = OrderedDict({'test_loss': loss, 'test_acc': test_acc})
        return output

    def validation_epoch_end(self, outputs):
        val_acc = torch.cat([out['val_acc'] for out in outputs]).view(-1)
        val_acc = torch.sum(val_acc).item()/(len(val_acc) * 1.0)

        val_loss_mean = torch.stack([out['val_loss'] for out in outputs]).mean()

        values = {'val_loss_mean': val_loss_mean, 'val_acc': val_acc}
        self.log_dict(values)

    def test_epoch_end(self, outputs):
        test_acc = torch.cat([out['test_acc'] for out in outputs]).view(-1)
        test_acc = torch.sum(test_acc).item()/(len(test_acc) * 1.0)

        test_loss_mean = torch.stack([out['test_loss'] for out in outputs]).mean()

        values = {'test_loss_mean': test_loss_mean, 'test_acc': test_acc}
        self.log_dict(values)

    def configure_optimizers(self):
        optim = getattr(torch.optim, self.cfgs.OPT)
        optimizer = optim(self.parameters(), lr=self.cfgs.LR,
                          **self.cfgs.OPT_PARAMS)
        scheduler = {
            'scheduler': MultiStepLR(optimizer, milestones=self.cfgs.LR_DECAY_LIST, gamma=self.cfgs.LR_DECAY_RATE),
            'name': 'LR scheduler'
        }
        return [optimizer], [scheduler]

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, closure, on_tpu=False,
                       using_native_amp=False, using_lbfgs=False):
        if current_epoch < self.cfgs.WARMUP_EPOCH:
            warmup_lr = (current_epoch+1)/self.cfgs.WARMUP_EPOCH * self.cfgs.LR
            lr = min(warmup_lr, self.cfgs.LR)
            for pg in optimizer.param_groups:
                pg['lr'] = lr
        optimizer.step(closure=closure)


class IncrementalTransformerEncoderLabelling(pl.LightningModule):  # Unused
    """
    Standard incremental Transformer encoder for sequence
    labelling task.
    """
    def __init__(self, cfgs, token2idx, label2idx, 
                 pretrained_emb=None, position_enc=True):
        super(IncrementalTransformerEncoderLabelling, self).__init__()
        self.cfgs = cfgs
        self.token_size = len(token2idx)
        self.label_size = len(label2idx)
        self.label2idx = label2idx
        self.idx2label = {value: key for key, value in label2idx.items()}
        self.encoder = transformer.IncrementalEncoderLabelling(cfgs, self.token_size,
                                                               self.label_size,
                                                               pretrained_emb, position_enc
                                                               )
        self.out_proj = nn.Linear(cfgs.HIDDEN_SIZE, self.label_size)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, pred_mask=None, valid=False):
        logits = self.out_proj(self.encoder(x, pred_mask, valid))
        return logits

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        if self.cfgs.DATASET == 'srl-nw-wsj':
            pred_mask = (targets == self.label2idx['B-V']).type(torch.long)
            logits = self.forward(inputs, pred_mask=pred_mask)
        else:
            logits = self.forward(inputs)

        active_labels_mask = (targets != 0)
        active_logits = logits[active_labels_mask]
        active_targets = targets[active_labels_mask]

        loss = self.loss(active_logits, active_targets)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        if self.cfgs.DATASET == 'srl-nw-wsj':
            pred_mask = (targets == self.label2idx['B-V']).type(torch.long)
            logits = self.forward(inputs, pred_mask=pred_mask)
        else:
            logits = self.forward(inputs)

        active_labels_mask = (targets != 0)
        active_logits = logits[active_labels_mask]
        active_targets = targets[active_labels_mask]

        loss = self.loss(active_logits, active_targets)

        # Compute non-incremental accuracy for this batch
        # where the index is not masked.
        pred = torch.argmax(active_logits, dim=1)
        val_acc = (active_targets == pred)

        output = OrderedDict({'val_loss': loss,
                              'val_acc': val_acc,
                              'val_pred': pred,
                              'val_label': active_targets})
        return output

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        if self.cfgs.DATASET == 'srl-nw-wsj':
            pred_mask = (targets == self.label2idx['B-V']).type(torch.long)
            logits = self.forward(inputs, pred_mask=pred_mask)
        else:
            logits = self.forward(inputs)

        active_labels_mask = (targets != 0)
        active_logits = logits[active_labels_mask]
        active_targets = targets[active_labels_mask]

        loss = self.loss(active_logits, active_targets)

        pred = torch.argmax(active_logits, dim=1)
        test_acc = (active_targets == pred)

        output = OrderedDict({'test_loss': loss,
                              'test_acc': test_acc,
                              'test_pred': pred,
                              'test_label': active_targets})
        return output

    def validation_epoch_end(self, outputs):
        val_acc = None
        val_f1 = None
        val_loss_mean = torch.stack([out['val_loss'] for out in outputs]).mean()

        if self.cfgs.DATASET not in self.cfgs.BIO_SCHEME:
            val_acc = torch.cat([out['val_acc'] for out in outputs]).view(-1)
            val_acc = torch.sum(val_acc).item()/(len(val_acc) * 1.0)
        else:
            # Compute F-score
            preds = []
            labels = []
            for out in outputs:
                preds.append(
                    [self.idx2label[pred.item()] for pred in out['val_pred']]
                )
                labels.append(
                    [self.idx2label[pred.item()] for pred in out['val_label']]
                )
            val_f1 = f1_score(labels, preds)

        values = {'val_loss_mean': val_loss_mean,
                  'val_acc': val_acc,
                  'val_f1': val_f1}
        self.log_dict(values)

    def test_epoch_end(self, outputs):
        test_acc = None
        test_f1 = None
        test_loss_mean = torch.stack([out['test_loss'] for out in outputs]).mean()

        if self.cfgs.DATASET not in self.cfgs.BIO_SCHEME:
            test_acc = torch.cat([out['test_acc'] for out in outputs]).view(-1)
            test_acc = torch.sum(test_acc).item()/(len(test_acc) * 1.0)
        else:
            # Compute F-score
            preds = []
            labels = []
            for out in outputs:
                preds.append(
                    [self.idx2label[pred.item()] for pred in out['test_pred']]
                )
                labels.append(
                    [self.idx2label[pred.item()] for pred in out['test_label']]
                )

            test_f1 = f1_score(labels, preds)

        values = {'test_loss_mean': test_loss_mean,
                  'test_acc': test_acc,
                  'test_f1': test_f1}
        self.log_dict(values)

    def configure_optimizers(self):
        optim = getattr(torch.optim, self.cfgs.OPT)
        optimizer = optim(self.parameters(), lr=self.cfgs.LR,
                          **self.cfgs.OPT_PARAMS)
        scheduler = {
            'scheduler': MultiStepLR(optimizer, milestones=self.cfgs.LR_DECAY_LIST, gamma=self.cfgs.LR_DECAY_RATE),
            'name': 'LR scheduler'
        }
        return [optimizer], [scheduler]

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, closure, on_tpu=False,
                       using_native_amp=False, using_lbfgs=False):
        if current_epoch < self.cfgs.WARMUP_EPOCH:
            warmup_lr = (current_epoch+1)/self.cfgs.WARMUP_EPOCH * self.cfgs.LR
            lr = min(warmup_lr, self.cfgs.LR)
            for pg in optimizer.param_groups:
                pg['lr'] = lr
        optimizer.step(closure=closure)


class IncrementalTransformerEncoderClassification(pl.LightningModule):  # Unused
    """
    Standard Transformer encoder as baseline for sequence
    classification task.
    """
    def __init__(self, cfgs, token2idx, label2idx,
                 pretrained_emb=None, position_enc=True):
        super(IncrementalTransformerEncoderClassification, self).__init__()
        self.cfgs = cfgs
        self.token_size = len(token2idx)
        self.label_size = len(label2idx)
        self.idx2label = {value: key for key, value in label2idx.items()}
        self.encoder = transformer.IncrementalEncoderClassification(cfgs, self.token_size,
                                                                    self.label_size,
                                                                    pretrained_emb, position_enc
                                                                    )
        self.out_proj = nn.Linear(cfgs.HIDDEN_SIZE, self.label_size)
        self.loss = LabelSmoothing(smoothing=0.1)

    def forward(self, x, valid=False, sub_mask=True):
        logits, mean_mask = self.encoder(x, valid, sub_mask)
        logits = self.out_proj(logits)
        return logits, mean_mask

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        targets = targets.expand(targets.size(0), self.cfgs.MAX_TOKEN)

        logits, mean_mask = self.forward(inputs)
        active_targets = targets[mean_mask]

        loss = self.loss(logits, active_targets)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        targets = targets.squeeze(-1)
        logits, _ = self.forward(inputs, valid=True)

        loss = self.loss(logits, targets)

        # Compute non-incremental accuracy for this batch
        # where the index is not masked.
        pred = torch.argmax(logits, dim=1)
        val_acc = (targets == pred)

        output = OrderedDict({'val_loss': loss, 'val_acc': val_acc})
        return output

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        targets = targets.squeeze(-1)
        logits, _ = self.forward(inputs, valid=True)

        loss = self.loss(logits, targets)

        pred = torch.argmax(logits, dim=1)
        test_acc = (targets == pred)

        output = OrderedDict({'test_loss': loss, 'test_acc': test_acc})
        return output

    def validation_epoch_end(self, outputs):
        val_acc = torch.cat([out['val_acc'] for out in outputs]).view(-1)
        val_acc = torch.sum(val_acc).item()/(len(val_acc) * 1.0)

        val_loss_mean = torch.stack([out['val_loss'] for out in outputs]).mean()

        values = {'val_loss_mean': val_loss_mean, 'val_acc': val_acc}
        self.log_dict(values)

    def test_epoch_end(self, outputs):
        test_acc = torch.cat([out['test_acc'] for out in outputs]).view(-1)
        test_acc = torch.sum(test_acc).item()/(len(test_acc) * 1.0)

        test_loss_mean = torch.stack([out['test_loss'] for out in outputs]).mean()

        values = {'test_loss_mean': test_loss_mean, 'test_acc': test_acc}
        self.log_dict(values)

    def configure_optimizers(self):
        optim = getattr(torch.optim, self.cfgs.OPT)
        optimizer = optim(self.parameters(), lr=self.cfgs.LR,
                          **self.cfgs.OPT_PARAMS)
        scheduler = {
            'scheduler': MultiStepLR(optimizer, milestones=self.cfgs.LR_DECAY_LIST, gamma=self.cfgs.LR_DECAY_RATE),
            'name': 'LR scheduler'
        }
        return [optimizer], [scheduler]

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx, closure, on_tpu=False,
                       using_native_amp=False, using_lbfgs=False):
        if current_epoch < self.cfgs.WARMUP_EPOCH:
            warmup_lr = (current_epoch+1)/self.cfgs.WARMUP_EPOCH * self.cfgs.LR
            lr = min(warmup_lr, self.cfgs.LR)
            for pg in optimizer.param_groups:
                pg['lr'] = lr
        optimizer.step(closure=closure)
