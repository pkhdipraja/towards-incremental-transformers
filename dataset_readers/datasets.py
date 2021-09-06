from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
from dataset_readers.data_utils import proc_seqs, proc_tags, proc_labels

import numpy as np
import en_vectors_web_lg


class Loader(object):
    def __init__(self, cfgs):
        self.cfgs = cfgs

    def load(self):
        """
        Load dataset.
        """
        if self.cfgs.TASK_TYPE == 'labelling':
            return self._load_sequence_labelling()
        else:
            return self._load_sequence_classification()

    def _load_sequence_labelling(self):
        """
        Load sequence labelling dataset, return dict of train,
        valid and test data.
        """
        data_dict = {}
        for split in self.cfgs.SPLIT:
            sentence_list = []
            tag_list = []
            split_list = self.cfgs.SPLIT[split].split('+')

            for item in split_list:
                with open(self.cfgs.DATA_PATH[self.cfgs.DATASET][item], 'r') as f:
                    sentence_iter = []
                    tag_iter = []

                    for line in f:
                        # each sentence are separated by double newline
                        if line != '\n':
                            word, label = line.split()

                            sentence_iter.append(word)
                            tag_iter.append(label)

                        else:
                            if len(sentence_iter) <= self.cfgs.MAX_TOKEN:
                                sentence_list.append(tuple(sentence_iter))
                                tag_list.append(tuple(tag_iter))

                            sentence_iter = []
                            tag_iter = []

            data_dict[split] = (sentence_list, tag_list)

        return data_dict

    def _load_sequence_classification(self):
        """
        Load sequence classification dataset, return dict of train,
        valid and test data.
        """
        data_dict = {}
        for split in self.cfgs.SPLIT:
            sentence_list = []
            label_list = []
            split_list = self.cfgs.SPLIT[split].split('+')

            for item in split_list:
                with open(self.cfgs.DATA_PATH[self.cfgs.DATASET][item], 'r') as f:
                    sentence_iter = []

                    for line in f:
                        if line[:8] == '<LABEL>:':
                            label = line.split()[1]
                        elif line != '\n':
                            sentence_iter.append(line.split()[0])
                        else:
                            if len(sentence_iter) <= self.cfgs.MAX_TOKEN:
                                sentence_list.append(tuple(sentence_iter))
                                label_list.append(label)

                            sentence_iter = []
                            label = None

            data_dict[split] = (sentence_list, label_list)

        return data_dict


class SeqTokenizer(object):
    def __init__(self, cfgs):
        self.cfgs = cfgs
        self.token2idx = {
            'PADDING': 0,
            'UNK': 1,
            'NULL': 2,  # for delayed output
        }

        self.label2idx = {
            'PADDING': 0
        }

        self.pretrained_emb = []

        if cfgs.USE_GLOVE:
            self.spacy_tool = en_vectors_web_lg.load()
            self.pretrained_emb.append(self.spacy_tool('PADDING').vector)
            self.pretrained_emb.append(self.spacy_tool('UNK').vector)
            self.pretrained_emb.append(self.spacy_tool('NULL').vector)

    def tokenize_label(self, data):
        """
        Create tokens to index, tags to index map, and embeddings
        for sequence labelling task.
        """
        for split in data:
            if split in ['train']:
                sentence_list, tag_list = data[split]

                for sentence_iter, tag_iter in zip(sentence_list, tag_list):
                    for word, label in zip(sentence_iter, tag_iter):
                        if word not in self.token2idx:
                            self.token2idx[word] = len(self.token2idx)
                            if self.cfgs.USE_GLOVE:
                                self.pretrained_emb.append(self.spacy_tool(word).vector)

                        if label not in self.label2idx:
                            self.label2idx[label] = len(self.label2idx)

            else:
                # Only get labels unseen in the training set to avoid errors
                _, tag_list = data[split]

                for tag_iter in tag_list:
                    for label in tag_iter:
                        if label not in self.label2idx:
                            self.label2idx[label] = len(self.label2idx)

        if self.cfgs.USE_GLOVE:
            self.pretrained_emb = np.array(self.pretrained_emb)

    def tokenize_classification(self, data):
        """
        Create tokens to index, labels to index map, and embeddings
        for sequence classification task.
        """
        for split in data:
            if split in ['train']:
                sentence_list, label_list = data[split]

                for sentence_iter, label_iter in zip(sentence_list, label_list):
                    for word in sentence_iter:
                        if word not in self.token2idx:
                            self.token2idx[word] = len(self.token2idx)
                            if self.cfgs.USE_GLOVE:
                                self.pretrained_emb.append(self.spacy_tool(word).vector)

                    label = label_iter
                    if label not in self.label2idx:
                        self.label2idx[label] = len(self.label2idx)

            else:
                # Only get labels unseen in the training set to avoid errors
                _, label_list = data[split]

                for label_iter in label_list:
                    label = label_iter

                    if label not in self.label2idx:
                        self.label2idx[label] = len(self.label2idx)

        if self.cfgs.USE_GLOVE:
            self.pretrained_emb = np.array(self.pretrained_emb)


class SeqLabellingDataset(Dataset):
    """
    Dataset object for sequence labelling.
    """
    def __init__(self, cfgs, data, tokenizer, train=True):
        super(SeqLabellingDataset, self).__init__()
        self.cfgs = cfgs
        self.train = train
        self.sequence_list, self.tag_list = data
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        sentence_iter = self.sequence_list[idx]
        tag_iter = self.tag_list[idx]

        sent_tensor_iter = proc_seqs(sentence_iter, self.tokenizer.token2idx,
                                     self.cfgs.MAX_TOKEN, self.train,
                                     self.cfgs.UNK_PROB)

        tag_tensor_iter = proc_tags(tag_iter, self.tokenizer.label2idx,
                                    self.cfgs.MAX_TOKEN)

        return sent_tensor_iter, tag_tensor_iter

    def __len__(self):
        return self.sequence_list.__len__()


class SeqClassificationDataset(Dataset):
    """
    Dataset object for sequence classification.
    """
    def __init__(self, cfgs, data, tokenizer, train=True):
        super(SeqClassificationDataset, self).__init__()
        self.cfgs = cfgs
        self.train = train
        self.sequence_list, self.label_list = data
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        sentence_iter = self.sequence_list[idx]
        label_iter = self.label_list[idx]

        sent_tensor_iter = proc_seqs(sentence_iter, self.tokenizer.token2idx,
                                     self.cfgs.MAX_TOKEN, self.train,
                                     self.cfgs.UNK_PROB)

        label_tensor_iter = proc_labels(label_iter, self.tokenizer.label2idx)

        return sent_tensor_iter, label_tensor_iter

    def __len__(self):
        return self.sequence_list.__len__()


class SeqLabellingDataModule(LightningDataModule):
    """
    Data module for sequence labelling.
    """
    def __init__(self, cfgs, valid=False):
        super(SeqLabellingDataModule, self).__init__()
        self.cfgs = cfgs
        self.data_loader = Loader(cfgs)
        self.tokenizer = SeqTokenizer(cfgs)
        self.data = self.data_loader.load()
        self.valid = valid

    def prepare_data(self):
        self.tokenizer.tokenize_label(self.data)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_set = SeqLabellingDataset(self.cfgs, self.data['train'],
                                                 self.tokenizer, train=True)

            self.valid_set = SeqLabellingDataset(self.cfgs, self.data['valid'],
                                                 self.tokenizer, train=False)

        if stage == 'test' or stage is None:
            if self.valid:
                self.test_set = SeqLabellingDataset(self.cfgs, self.data['valid'],
                                                    self.tokenizer, train=False)
            else:
                self.test_set = SeqLabellingDataset(self.cfgs, self.data['test'],
                                                    self.tokenizer, train=False)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.cfgs.BATCH_SIZE,
                          shuffle=True, num_workers=self.cfgs.NUM_WORKERS,
                          pin_memory=self.cfgs.PIN_MEM)

    def val_dataloader(self):
        return DataLoader(self.valid_set, batch_size=1,
                          shuffle=False, num_workers=self.cfgs.NUM_WORKERS,
                          pin_memory=self.cfgs.PIN_MEM)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=1,
                          shuffle=False, num_workers=self.cfgs.NUM_WORKERS,
                          pin_memory=self.cfgs.PIN_MEM)

    def embedding(self):
        """
        Return GloVe embeddings
        """
        return self.tokenizer.pretrained_emb


class SeqClassificationDataModule(LightningDataModule):
    """
    Data module for sequence classification.
    """
    def __init__(self, cfgs, valid=False):
        super(SeqClassificationDataModule, self).__init__()
        self.cfgs = cfgs
        self.data_loader = Loader(cfgs)
        self.tokenizer = SeqTokenizer(cfgs)
        self.data = self.data_loader.load()
        self.valid = valid

    def prepare_data(self):
        self.tokenizer.tokenize_classification(self.data)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_set = SeqClassificationDataset(self.cfgs, self.data['train'],
                                                      self.tokenizer, train=True)

            self.valid_set = SeqClassificationDataset(self.cfgs, self.data['valid'],
                                                      self.tokenizer, train=False)

        if stage == 'test' or stage is None:
            if self.valid:
                self.test_set = SeqClassificationDataset(self.cfgs, self.data['valid'],
                                                         self.tokenizer, train=False)
            else:
                self.test_set = SeqClassificationDataset(self.cfgs, self.data['test'],
                                                         self.tokenizer, train=False)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.cfgs.BATCH_SIZE,
                          shuffle=True, num_workers=self.cfgs.NUM_WORKERS,
                          pin_memory=self.cfgs.PIN_MEM)

    def val_dataloader(self):
        return DataLoader(self.valid_set, batch_size=1,
                          shuffle=False, num_workers=self.cfgs.NUM_WORKERS,
                          pin_memory=self.cfgs.PIN_MEM)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=1,
                          shuffle=False, num_workers=self.cfgs.NUM_WORKERS,
                          pin_memory=self.cfgs.PIN_MEM)
