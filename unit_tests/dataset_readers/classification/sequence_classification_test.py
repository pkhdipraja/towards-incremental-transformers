from unit_tests.dataset_readers.classification.conftest import DummyConfig, dummy_tokenizer, expected_data
from dataset_readers.datasets import SeqClassificationDataset, Loader, SeqTokenizer

import pytest
import torch


@pytest.mark.parametrize('dataset', ('atis-intent', 'proscons', 'sent-negpos', 'snips-intent'))
def test_loader(dataset, expected_data):
    cfgs = DummyConfig()
    cfgs.DATASET = dataset

    expected_sents = expected_data[dataset]['sentences']
    expected_label = expected_data[dataset]['label']

    data_loader = Loader(cfgs)
    data = data_loader.load()
    sentence_list, label_list = data['train']

    for sents, gold_sents in zip(sentence_list, expected_sents):
        assert sents == gold_sents, "Sentences does not match expected values."

    for label, gold_label in zip(label_list, expected_label):
        assert label == gold_label, "Labels does not match expected values."


def test_tokenizer(dummy_tokenizer):
    cfgs = DummyConfig()
    dummy_data, expected_token2idx, expected_label2idx, _ = dummy_tokenizer

    tokenizer = SeqTokenizer(cfgs)
    tokenizer.tokenize_classification(dummy_data)

    token2idx = tokenizer.token2idx
    label2idx = tokenizer.label2idx

    # ensure that vocab is only added from train, however ensure that 
    # label from val/test is also added.
    assert token2idx == expected_token2idx, "Token to index map does not match expected values."
    assert label2idx == expected_label2idx, "Label to index map does not match expected values."


def test_dataset(dummy_tokenizer):
    cfgs = DummyConfig()
    dummy_data, _, _, dummy_data_tensor = dummy_tokenizer

    tokenizer = SeqTokenizer(cfgs)
    tokenizer.tokenize_classification(dummy_data)

    train_set = SeqClassificationDataset(cfgs, dummy_data['train'], tokenizer,
                                         train=True)

    test_set = SeqClassificationDataset(cfgs, dummy_data['test'], tokenizer,
                                        train=False)

    train_sent_iter, train_label_iter = train_set[0]
    test_sent_iter, test_label_iter = test_set[0]

    gold_train_sent_iter, gold_train_label_iter = dummy_data_tensor['train']
    gold_test_sent_iter, gold_test_label_iter = dummy_data_tensor['test']

    # ensure that tensor of sequences and label containing the idx 
    # is also created similar to token2idx and label2idx map
    assert torch.equal(train_sent_iter, gold_train_sent_iter), "Sentence tensor mismatch."
    assert torch.equal(train_label_iter, gold_train_label_iter), "Label tensor mismatch."
    assert torch.equal(test_sent_iter, gold_test_sent_iter), "Sentence tensor mismatch."
    assert torch.equal(test_label_iter, gold_test_label_iter), "Label tensor mismatch."