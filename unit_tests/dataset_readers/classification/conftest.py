import torch
import pytest
import os


class DummyConfig(object):
    def __init__(self):
        self.DATASET = None
        self.TASK_TYPE = 'classification'

        self.SPLIT = {
            'train': 'train'
        }

        self.PATH_PREFIX_TAGS = 'unit_tests/dataset_readers/sample_test/classification'

        self.DATA_PATH = {
            'atis-intent': {
                'train': os.path.join(os.getcwd(), self.PATH_PREFIX_TAGS, 'sample.atis_intent')
            },
            'proscons': {
                'train': os.path.join(os.getcwd(), self.PATH_PREFIX_TAGS, 'sample.proscons')
            },
            'sent-negpos': {
                'train': os.path.join(os.getcwd(), self.PATH_PREFIX_TAGS, 'sample.sent_negpos')
            },
            'snips-intent': {
                'train': os.path.join(os.getcwd(), self.PATH_PREFIX_TAGS, 'sample.snips_intent')
            }
        }

        self.MAX_TOKEN = 60
        self.USE_GLOVE = True
        self.UNK_PROB = 0.0


@pytest.fixture
def expected_data():
    expected_dict = {
        'atis-intent': {
            'sentences': [
                ('i', 'would', 'like', 'to', 'find', 'a', 'flight', 'from', 'charlotte',
                 'to', 'las', 'vegas', 'that', 'makes', 'a', 'stop', 'in', 'st.', 'louis'),
                ('on', 'april', 'first', 'i', 'need', 'a', 'ticket', 'from', 'tacoma', 'to',
                 'san', 'jose', 'departing', 'before', '7', 'am')

            ],
            'label': [
                'atis_flight', 'atis_airfare'
            ]
        },
        'proscons': {
            'sentences': [
                ('high', 'resolution', ',', 'excellent', 'features', ',', 'compact', 'size'),
                ('Almost', 'too', 'fast', '.', 'Occasionally', 'feeds', 'paper', 'wrong', '.')
            ],
            'label': [
                'pro', 'con'
            ]
        },
        'sent-negpos': {
            'sentences': [
                ('It', 'worked', 'very', 'well', '.'),
                ('The', 'black', 'eyed', 'peas', 'and', 'sweet', 'potatoes', '...', 'UNREAL', '!')
            ],
            'label': [
                '1', '1'
            ]
        },
        'snips-intent': {
            'sentences': [
                ('add', 'sabrina', 'salerno', 'to', 'the', 'grime', 'instrumentals', 'playlist'),
                ('i', 'want', 'to', 'bring', 'four', 'people', 'to', 'a', 'place', 'that', 's',
                 'close', 'to', 'downtown', 'that', 'serves', 'churrascaria', 'cuisine')
            ],
            'label': [
                'AddToPlaylist', 'BookRestaurant'
            ]
        }
    }

    return expected_dict


@pytest.fixture
def dummy_tokenizer():
    dummy_data = {
        'train': (
            [('add', 'sabrina', 'salerno', 'to', 'the', 'grime', 'instrumentals', 'playlist')],
            ['AddToPlaylist']
        ),
        'test': (
            [('i', 'want', 'to', 'bring', 'four', 'people', 'to', 'a', 'place', 'that', 's',
              'close', 'to', 'downtown', 'that', 'serves', 'churrascaria', 'cuisine')],
            ['BookRestaurant']
        )
    }

    token2idx = {
        'PADDING': 0, 'UNK': 1, 'NULL': 2, 'add': 3, 'sabrina': 4, 'salerno': 5, 'to': 6, 'the': 7,
        'grime': 8, 'instrumentals': 9, 'playlist': 10
    }

    label2idx = {
        'PADDING': 0, 'AddToPlaylist': 1, 'BookRestaurant': 2
    }

    dummy_data_tensor = {
        'train': [
            torch.cat((torch.tensor([3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.long),
                       torch.zeros(52, dtype=torch.long))),
            torch.tensor([1], dtype=torch.long)
        ],
        'test': [
            torch.cat((torch.tensor([1, 1, 6, 1, 1, 1, 6, 1, 1, 
                                     1, 1, 1, 6, 1, 1, 1, 1, 1], dtype=torch.long),
                       torch.zeros(42, dtype=torch.long))),
            torch.tensor([2], dtype=torch.long)
        ]
    }

    return dummy_data, token2idx, label2idx, dummy_data_tensor
