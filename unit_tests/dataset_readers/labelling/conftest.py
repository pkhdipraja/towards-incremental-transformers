import torch
import pytest
import os


class DummyConfig(object):
    def __init__(self):
        self.DATASET = None
        self.TASK_TYPE = 'labelling'

        self.SPLIT = {
            'train': 'train'
        }

        self.PATH_PREFIX_TAGS = 'unit_tests/dataset_readers/sample_test/labelling'

        self.DATA_PATH = {
            'atis-slot': {
                'train': os.path.join(os.getcwd(), self.PATH_PREFIX_TAGS, 'sample.atis_slot')
            },
            'ner-nw-wsj': {
                'train': os.path.join(os.getcwd(), self.PATH_PREFIX_TAGS, 'sample.ner_nw_wsj')
            },
            'chunk': {
                'train': os.path.join(os.getcwd(), self.PATH_PREFIX_TAGS, 'sample.chunk')
            },
            'pos-nw-wsj': {
                'train': os.path.join(os.getcwd(), self.PATH_PREFIX_TAGS, 'sample.pos_nw_wsj')
            },
            'snips-slot': {
                'train': os.path.join(os.getcwd(), self.PATH_PREFIX_TAGS, 'sample.snips_slot')
            },
            'srl-nw-wsj': {
                'train': os.path.join(os.getcwd(), self.PATH_PREFIX_TAGS, 'sample.srl_nw_wsj')
            }
        }

        self.MAX_TOKEN = 60
        self.USE_GLOVE = True
        self.UNK_PROB = 0.0


@pytest.fixture
def expected_data():
    expected_dict = {
        'atis-slot': {
            'sentences': [
                ('i', 'would', 'like', 'to', 'find', 'a', 'flight', 'from', 'charlotte',
                 'to', 'las', 'vegas', 'that', 'makes', 'a', 'stop', 'in', 'st.', 'louis'),
                ('on', 'april', 'first', 'i', 'need', 'a', 'ticket', 'from', 'tacoma', 'to',
                 'san', 'jose', 'departing', 'before', '7', 'am')

            ],
            'tags': [
                ('O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-fromloc.city_name', 'O', 'B-toloc.city_name',
                 'I-toloc.city_name', 'O', 'O', 'O', 'O', 'O', 'B-stoploc.city_name', 'I-stoploc.city_name'),
                ('O', 'B-depart_date.month_name', 'B-depart_date.day_number', 'O', 'O', 'O', 'O', 'O',
                 'B-fromloc.city_name', 'O', 'B-toloc.city_name', 'I-toloc.city_name', 'O', 'B-depart_time.time_relative',
                 'B-depart_time.time', 'I-depart_time.time')
            ]
        },
        'ner-nw-wsj': {
            'sentences': [
                ('Kenneth', 'J.', 'Thygerson', ',', 'who', 'was', 'named', 'president', 'of', 'this', 'thrift',
                 'holding', 'company', 'in', 'August', ',', 'resigned', ',', 'citing', 'personal', 'reasons', '.'),
                ('Mr.', 'Thygerson', 'said', 'he', 'had', 'planned', 'to', 'travel', 'between', 'the', 'job', 'in',
                 'Denver', 'and', 'his', 'San', 'Diego', 'home', ',', 'but', 'has', 'found', 'the', 'commute', 'too',
                 'difficult', 'to', 'continue', '.')
            ],
            'tags': [
                ('B-PERSON', 'I-PERSON', 'I-PERSON', 'O', 'O', 'O', 'O', 'O', 'O',
                 'O', 'O', 'O', 'O', 'O', 'B-DATE', 'O', 'O', 'O', 'O', 'O', 'O', 'O'),
                ('O', 'B-PERSON', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O','O', 'B-GPE', 'O', 'O',
                 'B-GPE', 'I-GPE', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O')

            ]
        },
        'chunk': {
            'sentences': [
                ('Rockwell', 'International', 'Corp.', '\'s', 'Tulsa', 'unit', 'said', 'it', 'signed', 'a', 'tentative',
                 'agreement', 'extending', 'its', 'contract', 'with', 'Boeing', 'Co.', 'to', 'provide', 'structural',
                 'parts', 'for', 'Boeing', '\'s', '747', 'jetliners', '.'),
                ('Rockwell', 'said', 'the', 'agreement', 'calls', 'for', 'it', 'to', 'supply', '200', 'additional',
                 'so-called', 'shipsets', 'for', 'the', 'planes', '.')
            ],
            'tags': [
                ('B-NP', 'I-NP', 'I-NP', 'B-NP', 'I-NP', 'I-NP', 'B-VP', 'B-NP', 'B-VP', 'B-NP',
                 'I-NP', 'I-NP', 'B-VP', 'B-NP', 'I-NP', 'B-PP', 'B-NP', 'I-NP', 'B-VP', 'I-VP',
                 'B-NP', 'I-NP', 'B-PP', 'B-NP', 'B-NP', 'I-NP', 'I-NP', 'O'),
                ('B-NP', 'B-VP', 'B-NP', 'I-NP', 'B-VP', 'B-SBAR', 'B-NP', 'B-VP', 'I-VP',
                 'B-NP', 'I-NP', 'I-NP', 'I-NP', 'B-PP', 'B-NP', 'I-NP', 'O')
            ]
        },
        'pos-nw-wsj': {
            'sentences': [
                ('Kenneth', 'J.', 'Thygerson', ',', 'who', 'was', 'named', 'president', 'of', 'this', 'thrift',
                 'holding', 'company', 'in', 'August', ',', 'resigned', ',', 'citing', 'personal', 'reasons', '.'),
                ('Mr.', 'Thygerson', 'said', 'he', 'had', 'planned', 'to', 'travel', 'between', 'the', 'job', 'in',
                 'Denver', 'and', 'his', 'San', 'Diego', 'home', ',', 'but', 'has', 'found', 'the', 'commute', 'too',
                 'difficult', 'to', 'continue', '.')
            ],
            'tags': [
                ('NNP', 'NNP', 'NNP', ',', 'WP', 'VBD', 'VBN', 'NN', 'IN', 'DT', 'NN',
                 'VBG', 'NN', 'IN', 'NNP', ',', 'VBD', ',', 'VBG', 'JJ', 'NNS', '.'),
                ('NNP', 'NNP', 'VBD', 'PRP', 'VBD', 'VBN', 'TO', 'VB', 'IN', 'DT', 'NN', 'IN', 'NNP', 'CC', 
                 'PRP$', 'NNP', 'NNP', 'NN', ',', 'CC', 'VBZ', 'VBN', 'DT', 'NN', 'RB', 'JJ', 'TO', 'VB', '.')
            ]
        },
        'snips-slot': {
            'sentences': [
                ('add', 'sabrina', 'salerno', 'to', 'the', 'grime', 'instrumentals', 'playlist'),
                ('i', 'want', 'to', 'bring', 'four', 'people', 'to', 'a', 'place', 'that', 's', 
                 'close', 'to', 'downtown', 'that', 'serves', 'churrascaria', 'cuisine')
            ],
            'tags': [
                ('O', 'B-artist', 'I-artist', 'O', 'O', 'B-playlist', 'I-playlist', 'O'),
                ('O', 'O', 'O', 'O', 'B-party_size_number', 'O', 'O', 'O', 'O', 'O',
                 'O', 'B-spatial_relation', 'O', 'B-poi', 'O', 'O', 'B-restaurant_type', 'O')
            ]
        },
        'srl-nw-wsj': {
            'sentences': [
                ('Kenneth', 'J.', 'Thygerson', ',', 'who', 'was', 'named', 'president', 'of', 'this', 'thrift',
                 'holding', 'company', 'in', 'August', ',', 'resigned', ',', 'citing', 'personal', 'reasons', '.'),
                ('Kenneth', 'J.', 'Thygerson', ',', 'who', 'was', 'named', 'president', 'of', 'this', 'thrift',
                 'holding', 'company', 'in', 'August', ',', 'resigned', ',', 'citing', 'personal', 'reasons', '.')
            ],
            'tags': [
                ('B-ARG1', 'I-ARG1', 'I-ARG1', 'O', 'B-R-ARG1', 'O', 'B-V', 'B-ARG2', 'I-ARG2', 'I-ARG2', 'I-ARG2',
                 'I-ARG2', 'I-ARG2', 'B-ARGM-TMP', 'I-ARGM-TMP', 'O', 'O', 'O', 'O', 'O', 'O', 'O'),
                ('B-ARG0', 'I-ARG0', 'I-ARG0', 'I-ARG0', 'I-ARG0', 'I-ARG0', 'I-ARG0', 'I-ARG0', 'I-ARG0', 'I-ARG0', 'I-ARG0',
                 'I-ARG0', 'I-ARG0', 'I-ARG0', 'I-ARG0', 'I-ARG0', 'B-V', 'O', 'B-ARGM-ADV', 'I-ARGM-ADV', 'I-ARGM-ADV', 'O')
            ]
        }
    }

    return expected_dict


@pytest.fixture
def dummy_tokenizer():
    dummy_data = {
        'train': (
            [('add', 'sabrina', 'salerno', 'to', 'the', 'grime', 'instrumentals', 'playlist')],
            [('O', 'B-artist', 'I-artist', 'O', 'O', 'B-playlist', 'I-playlist', 'O')]
        ),
        'test': (
            [('what', 's', 'the', 'weather', 'here', 'on', '2/7/2021')],
            [('O', 'O', 'O', 'O', 'B-current_location', 'O', 'B-timeRange')]
        )
    }

    token2idx = {
        'PADDING': 0, 'UNK': 1, 'NULL': 2, 'add': 3, 'sabrina': 4, 'salerno': 5, 'to': 6, 'the': 7,
        'grime': 8, 'instrumentals': 9, 'playlist': 10
    }

    label2idx = {
        'PADDING': 0, 'O': 1, 'B-artist': 2, 'I-artist': 3, 'B-playlist': 4, 'I-playlist': 5,
        'B-current_location': 6, 'B-timeRange': 7
    }

    dummy_data_tensor = {
        'train': [
            torch.cat((torch.tensor([3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.long),
                       torch.zeros(52, dtype=torch.long))),
            torch.cat((torch.tensor([1, 2, 3, 1, 1, 4, 5, 1], dtype=torch.long),
                       torch.zeros(52, dtype=torch.long)))
        ],
        'test': [
            torch.cat((torch.tensor([1, 1, 7, 1, 1, 1, 1], dtype=torch.long),
                       torch.zeros(53, dtype=torch.long))),
            torch.cat((torch.tensor([1, 1, 1, 1, 6, 1, 7], dtype=torch.long),
                       torch.zeros(53, dtype=torch.long))),
        ]
    }

    return dummy_data, token2idx, label2idx, dummy_data_tensor
