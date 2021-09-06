import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model_utils import PositionalEncoding, RecurrentPositionalEncoding, make_mask
from fast_transformers.builders import RecurrentEncoderBuilder, TransformerEncoderBuilder
from fast_transformers.masking import LengthMask, TriangularCausalMask, FullMask
from fast_transformers.utils import make_mirror


class LinearEncoderLabelling(nn.Module):
    """
    N stack of Linear Transformers for sequence labelling.
    """
    def __init__(self, cfgs, token_size, label_size,
                 pretrained_emb=None, position_enc=True):
        super(LinearEncoderLabelling, self).__init__()
        self.cfgs = cfgs
        assert cfgs.HIDDEN_SIZE % cfgs.ATTENTION_HEAD == 0

        # d_query = d_key = d_value
        self.HIDDEN_SIZE_HEAD = cfgs.HIDDEN_SIZE // cfgs.ATTENTION_HEAD

        self.params = {
                        'attention_type': 'linear',
                        'n_layers': cfgs.LAYER,
                        'n_heads': cfgs.ATTENTION_HEAD,
                        'feed_forward_dimensions': cfgs.FF_SIZE,
                        'query_dimensions': self.HIDDEN_SIZE_HEAD,
                        'value_dimensions': self.HIDDEN_SIZE_HEAD,
                        'dropout': cfgs.DROPOUT,
                        'attention_dropout': cfgs.DROPOUT,
                        'activation': 'relu',
                        'final_normalization': True
                    }
        self.encoder = TransformerEncoderBuilder.from_dictionary(self.params
                                                                 ).get()

        self.position_enc = position_enc

        if cfgs.USE_GLOVE:
            self.embedding = nn.Embedding(
                num_embeddings=token_size,
                embedding_dim=cfgs.WORD_EMBED_SIZE
            )
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))
            self.proj = nn.Linear(cfgs.WORD_EMBED_SIZE,
                                  cfgs.HIDDEN_SIZE)
        else:
            self.embedding = nn.Embedding(
                num_embeddings=token_size,
                embedding_dim=cfgs.HIDDEN_SIZE
            )

        if cfgs.DATASET == 'srl-nw-wsj':
            self.pred_embedding = nn.Embedding(
                num_embeddings=2,  # True or False
                embedding_dim=cfgs.PRED_EMBED_SIZE
            )
            self.pred_proj = nn.Linear(cfgs.HIDDEN_SIZE + cfgs.PRED_EMBED_SIZE,
                                       cfgs.HIDDEN_SIZE)  # To mix information about binary label into the word embedding itself

        if position_enc:
            self.position = PositionalEncoding(cfgs)

        # self.out_proj = nn.Linear(cfgs.HIDDEN_SIZE,
        #                           label_size)

        # Xavier init
        for param in self.encoder.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, x, pred_mask=None):
            length_mask = LengthMask(torch.sum(torch.abs(x) != 0, dim=-1),
                                     max_len=x.size(-1))
            attn_mask = None  # automatically attend to all tokens

            x = self.embedding(x)
            if self.cfgs.USE_GLOVE:
                x = self.proj(x)

            if self.position_enc:
                x = self.position(x)

            if self.cfgs.DATASET == 'srl-nw-wsj':
                pred_embeds = self.pred_embedding(pred_mask)
                x = torch.cat((x, pred_embeds), dim=2)
                x = self.pred_proj(x)

            x = self.encoder(x,
                             attn_mask=attn_mask,
                             length_mask=length_mask
                             )
            # x = self.out_proj(x)
            return x


class LinearCausalEncoderLabelling(nn.Module):
    """
    N stack of Linear Transformers with causal masking for
    sequence labelling.
    """
    def __init__(self, cfgs, token_size, label_size,
                 pretrained_emb=None, position_enc=True):
        super(LinearCausalEncoderLabelling, self).__init__()
        self.cfgs = cfgs
        assert cfgs.HIDDEN_SIZE % cfgs.ATTENTION_HEAD == 0

        # d_query = d_key = d_value
        self.HIDDEN_SIZE_HEAD = cfgs.HIDDEN_SIZE // cfgs.ATTENTION_HEAD

        self.params = {
                        'attention_type': 'causal-linear',
                        'n_layers': cfgs.LAYER,
                        'n_heads': cfgs.ATTENTION_HEAD,
                        'feed_forward_dimensions': cfgs.FF_SIZE,
                        'query_dimensions': self.HIDDEN_SIZE_HEAD,
                        'value_dimensions': self.HIDDEN_SIZE_HEAD,
                        'dropout': cfgs.DROPOUT,
                        'attention_dropout': cfgs.DROPOUT,
                        'activation': 'relu',
                        'final_normalization': True
                    }
        self.encoder = TransformerEncoderBuilder.from_dictionary(self.params
                                                                 ).get()

        self.position_enc = position_enc

        if cfgs.USE_GLOVE:
            self.embedding = nn.Embedding(
                num_embeddings=token_size,
                embedding_dim=cfgs.WORD_EMBED_SIZE
            )
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))
            self.proj = nn.Linear(cfgs.WORD_EMBED_SIZE,
                                  cfgs.HIDDEN_SIZE)
        else:
            self.embedding = nn.Embedding(
                num_embeddings=token_size,
                embedding_dim=cfgs.HIDDEN_SIZE
            )

        if cfgs.DATASET == 'srl-nw-wsj':
            self.pred_embedding = nn.Embedding(
                num_embeddings=2,  # True or False
                embedding_dim=cfgs.PRED_EMBED_SIZE
            )
            self.pred_proj = nn.Linear(cfgs.HIDDEN_SIZE + cfgs.PRED_EMBED_SIZE,
                                       cfgs.HIDDEN_SIZE)  # To mix information about binary label into the word embedding itself

        if position_enc:
            self.position = PositionalEncoding(cfgs)

        # self.out_proj = nn.Linear(cfgs.HIDDEN_SIZE,
        #                           label_size)

        # Xavier init
        for param in self.encoder.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, x, pred_mask=None):
        length_mask = LengthMask(torch.sum(torch.abs(x) != 0, dim=-1),
                                 max_len=self.cfgs.MAX_TOKEN)
        attn_mask = TriangularCausalMask(self.cfgs.MAX_TOKEN)

        x = self.embedding(x)
        if self.cfgs.USE_GLOVE:
            x = self.proj(x)

        if self.position_enc:
            x = self.position(x)

        if self.cfgs.DATASET == 'srl-nw-wsj':
            pred_embeds = self.pred_embedding(pred_mask)
            x = torch.cat((x, pred_embeds), dim=2)
            x = self.pred_proj(x)

        x = self.encoder(x,
                         attn_mask=attn_mask,
                         length_mask=length_mask
                         )
        # x = self.out_proj(x)
        return x


class RecurrentEncoderLabelling(nn.Module):
    """
    N stack of Linear Transformers as RNN for
    sequence labelling.
    """
    def __init__(self, cfgs, token_size, label_size,
                 pretrained_emb=None, position_enc=True):
        super(RecurrentEncoderLabelling, self).__init__()
        self.cfgs = cfgs
        assert cfgs.HIDDEN_SIZE % cfgs.ATTENTION_HEAD == 0

        # d_query = d_key = d_value
        self.HIDDEN_SIZE_HEAD = cfgs.HIDDEN_SIZE // cfgs.ATTENTION_HEAD

        self.params = {
                        'attention_type': 'linear',
                        'n_layers': cfgs.LAYER,
                        'n_heads': cfgs.ATTENTION_HEAD,
                        'feed_forward_dimensions': cfgs.FF_SIZE,
                        'query_dimensions': self.HIDDEN_SIZE_HEAD,
                        'value_dimensions': self.HIDDEN_SIZE_HEAD,
                        'dropout': cfgs.DROPOUT,
                        'attention_dropout': cfgs.DROPOUT,
                        'activation': 'relu',
                        'final_normalization': True
                    }

        self.encoder = RecurrentEncoderBuilder.from_dictionary(self.params
                                                               ).get()

        self.position_enc = position_enc

        if cfgs.USE_GLOVE:
            self.embedding = nn.Embedding(
                num_embeddings=token_size,
                embedding_dim=cfgs.WORD_EMBED_SIZE
            )
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))
            self.proj = nn.Linear(cfgs.WORD_EMBED_SIZE,
                                  cfgs.HIDDEN_SIZE)
        else:
            self.embedding = nn.Embedding(
                num_embeddings=token_size,
                embedding_dim=cfgs.HIDDEN_SIZE
            )

        if position_enc:
            self.position = RecurrentPositionalEncoding(cfgs)

        # self.out_proj = nn.Linear(cfgs.HIDDEN_SIZE,
        #                           label_size)

        # Xavier init
        for param in self.encoder.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, x, i=0, memory=None):
        x = self.embedding(x)
        if self.cfgs.USE_GLOVE:
            x = self.proj(x)
        if self.position_enc:
            x = self.position(x, i)
        x, memory = self.encoder(x.squeeze(dim=1), memory)
        # x = self.out_proj(x)
        return x, memory


class LinearEncoderClassification(nn.Module):
    """
    N stack of Linear Transformers for sequence classification.
    """
    def __init__(self, cfgs, token_size, label_size,
                 pretrained_emb=None, position_enc=True):
        super(LinearEncoderClassification, self).__init__()
        self.cfgs = cfgs
        assert cfgs.HIDDEN_SIZE % cfgs.ATTENTION_HEAD == 0

        # d_query = d_key = d_value
        self.HIDDEN_SIZE_HEAD = cfgs.HIDDEN_SIZE // cfgs.ATTENTION_HEAD

        self.params = {
                        'attention_type': 'linear',
                        'n_layers': cfgs.LAYER,
                        'n_heads': cfgs.ATTENTION_HEAD,
                        'feed_forward_dimensions': cfgs.FF_SIZE,
                        'query_dimensions': self.HIDDEN_SIZE_HEAD,
                        'value_dimensions': self.HIDDEN_SIZE_HEAD,
                        'dropout': cfgs.DROPOUT,
                        'attention_dropout': cfgs.DROPOUT,
                        'activation': 'relu',
                        'final_normalization': True
                    }
        self.encoder = TransformerEncoderBuilder.from_dictionary(self.params
                                                                 ).get()

        self.position_enc = position_enc

        if cfgs.USE_GLOVE:
            self.embedding = nn.Embedding(
                num_embeddings=token_size,
                embedding_dim=cfgs.WORD_EMBED_SIZE
            )
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))
            self.proj = nn.Linear(cfgs.WORD_EMBED_SIZE,
                                  cfgs.HIDDEN_SIZE)
        else:
            self.embedding = nn.Embedding(
                num_embeddings=token_size,
                embedding_dim=cfgs.HIDDEN_SIZE
            )

        if position_enc:
            self.position = PositionalEncoding(cfgs)

        # self.out_proj = nn.Linear(cfgs.HIDDEN_SIZE,
        #                           label_size)

        # Xavier init
        for param in self.encoder.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

        # Needed by LinearCausalEncoderClassification for test time.
        increment_mask = torch.tril(torch.ones(self.cfgs.MAX_TOKEN,
                                               self.cfgs.MAX_TOKEN,
                                               dtype=torch.bool)).unsqueeze(0).unsqueeze(3)
        sum_mask = torch.arange(1, self.cfgs.MAX_TOKEN+1).unsqueeze(1)
        self.register_buffer('increment_mask', increment_mask)
        self.register_buffer('sum_mask', sum_mask)

    def forward(self, x):
        length_mask = LengthMask(torch.sum(torch.abs(x) != 0, dim=-1),
                                 max_len=self.cfgs.MAX_TOKEN)
        attn_mask = None  # automatically attend to all tokens
        active_mask = (torch.sum(
                torch.abs(x.unsqueeze(2)), dim=-1
                ) == 0).unsqueeze(2)

        x = self.embedding(x)
        if self.cfgs.USE_GLOVE:
            x = self.proj(x)

        if self.position_enc:
            x = self.position(x)
        x = self.encoder(x,
                         attn_mask=attn_mask,
                         length_mask=length_mask
                         )

        # Apply mask to outputs and take average of hidden layer
        masked_x = x.masked_fill(active_mask, 0)
        mask_sum = torch.logical_not(active_mask).sum(1)
        masked_x_sum = masked_x.sum(1)

        avg_x = masked_x_sum/mask_sum

        # x = self.out_proj(avg_x)
        return avg_x


class LinearCausalEncoderClassification(nn.Module):
    """
    N stack of Linear Transformers with causal masking for
    sequence classification.
    """
    def __init__(self, cfgs, token_size, label_size,
                 pretrained_emb=None, position_enc=True):
        super(LinearCausalEncoderClassification, self).__init__()
        self.cfgs = cfgs
        assert cfgs.HIDDEN_SIZE % cfgs.ATTENTION_HEAD == 0

        # d_query = d_key = d_value
        self.HIDDEN_SIZE_HEAD = cfgs.HIDDEN_SIZE // cfgs.ATTENTION_HEAD

        self.params = {
                        'attention_type': 'causal-linear',
                        'n_layers': cfgs.LAYER,
                        'n_heads': cfgs.ATTENTION_HEAD,
                        'feed_forward_dimensions': cfgs.FF_SIZE,
                        'query_dimensions': self.HIDDEN_SIZE_HEAD,
                        'value_dimensions': self.HIDDEN_SIZE_HEAD,
                        'dropout': cfgs.DROPOUT,
                        'attention_dropout': cfgs.DROPOUT,
                        'activation': 'relu',
                        'final_normalization': True
                    }
        self.encoder = TransformerEncoderBuilder.from_dictionary(self.params
                                                                 ).get()

        self.position_enc = position_enc

        if cfgs.USE_GLOVE:
            self.embedding = nn.Embedding(
                num_embeddings=token_size,
                embedding_dim=cfgs.WORD_EMBED_SIZE
            )
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))
            self.proj = nn.Linear(cfgs.WORD_EMBED_SIZE,
                                  cfgs.HIDDEN_SIZE)
        else:
            self.embedding = nn.Embedding(
                num_embeddings=token_size,
                embedding_dim=cfgs.HIDDEN_SIZE
            )

        if position_enc:
            self.position = PositionalEncoding(cfgs)

        # self.out_proj = nn.Linear(cfgs.HIDDEN_SIZE,
        #                           label_size)

        # Xavier init
        for param in self.encoder.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

        increment_mask = torch.tril(torch.ones(self.cfgs.MAX_TOKEN,
                                               self.cfgs.MAX_TOKEN,
                                               dtype=torch.bool)).unsqueeze(0).unsqueeze(3)
        sum_mask = torch.arange(1, self.cfgs.MAX_TOKEN+1).unsqueeze(1)
        self.register_buffer('increment_mask', increment_mask)
        self.register_buffer('sum_mask', sum_mask)

    def incremental_mean(self, x, mask):
        # Assuming x is of (batch_size, max_seq_len, hidden_size)
        # Expand new dimension, where each element are masked incrementally
        x = x.unsqueeze(1).expand(-1, self.cfgs.MAX_TOKEN, -1, -1)
        increment_mask_current = torch.logical_not(
            self.increment_mask.clone().detach().requires_grad_(False)
        )
        x = x.masked_fill(increment_mask_current, 0)

        # Mask for active element in the sequence
        active_seq_mask = mask.unsqueeze(2)
        x = x.masked_fill(active_seq_mask, 0)

        # Sum each element in an incremental manner
        x_sum = x.sum(2)

        # Mask for element counts and fill
        # sum_mask_curr = self.sum_mask * torch.logical_not(mask)
        sum_mask_curr = self.sum_mask.masked_fill(mask, -1e9)
        x_mean = x_sum/sum_mask_curr

        active_seq_mask = torch.logical_not(active_seq_mask).squeeze(-2).squeeze(-1)
        if self.cfgs.DELAY > 0:
            active_seq_mask[:, torch.arange(self.cfgs.DELAY, dtype=torch.long)] = False

        # Select only mean of active elements
        active_mean = x_mean[active_seq_mask]
        return active_mean

    def forward(self, x, valid=False):
        length_mask = LengthMask(torch.sum(torch.abs(x) != 0, dim=-1),
                                 max_len=self.cfgs.MAX_TOKEN)
        attn_mask = TriangularCausalMask(self.cfgs.MAX_TOKEN)
        active_mask = (torch.sum(
                torch.abs(x.unsqueeze(2)), dim=-1
                ) == 0).unsqueeze(2)

        x = self.embedding(x)
        if self.cfgs.USE_GLOVE:
            x = self.proj(x)

        if self.position_enc:
            x = self.position(x)

        x = self.encoder(x,
                         attn_mask=attn_mask,
                         length_mask=length_mask
                         )

        if valid:
            # Apply mask to outputs and take average of hidden layer
            if self.cfgs.DELAY > 0:
                active_mask[:, torch.arange(self.cfgs.DELAY, dtype=torch.long)] = True
            masked_x = x.masked_fill(active_mask, 0)
            mask_sum = torch.logical_not(active_mask).sum(1)
            masked_x_sum = masked_x.sum(1)
            avg_x = masked_x_sum/mask_sum
        else:
            # Apply incremental mask to outputs and take incremental average
            # We treat incremental sequence classification as sequence labelling problem
            # for training where all the labels are the same.
            avg_x = self.incremental_mean(x, active_mask)
        # Shape: (active_tokens, label_size) for training, (batch_size, label_size) for valid/test
        # Need to move out_proj for recurrent setting or maybe not?
        # avg_x = self.out_proj(avg_x)

        active_mean_mask = torch.logical_not(active_mask).squeeze()
        return avg_x, active_mean_mask


class RecurrentEncoderClassification(nn.Module):
    """
    N stack of Linear Transformers as RNN for
    sequence classification.
    """
    def __init__(self, cfgs, token_size, label_size,
                 pretrained_emb=None, position_enc=True):
        super(RecurrentEncoderClassification, self).__init__()
        self.cfgs = cfgs
        assert cfgs.HIDDEN_SIZE % cfgs.ATTENTION_HEAD == 0

        # d_query = d_key = d_value
        self.HIDDEN_SIZE_HEAD = cfgs.HIDDEN_SIZE // cfgs.ATTENTION_HEAD

        self.params = {
                        'attention_type': 'linear',
                        'n_layers': cfgs.LAYER,
                        'n_heads': cfgs.ATTENTION_HEAD,
                        'feed_forward_dimensions': cfgs.FF_SIZE,
                        'query_dimensions': self.HIDDEN_SIZE_HEAD,
                        'value_dimensions': self.HIDDEN_SIZE_HEAD,
                        'dropout': cfgs.DROPOUT,
                        'attention_dropout': cfgs.DROPOUT,
                        'activation': 'relu',
                        'final_normalization': True
                    }

        self.encoder = RecurrentEncoderBuilder.from_dictionary(self.params
                                                               ).get()

        self.position_enc = position_enc

        if cfgs.USE_GLOVE:
            self.embedding = nn.Embedding(
                num_embeddings=token_size,
                embedding_dim=cfgs.WORD_EMBED_SIZE
            )
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))
            self.proj = nn.Linear(cfgs.WORD_EMBED_SIZE,
                                  cfgs.HIDDEN_SIZE)
        else:
            self.embedding = nn.Embedding(
                num_embeddings=token_size,
                embedding_dim=cfgs.HIDDEN_SIZE
            )

        if position_enc:
            self.position = RecurrentPositionalEncoding(cfgs)

        # self.out_proj = nn.Linear(cfgs.HIDDEN_SIZE,
        #                           label_size)

        # Xavier init
        for param in self.encoder.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

        # Needed by LinearCausalEncoderClassification for test time.
        increment_mask = torch.tril(torch.ones(self.cfgs.MAX_TOKEN,
                                               self.cfgs.MAX_TOKEN,
                                               dtype=torch.bool)).unsqueeze(0).unsqueeze(3)
        sum_mask = torch.arange(1, self.cfgs.MAX_TOKEN+1).unsqueeze(1)
        self.register_buffer('increment_mask', increment_mask)
        self.register_buffer('sum_mask', sum_mask)

    def forward(self, x, i=0, memory=None):
        x = self.embedding(x)
        if self.cfgs.USE_GLOVE:
            x = self.proj(x)
        if self.position_enc:
            x = self.position(x, i)
        x, memory = self.encoder(x.squeeze(dim=1), memory)
        return x, memory
