import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model_utils import LayerNorm, PositionalEncoding, make_mask, subsequent_mask


class MultiHeadedAttention(nn.Module):
    """
    Multi-headed attention module.
    """
    def __init__(self, cfgs):
        super(MultiHeadedAttention, self).__init__()
        self.cfgs = cfgs
        assert cfgs.HIDDEN_SIZE % cfgs.ATTENTION_HEAD == 0

        # d_query = d_key = d_value
        self.HIDDEN_SIZE_HEAD = cfgs.HIDDEN_SIZE // cfgs.ATTENTION_HEAD

        self.linear_query = nn.Linear(cfgs.HIDDEN_SIZE,
                                      cfgs.HIDDEN_SIZE)
        self.linear_key = nn.Linear(cfgs.HIDDEN_SIZE,
                                    cfgs.HIDDEN_SIZE)
        self.linear_value = nn.Linear(cfgs.HIDDEN_SIZE, 
                                      cfgs.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(cfgs.HIDDEN_SIZE,
                                      cfgs.HIDDEN_SIZE)
        self.dropout = nn.Dropout(cfgs.DROPOUT)

    def attn(self, query, key, value, mask=None):
        """
        Compute scaled dot-product attention.
        """
        d_key = query.size(-1)

        # Shape: (batch_size, attention_head, query_len, key_len)
        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_key)

        # Mask to avoid attending on padding
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        # Shape: (batch_size, attention_head, query_len, key_len)
        attn_prob = F.softmax(scores, dim=-1)
        attn_prob = self.dropout(attn_prob)

        # Shape: (batch_size, attention_head, query_len, d_value)
        return torch.matmul(attn_prob, value), attn_prob

    def forward(self, query, key, value, mask=None):
        n_batches = query.size(0)

        # Project all inputs
        # Shape: (batch_size, attention_head, query_len, d_query)
        query = self.linear_query(query).view(
                n_batches,
                -1,
                self.cfgs.ATTENTION_HEAD,
                self.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        # Shape: (batch_size, attention_head, key_len, d_key)
        key = self.linear_key(key).view(
              n_batches,
              -1,
              self.cfgs.ATTENTION_HEAD,
              self.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        # Shape: (batch_size, attention_head, key_len/value_len, d_value)
        value = self.linear_value(value).view(
                n_batches,
                -1,
                self.cfgs.ATTENTION_HEAD,
                self.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        # Apply attention on all the projected vectors
        attn_vec, _ = self.attn(query, key, value, mask)

        # Concatenate and apply final linear layer
        # Shape: (batch_size, query_len, attention_head * head_dim/d_key)
        attn_vec = attn_vec.transpose(1, 2).contiguous().view(
                   n_batches,
                   -1,
                   self.cfgs.HIDDEN_SIZE
        )

        # Shape: (batch_size, query_len, attention_head * head_dim/d_key)
        attn_vec = self.linear_merge(attn_vec)
        return attn_vec


class FeedForward(nn.Module):
    """
    Position-wise FFNN.
    """
    def __init__(self, cfgs):
        super(FeedForward, self).__init__()
        self.cfgs = cfgs
        self.linear_1 = nn.Linear(cfgs.HIDDEN_SIZE,
                                  cfgs.FF_SIZE)
        self.linear_2 = nn.Linear(cfgs.FF_SIZE,
                                  cfgs.HIDDEN_SIZE)
        self.dropout = nn.Dropout(cfgs.DROPOUT)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class EncoderLayer(nn.Module):
    """
    A standard Transformer encoder architecture.
    """
    def __init__(self, cfgs):
        super(EncoderLayer, self).__init__()
        self.cfgs = cfgs
        self.multi_head_attn = MultiHeadedAttention(cfgs)
        self.ffnn = FeedForward(cfgs)

        self.dropout_1 = nn.Dropout(cfgs.DROPOUT)
        self.norm_1 = LayerNorm(cfgs.HIDDEN_SIZE)

        self.dropout_2 = nn.Dropout(cfgs.DROPOUT)
        self.norm_2 = LayerNorm(cfgs.HIDDEN_SIZE)

    def forward(self, x, mask):
        # Shape: (batch_size, query_len, hidden_size)
        x = self.norm_1(x + self.dropout_1(
            self.multi_head_attn(x, x, x, mask)
        ))

        # Shape: (batch_size, query_len, hidden_size)
        x = self.norm_2(x + self.dropout_2(
            self.ffnn(x)
        ))

        return x


class EncoderLabelling(nn.Module):
    """
    N stack of Encoder layers for sequence labelling.
    """
    def __init__(self, cfgs, token_size, label_size,
                 pretrained_emb=None, position_enc=True):
        super(EncoderLabelling, self).__init__()
        self.cfgs = cfgs
        self.enc_list = nn.ModuleList([EncoderLayer(cfgs)
                                       for _ in range(cfgs.LAYER)])

        self.norm = LayerNorm(cfgs.HIDDEN_SIZE)
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
            # To mix information about binary label into the word embedding itself
            self.pred_proj = nn.Linear(cfgs.HIDDEN_SIZE + cfgs.PRED_EMBED_SIZE,
                                       cfgs.HIDDEN_SIZE)

        if position_enc:
            self.position = PositionalEncoding(cfgs)

        # self.out_proj = nn.Linear(cfgs.HIDDEN_SIZE,
        #                           label_size)

        # Xavier init
        for param in self.enc_list.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, x, pred_mask=None):
        mask = make_mask(x.unsqueeze(2))
        x = self.embedding(x)
        if self.cfgs.USE_GLOVE:
            x = self.proj(x)

        if self.position_enc:
            x = self.position(x)

        if self.cfgs.DATASET == 'srl-nw-wsj':
            pred_embeds = self.pred_embedding(pred_mask)
            x = torch.cat((x, pred_embeds), dim=2)
            x = self.pred_proj(x)

        for enc in self.enc_list:
            x = enc(x, mask)

        x = self.norm(x)
        # x = self.out_proj(x)

        return x


class EncoderClassification(nn.Module):
    """
    N stack of Encoder layers for sequence classification.
    """
    def __init__(self, cfgs, token_size, label_size,
                 pretrained_emb=None, position_enc=True):
        super(EncoderClassification, self).__init__()
        self.cfgs = cfgs
        self.enc_list = nn.ModuleList([EncoderLayer(cfgs)
                                       for _ in range(cfgs.LAYER)])

        self.norm = LayerNorm(cfgs.HIDDEN_SIZE)
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
        for param in self.enc_list.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, x):
        mask = make_mask(x.unsqueeze(2))
        active_mask = (torch.sum(
                    torch.abs(x.unsqueeze(2)), dim=-1
                    ) == 0).unsqueeze(2)

        x = self.embedding(x)
        if self.cfgs.USE_GLOVE:
            x = self.proj(x)

        if self.position_enc:
            x = self.position(x)

        for enc in self.enc_list:
            x = enc(x, mask)

        x = self.norm(x)

        # Apply mask to outputs and take average of hidden layer
        masked_x = x.masked_fill(active_mask, 0)
        mask_sum = torch.logical_not(active_mask).sum(1)
        masked_x_sum = masked_x.sum(1)

        avg_x = masked_x_sum/mask_sum

        # x = self.out_proj(avg_x)
        return avg_x


class IncrementalEncoderLabelling(nn.Module):  # Unused
    """
    N stack of incremental Encoder layers for sequence labelling.
    """
    def __init__(self, cfgs, token_size, label_size,
                 pretrained_emb=None, position_enc=True):
        super(IncrementalEncoderLabelling, self).__init__()
        self.cfgs = cfgs
        self.enc_list = nn.ModuleList([EncoderLayer(cfgs)
                                       for _ in range(cfgs.LAYER)])

        self.norm = LayerNorm(cfgs.HIDDEN_SIZE)
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
            # To mix information about binary label into the word embedding itself
            self.pred_proj = nn.Linear(cfgs.HIDDEN_SIZE + cfgs.PRED_EMBED_SIZE,
                                       cfgs.HIDDEN_SIZE)

        if position_enc:
            self.position = PositionalEncoding(cfgs)

        # self.out_proj = nn.Linear(cfgs.HIDDEN_SIZE,
        #                           label_size)

        # Xavier init
        for param in self.enc_list.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, x, pred_mask=None, valid=False):
        mask = make_mask(x.unsqueeze(2))
        if not valid:
            mask = mask.expand(-1, -1, x.size(1), -1)
            # Mask future tokens and padding.
            mask = subsequent_mask(x.size(-1)).unsqueeze(0).to(mask.device) | mask

        x = self.embedding(x)
        if self.cfgs.USE_GLOVE:
            x = self.proj(x)

        if self.position_enc:
            x = self.position(x)

        if self.cfgs.DATASET == 'srl-nw-wsj':
            pred_embeds = self.pred_embedding(pred_mask)
            x = torch.cat((x, pred_embeds), dim=2)
            x = self.pred_proj(x)

        for enc in self.enc_list:
            x = enc(x, mask)

        x = self.norm(x)
        # x = self.out_proj(x)

        return x


class IncrementalEncoderClassification(nn.Module):  # Unused
    """
    N stack of incremental Encoder layers for sequence classification.
    """
    def __init__(self, cfgs, token_size, label_size,
                 pretrained_emb=None, position_enc=True):
        super(IncrementalEncoderClassification, self).__init__()
        self.cfgs = cfgs
        self.enc_list = nn.ModuleList([EncoderLayer(cfgs)
                                       for _ in range(cfgs.LAYER)])

        self.norm = LayerNorm(cfgs.HIDDEN_SIZE)
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
        for param in self.enc_list.parameters():
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
        # Select only mean of active elements
        active_mean = x_mean[active_seq_mask]
        return active_mean

    def forward(self, x, valid=False, sub_mask=True):
        mask = make_mask(x.unsqueeze(2))
        if sub_mask:
            mask = mask.expand(-1, -1, x.size(1), -1)
            # Mask future tokens and padding.
            mask = subsequent_mask(x.size(-1)).unsqueeze(0).to(mask.device) | mask

        active_mask = (torch.sum(
                    torch.abs(x.unsqueeze(2)), dim=-1
                    ) == 0).unsqueeze(2)

        x = self.embedding(x)
        if self.cfgs.USE_GLOVE:
            x = self.proj(x)

        if self.position_enc:
            x = self.position(x)

        for enc in self.enc_list:
            x = enc(x, mask)

        x = self.norm(x)

        if valid:
            # Apply mask to outputs and take average of hidden layer
            masked_x = x.masked_fill(active_mask, 0)
            mask_sum = torch.logical_not(active_mask).sum(1)
            masked_x_sum = masked_x.sum(1)

            avg_x = masked_x_sum/mask_sum
        else:
            avg_x = self.incremental_mean(x, active_mask)

        # x = self.out_proj(avg_x)
        active_mean_mask = torch.logical_not(active_mask).squeeze()
        return avg_x, active_mean_mask
