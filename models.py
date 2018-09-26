import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence


class Model(nn.Module):
    """
    References :
     1 - https://arxiv.org/abs/1704.03162
     2 - https://arxiv.org/pdf/1511.02274
     3 - https://arxiv.org/abs/1708.00584
    """

    def __init__(self, config, num_tokens):
        super(Model, self).__init__()

        dim_v = config['model']['pooling']['dim_v']
        dim_q = config['model']['pooling']['dim_q']
        dim_h = config['model']['pooling']['dim_h']

        n_glimpses = config['model']['attention']['glimpses']

        self.text = TextEncoder(
            num_tokens=num_tokens,
            emb_size=config['model']['seq2vec']['emb_size'],
            dim_q=dim_q,
            drop=config['model']['seq2vec']['dropout'],
        )
        self.attention = Attention(
            dim_v=dim_v,
            dim_q=dim_q,
            dim_h=config['model']['attention']['mid_features'],
            n_glimpses=n_glimpses,
            drop=config['model']['attention']['dropout'],
        )
        self.classifier = Classifier(
            dim_input=n_glimpses * dim_v + dim_q,
            dim_h=dim_h,
            top_ans=config['annotations']['top_ans'],
            drop=config['model']['classifier']['dropout'],
        )

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, v, q, q_len):

        q = self.text(q, list(q_len.data))
        # L2 normalization on the depth dimension
        v = F.normalize(v, p=2, dim=1)
        attention_maps = self.attention(v, q)
        v = apply_attention(v, attention_maps)
        # concatenate attended features and encoded question
        combined = torch.cat([v, q], dim=1)
        answer = self.classifier(combined)
        return answer


class Classifier(nn.Sequential):
    def __init__(self, dim_input, dim_h, top_ans, drop=0.0):
        super(Classifier, self).__init__()
        self.add_module('drop1', nn.Dropout(drop))
        self.add_module('lin1', nn.Linear(dim_input, dim_h))
        self.add_module('relu', nn.ReLU())
        self.add_module('drop2', nn.Dropout(drop))
        self.add_module('lin2', nn.Linear(dim_h, top_ans))


class TextEncoder(nn.Module):
    def __init__(self, num_tokens, emb_size, dim_q, drop=0.0):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(num_tokens, emb_size, padding_idx=0)
        self.dropout = nn.Dropout(drop)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(input_size=emb_size,
                            hidden_size=dim_q,
                            num_layers=1)
        self.dim_q = dim_q

        # Initialize parameters
        self._init_lstm(self.lstm.weight_ih_l0)
        self._init_lstm(self.lstm.weight_hh_l0)
        self.lstm.bias_ih_l0.data.zero_()
        self.lstm.bias_hh_l0.data.zero_()

        init.xavier_uniform_(self.embedding.weight)

    def _init_lstm(self, weight):
        for w in weight.chunk(4, 0):
            init.xavier_uniform_(w)

    def forward(self, q, q_len):
        embedded = self.embedding(q)
        tanhed = self.tanh(self.dropout(embedded))
        # pack to feed to the LSTM
        packed = pack_padded_sequence(tanhed, q_len, batch_first=True)
        _, (_, c) = self.lstm(packed)
        # _, (c, _) = self.lstm(packed) # this is h
        return c.squeeze(0)


class Attention(nn.Module):
    def __init__(self, dim_v, dim_q, dim_h, n_glimpses, drop=0.0):
        super(Attention, self).__init__()
        # As specified in https://arxiv.org/pdf/1511.02274.pdf the bias is already included in fc_q
        self.conv_v = nn.Conv2d(dim_v, dim_h, 1, bias=False)
        self.fc_q = nn.Linear(dim_q, dim_h)
        self.conv_x = nn.Conv2d(dim_h, n_glimpses, 1)

        self.dropout = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, v, q):
        # bring to the same shape
        v = self.conv_v(self.dropout(v))
        q = self.fc_q(self.dropout(q))
        q = repeat_encoded_question(q, v)
        # sum element-wise and ReLU
        x = self.relu(v + q)

        x = self.conv_x(self.dropout(x))  # We obtain n_glimpses attention maps [batch_size][n_glimpses][14][14]
        return x


def repeat_encoded_question(q, v):
    """
    Repeat the encoded question over all the spatial positions of the input image feature tensor.
    :param q: shape [batch_size][h]
    :param v: shape [batch_size][h][14][14]
    :return: a tensor constructed repeating q 14x14 with shape [batch_size][h][14][14]
    """
    batch_size, h = q.size()
    # repeat the encoded question [14x14] times (over all the spatial positions of the image feature matrix)
    q_tensor = q.view(batch_size, h, *([1, 1])).expand_as(v)
    return q_tensor


def apply_attention(v, attention):
    """
    Apply attention maps over the input image features.
    """
    batch_size, spatial_vec_size = v.size()[:2]
    glimpses = attention.size(1)

    # flatten the spatial dimensions [14x14] into a third dimension [196]
    v = v.view(batch_size, spatial_vec_size, -1)
    attention = attention.view(batch_size, glimpses, -1)
    n_image_regions = v.size(2)  # 14x14 = 196

    # Apply softmax to each attention map separately to create n_glimpses attention distribution over the image regions
    attention = attention.view(batch_size * glimpses, -1)  # [batch_size x n_glimpses][196]
    attention = F.softmax(attention, dim=1)

    # apply the weighting by creating a new dim to tile both tensors over
    target_size = [batch_size, glimpses, spatial_vec_size, n_image_regions]
    v = v.view(batch_size, 1, spatial_vec_size, n_image_regions).expand(
        *target_size)  # [batch_size][n_glimpses][2048][196]
    attention = attention.view(batch_size, glimpses, 1, n_image_regions).expand(
        *target_size)  # [batch_size][n_glimpses][2048][196]
    # Weighted sum over all the spatial regions vectors
    weighted = v * attention
    weighted_mean = weighted.sum(dim=3)  # [batch_size][n_glimpses][2048]

    # attended features are flattened in the same dimension
    return weighted_mean.view(batch_size, -1)  # [batch_size][n_glimpses * 2048]
