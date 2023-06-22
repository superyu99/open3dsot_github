import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from .ops.MSDeformableAttention.modules import MSDeformAttn

import copy


class SpeedAwareTransformer(pl.LightningModule):

    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        activation="relu",
    ):
        super().__init__()
        self.d_model = d_model #256
        self.nhead = nhead #8

        encoder_layer = EncoderLayer(  #MSAttention
            d_model, dim_feedforward, dropout, activation, nhead)

        self.encoder = Encoder(encoder_layer, num_encoder_layers)

        #普通multihead attention
        decoder_layer = DecoderLayer(
            d_model,
            dim_feedforward,
            dropout,  #普通多头注意力机制
            activation,
            nhead)
        self.decoder = Decoder(decoder_layer, num_decoder_layers)


    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(
            self,
            tgt,  #query
            query_pos,
            lidar_feature,  #图像特征
            radar_embed,  #points backbone的特征
    ):
        lidar_feature = lidar_feature.unsqueeze(1)
        tgt = self.encoder( #self attention
            tgt,
            query_pos,
            lidar_feature,
        )

        radar_embed = radar_embed.unsqueeze(1)
        tgt = self.decoder( #cross attention
            tgt,
            query_pos,
            radar_embed,
        )

        return tgt

class EncoderLayer(pl.LightningModule):
    def __init__(self, d_model=256, d_ffn=512,
                 dropout=0.1, activation="relu",
                 n_heads=8):
        super().__init__()
        # point cross attention
        self.self_attn_lidar = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        #ffn out
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout2(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm1(tgt)
        return tgt

    def forward(
        self,
        tgt,
        query_pos,  #这个应该是另一半直接加上来作为pos embed
        lidar_feature,
    ):
        #普通的 multihead attention
        tgt2 = self.self_attn_lidar(
            self.with_pos_embed(tgt,query_pos).transpose(0, 1),  #Q:[2, 1, 256] NBC 这个输入应该是：个数、batch、维度C [1 B C]
            lidar_feature.transpose(0, 1),  #K [1, 2, 256] N,B,C
            lidar_feature.transpose(0, 1),  #V [1, 2, 256] N,B,C 
        )[0].transpose(0, 1)
        tgt = tgt + self.dropout(tgt2)  #跳连
        tgt = self.norm(tgt)

        tgt = self.forward_ffn(tgt)  #[2*1*256]

        return tgt

class Encoder(pl.LightningModule):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers


    def forward(self, tgt, query_pos, lidar_feature):
        output = tgt

        for lid, layer in enumerate(self.layers):
            output = layer(
                output,
                query_pos,  #50*256可学习参数
                lidar_feature,
            )
        return output


class DecoderLayer(pl.LightningModule):
    def __init__(self, d_model=256, d_ffn=512,
                 dropout=0.1, activation="relu",
                 n_heads=8):
        super().__init__()
        # point cross attention
        self.cross_attn_radar = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        #ffn out
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)



    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout2(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm1(tgt)
        return tgt

    def forward(
        self,
        tgt,
        query_pos,  #这个应该是另一半直接加上来作为pos embed
        radar_embed,
    ):
        #普通的 multihead attention
        tgt2 = self.cross_attn_radar(
            self.with_pos_embed(tgt,query_pos).transpose(0, 1),  #Q:[2, 1, 256] NBC 这个输入应该是：个数、batch、维度C [1 B C]
            radar_embed.transpose(0, 1),  #K [1, 2, 256] N,B,C
            radar_embed.transpose(0, 1),  #V [1, 2, 256] N,B,C 
        )[0].transpose(0, 1)
        tgt = tgt + self.dropout(tgt2)  #跳连
        tgt = self.norm(tgt)

        tgt = self.forward_ffn(tgt)  #[2*50*256]

        return tgt

class Decoder(pl.LightningModule):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers


    def forward(self, tgt, query_pos, radar_embed):
        output = tgt

        for lid, layer in enumerate(self.layers):
            output = layer(
                output,
                query_pos,  #50*256可学习参数
                radar_embed,
            )
        return output




def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")