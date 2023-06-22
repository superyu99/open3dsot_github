import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from .ops.MSDeformableAttention.modules import MSDeformAttn

import copy

class ImageAwareTransformer(pl.LightningModule):
    def __init__(
            self,
            d_model=256,
            nhead=8,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_feedforward=512,
            dropout=0.1,
            activation="relu",
            num_feature_levels=1,
            enc_n_points=4):
        super().__init__()
        self.d_model = d_model #256
        self.nhead = nhead #8


        encoder_layer = VisualEncoderLayer(  #MSAttention
            d_model, dim_feedforward, dropout, activation, num_feature_levels,
            nhead, enc_n_points)
        self.encoder = VisualEncoder(encoder_layer, num_encoder_layers)

        #普通multihead attention
        decoder_layer = DecoderLayer(d_model, dim_feedforward, dropout, #普通多头注意力机制
                                     activation, nhead)
        self.decoder = Decoder(decoder_layer, num_decoder_layers)

      


    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(
            self,
            tgt,  #query
            img_feature,  #图像特征
            image_pos,  #图像pos
            tgt_pos_embed,  #与query同源的pos
            points_embed,  #points backbone的特征
            reference_points,
    ):
        # 图像先做cross attention
        bs, c, h, w = img_feature.shape
        spatial_shape = (h, w)
        img_feature = img_feature.flatten(2).transpose(1, 2)  #调整为：B*N*C
        image_pos = image_pos.flatten(2).transpose(1, 2)  #调整为：B*N*C

        # encoder 参数：
        # src, 
        # spatial_shapes, 
        # level_start_index, 
        # valid_ratios, 
        # pos=None, 
        # padding_mask=None):
        spatial_shape = torch.as_tensor(spatial_shape, dtype=torch.long, device=img_feature.device).unsqueeze(0)
        level_start_index = spatial_shape.new_zeros((1, ))
        valid_ratios = torch.ones(bs, 4, 2, dtype=torch.long, device=img_feature.device)


        memory = self.encoder(
            img_feature,
            spatial_shape,
            level_start_index,
            valid_ratios,
            image_pos,
        )

        points_embed = points_embed.unsqueeze(1)
        tgt = self.decoder(
            tgt,
            tgt_pos_embed,
            memory,
            points_embed)

        return tgt









class VisualEncoderLayer(pl.LightningModule):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src

class VisualEncoder(pl.LightningModule):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output

class DecoderLayer(pl.LightningModule):
    def __init__(self, d_model=256, d_ffn=512,
                 dropout=0.1, activation="relu",
                 n_heads=8):
        super().__init__()
        # point cross attention
        self.cross_attn_point = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        # img cross attention
        self.cross_attn_img = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
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
        image_embed,
        points_embed,
    ):
        #普通的 multihead attention
        tgt2 = self.cross_attn_img(
            self.with_pos_embed(tgt,query_pos).transpose(0, 1),  #Q:[2, 1, 256] NBC 这个输入应该是：个数、batch、维度C [1 B C]
            image_embed.transpose(0, 1),  #K [1, 2, 256] N,B,C
            image_embed.transpose(0, 1),  #V [1, 2, 256] N,B,C 
        )[0].transpose(0, 1)
        tgt = tgt + self.dropout(tgt2)  #跳连
        tgt = self.norm(tgt)

        #普通的 multihead attention
        tgt2 = self.cross_attn_point(
            self.with_pos_embed(tgt,query_pos).transpose(0, 1),  #Q:[2, 1, 256] NBC 这个输入应该是：个数、batch、维度C [1 B C]
            points_embed.transpose(0, 1),  #K [1, 2, 256] N,B,C
            points_embed.transpose(0, 1),  #V [1, 2, 256] N,B,C 
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


    def forward(self, tgt, query_pos, image_embed, points_embed):
        output = tgt

        for lid, layer in enumerate(self.layers):
            output = layer(
                output,
                query_pos,  #50*256可学习参数
                image_embed,
                points_embed,
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