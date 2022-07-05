# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib.utils import scaled_Laplacian, cheb_polynomial


class SScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, num_of_d):
        super(SScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.num_of_d = num_of_d

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k) # scores : [batch_size, n_heads, len_q, len_k]
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.
        attn = nn.Softmax(dim=3)(scores)
        context = torch.matmul(attn, V)
        # attn = F.softmax(scores, dim=2)
        return context


class TScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, num_of_d):
        super(TScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.num_of_d =num_of_d

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.
        attn = nn.Softmax(dim=3)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context



class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.
        return scores


class SMultiHeadAttention(nn.Module):
    def __init__(self, DEVICE, d_model, d_k ,d_v, n_heads, num_of_d):
        super(SMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.num_of_d = num_of_d
        self.DEVICE = DEVICE
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, self.num_of_d, -1, self.n_heads, self.d_k).transpose(2,
                                                                                                    3)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, self.num_of_d, -1, self.n_heads, self.d_k).transpose(2,
                                                                                                    3)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, self.num_of_d, -1, self.n_heads, self.d_v).transpose(2,
                                                                                                    3)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1,
                                                  1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context = SScaledDotProductAttention(self.d_k, self.num_of_d)(Q, K, V, attn_mask)
        context = context.transpose(2, 3).reshape(batch_size, self.num_of_d, -1,
                                                  self.n_heads * self.d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]

        return nn.LayerNorm(self.d_model).to(self.DEVICE)(output + residual)

class TMultiHeadAttention(nn.Module):
    def __init__(self, DEVICE, d_model, d_k ,d_v, n_heads, num_of_d):
        super(TMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.num_of_d = num_of_d
        self.DEVICE = DEVICE
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, self.num_of_d, -1, self.n_heads, self.d_k).transpose(2, 3)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, self.num_of_d, -1, self.n_heads, self.d_k).transpose(2, 3)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, self.num_of_d, -1, self.n_heads, self.d_v).transpose(2, 3)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1,
                                                  1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context = TScaledDotProductAttention(self.d_k, self.num_of_d)(Q, K, V, attn_mask)

        context = context.transpose(2, 3).reshape(batch_size, self.num_of_d, -1,
                                                  self.n_heads * self.d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]

        return nn.LayerNorm(self.d_model).to(self.DEVICE)(output + residual)


class MultiHeadAttention(nn.Module):
    def __init__(self, DEVICE, d_model, d_k, d_v, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.DEVICE = DEVICE
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)

    def forward(self, input_Q, input_K, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,
                                                                                     2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,
                                                                                     2)  # K: [batch_size, n_heads, len_k, d_k]
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1,
                                                      1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        attn = ScaledDotProductAttention(self.d_k)(Q, K, attn_mask)
        return attn


class T2MultiHeadAttention(nn.Module):
    def __init__(self, DEVICE, d_model, d_k ,d_v, n_heads, num_of_d):
        super(T2MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.num_of_d = num_of_d
        self.DEVICE = DEVICE
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, self.num_of_d, -1, self.n_heads, self.d_k).transpose(2, 3)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, self.num_of_d, -1, self.n_heads, self.d_k).transpose(2, 3)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, self.num_of_d, -1, self.n_heads, self.d_v).transpose(2, 3)  # V: [batch_size, n_heads, len_v(=len_k), d_v]
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1,
                                                  1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context = TScaledDotProductAttention(self.d_k, self.num_of_d)(Q, K, V, attn_mask)

        # output = self.fc(context)  # [batch_size, len_q, d_model]

        mr = residual.unsqueeze(2).repeat(1, 1, self.n_heads, 1, 1)
        return nn.LayerNorm(self.d_model).to(self.DEVICE)(context + mr)



class cheb_conv_withSAt(nn.Module):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self, K, cheb_polynomials, in_channels, out_channels, num_of_vertices):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv_withSAt, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = cheb_polynomials[0].device
        self.relu = nn.ReLU(inplace=True)
        self.Theta = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])
        self.mask = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(num_of_vertices,num_of_vertices).to(self.DEVICE)) for _ in range(K)])
    def forward(self, x, spatial_attention, adj_pa):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape

        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)

            for k in range(self.K):

                T_k = self.cheb_polynomials[k]  # (N,N)
                mask = self.mask[k]

                myspatial_attention = spatial_attention[:, k, :, :] + adj_pa.mul(mask)
                myspatial_attention = F.softmax(myspatial_attention, dim=1)

                T_k_with_at = T_k.mul(myspatial_attention)

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = T_k_with_at.permute(0, 2, 1).matmul(graph_signal)

                output = output + rhs.matmul(theta_k)  # (b, N, F_in)(F_in, F_out) = (b, N, F_out)

            outputs.append(output.unsqueeze(-1))  # (b, N, F_out, 1)

        return self.relu(torch.cat(outputs, dim=-1))  # (b, N, F_out, T)


class Embedding(nn.Module):
    def __init__(self, nb_seq, d_Em, num_of_features, Etype):
        super(Embedding, self).__init__()
        self.nb_seq = nb_seq
        self.Etype = Etype
        self.num_of_features = num_of_features
        self.pos_embed = nn.Embedding(nb_seq, d_Em)
        self.norm = nn.LayerNorm(d_Em)

    def forward(self, x, batch_size):
        if self.Etype == 'T':
            pos = torch.arange(self.nb_seq, dtype=torch.long).cuda()
            pos = pos.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_of_features,
                                                   self.nb_seq)  # [seq_len] -> [batch_size, seq_len]
            embedding = x + self.pos_embed(pos)
        elif self.Etype == 'S':
            pos = torch.arange(self.nb_seq, dtype=torch.long).cuda()
            pos = pos.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_of_features,
                                                   self.nb_seq)
            embedding = x + self.pos_embed(pos)
        else:
            pos = torch.arange(self.nb_seq, dtype=torch.long).cuda()
            pos = pos.unsqueeze(0).expand(batch_size, self.nb_seq)
            embedding = x + self.pos_embed(pos)
        Emx = self.norm(embedding)
        return Emx


class GDC(nn.Module):
    def __init__(self, in_channels, time_strides, kernel_size, dilation, padding):
        super(GDC, self).__init__()
        self.in_channels = in_channels
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.con2out = nn.Conv2d(in_channels, 2 * in_channels,
                                 kernel_size=(1, kernel_size),
                                 stride=(1, time_strides),
                                 dilation=(1, dilation),
                                 padding=(0, padding))
        self.bn = nn.BatchNorm2d(2 * in_channels)

    def forward(self, x):
        x_causal_conv = self.con2out(x)
        x_causal_conv = self.bn(x_causal_conv)
        x_p = x_causal_conv[:, : self.in_channels, :, :]
        x_q = x_causal_conv[:, -self.in_channels:, :, :]
        x_gtu = torch.mul(self.tanh(x_p), self.sigmoid(x_q))
        return x_gtu



class MHASTIGCN_block(nn.Module):

    def __init__(self, DEVICE, num_of_d, in_channels, K, nb_chev_filter, nb_time_filter, time_strides,
                 cheb_polynomials, adj_pa, n_gdc, num_of_vertices, num_of_timesteps, d_model, d_k, d_v, n_heads):
        super(MHASTIGCN_block, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)
        self.M = n_gdc

        self.adj_pa = torch.FloatTensor(adj_pa).cuda()

        self.pre_conv = nn.Conv2d(num_of_timesteps, d_model, kernel_size=(1, num_of_d))
        self.pre_conv1 = nn.Conv2d(num_of_timesteps, d_model, kernel_size=(1, num_of_d))

        self.EmbedT = Embedding(num_of_timesteps, num_of_vertices, num_of_d, 'T')
        self.EmbedS = Embedding(num_of_vertices, num_of_timesteps, num_of_d, 'S')
        self.EmbedST = Embedding(num_of_vertices, 2 * d_model, num_of_d, 'ST')
        self.EmbedT2 = Embedding(num_of_timesteps, nb_chev_filter, num_of_vertices, 'T')

        self.TAt = TMultiHeadAttention(DEVICE, num_of_vertices, d_k, d_v, n_heads, num_of_d)
        self.SAt = SMultiHeadAttention(DEVICE, num_of_timesteps, d_k, d_v, n_heads, num_of_d)
        self.qt_att = MultiHeadAttention(DEVICE, 2 * d_model, d_k, d_v, K)

        self.cheb_conv_SAt = cheb_conv_withSAt(K, cheb_polynomials, in_channels, nb_chev_filter, num_of_vertices)

        self.T2At = T2MultiHeadAttention(DEVICE, nb_chev_filter, d_k, d_v, n_gdc, num_of_vertices)

        self.convs = nn.ModuleList([])
        for i in range(n_gdc):
            self.convs.append(nn.Sequential(
                GDC(nb_time_filter, time_strides, 3, i+1, i+1)
            ))
        self.pool = nn.AdaptiveAvgPool2d(1*1)

        self.fc1 = nn.Linear(2 * nb_time_filter, nb_time_filter)
        self.fcs = nn.ModuleList([])
        for i in range(n_gdc):
            self.fcs.append(
                nn.Linear(nb_time_filter, nb_time_filter)
            )
        self.softmax = nn.Softmax(dim=1)

        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))

        self.dropout = nn.Dropout(p=0.05)

        self.ln = nn.LayerNorm(nb_time_filter)  # 需要将channel放到最后一个维度上

    def forward(self, x):
        '''
        :param x: (Batch_size, N, F_in, T)
        :param res_att: (Batch_size, N, F_in, T)
        :return: (Batch_size, N, nb_time_filter, T)
        '''
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape  # B,N,F,T

        # TAT
        if num_of_features == 1:
            t_emb = self.EmbedT(x.permute(0,2,3,1), batch_size)  # B,F,T,N
            s_emb = self.EmbedS(x.permute(0,2,1,3), batch_size)  # B,F,N,T
        else:
            t_emb = x.permute(0, 2, 3, 1)
            s_emb = x.permute(0, 2, 1, 3)

        t_att_out = self.TAt(t_emb, t_emb, t_emb, None)  # B,F,T,N; B,F,Ht,T,T
        x_t_att = self.pre_conv(t_att_out.permute(0, 2, 3, 1))[:, :, :, -1].permute(0, 2, 1)  # B,N,d_model

        s_att_out = self.SAt(s_emb, s_emb, s_emb, None)  # B,F,N,T; B,F,Ht,N,N
        x_s_att = self.pre_conv1(s_att_out.permute(0, 3, 2, 1))[:, :, :, -1].permute(0, 2, 1)  # B,N,d_model

        x_st_att = torch.cat([x_t_att, x_s_att], dim=-1)
        x_st_att = self.EmbedST(x_st_att,batch_size)
        x_st_att = self.dropout(x_st_att)
        qt_att = self.qt_att(x_st_att, x_st_att, None)  # B,Hs,N,N

        # graph convolution in spatial dim
        spatial_gcn = self.cheb_conv_SAt(x, qt_att, self.adj_pa)  # B,N,F,T

        # convolution along the time axis
        gcn_out_1 = spatial_gcn.permute(0, 2, 1, 3)
        gcn_out = self.EmbedT2(gcn_out_1.permute(0, 2, 3, 1), batch_size)
        gcn_out = self.dropout(gcn_out)
        gdc_input = self.T2At(gcn_out, gcn_out, gcn_out, None)

        for i, conv in enumerate(self.convs):
            fea = conv(gdc_input[:, :, i, :, :].squeeze(2).permute(0, 3, 1, 2)).unsqueeze_(dim=1)
            if i == 0:
                gdc_output = fea
            else:
                gdc_output = torch.cat([gdc_output, fea], dim=1)
        fea_U = torch.sum(gdc_output, dim=1)

        s = []
        s.append(fea_U.mean(-1).mean(-1))
        s.append(fea_U.std(-1).std(-1))
        s1 = torch.cat(s, dim=1)

        z = self.fc1(s1)
        for i, fc in enumerate(self.fcs):
            vector = fc(z).unsqueeze_(dim=1)
            if i == 0:
                att_weight = vector
            else:
                att_weight = torch.cat([att_weight,vector], dim=1)
        att_weight = self.softmax(att_weight)
        att_weight = att_weight.unsqueeze(-1).unsqueeze(-1)
        time_conv = (gdc_output * att_weight).sum(dim=1)

        if num_of_features == 1:
            time_conv_output = self.relu(time_conv)
        else:
            time_conv_output = self.relu(gcn_out_1 + time_conv)  # B,F,N,T

        # residual shortcut
        if num_of_features == 1:
            x_residual = self.residual_conv(x.permute(0, 2, 1, 3))
        else:
            x_residual = x.permute(0, 2, 1, 3)
        x_residual = self.ln(F.relu(x_residual + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)

        return x_residual


class MHASTIGCN_submodule(nn.Module):

    def __init__(self, DEVICE, num_of_d, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides,
                 cheb_polynomials, adj_pa, n_gdc, num_for_predict, len_input, num_of_vertices, d_model, d_k, d_v, n_heads):
        '''
        :param nb_block:
        :param in_channels:
        :param K:
        :param nb_chev_filter:
        :param nb_time_filter:
        :param time_strides:
        :param cheb_polynomials:
        :param num_for_predict:
        '''

        super(MHASTIGCN_submodule, self).__init__()

        self.BlockList = nn.ModuleList([MHASTIGCN_block(DEVICE, num_of_d, in_channels, K,
                                                     nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials,
                                                     adj_pa, n_gdc, num_of_vertices, len_input, d_model, d_k, d_v, n_heads)])

        self.BlockList.extend([MHASTIGCN_block(DEVICE, num_of_d * nb_time_filter, nb_chev_filter, K,
                                            nb_chev_filter, nb_time_filter, 1, cheb_polynomials,
                                            adj_pa, n_gdc, num_of_vertices, len_input//time_strides, d_model, d_k, d_v, n_heads) for _ in range(nb_block-1)])

        self.final_conv = nn.Conv2d(int((len_input/time_strides) * nb_block), 128, kernel_size=(1, nb_time_filter))
        self.final_fc = nn.Linear(128, num_for_predict)
        self.DEVICE = DEVICE

        self.to(DEVICE)

    def forward(self, x):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        """
        for block in self.BlockList:
            x = block(x)
            
        output = self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)
        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)
        """
        need_concat = []

        for block in self.BlockList:
            x = block(x)
            need_concat.append(x)

        final_x = torch.cat(need_concat, dim=-1)
        output1 = self.final_conv(final_x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)
        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,128)
        output = self.final_fc(output1)

        return output


def make_model(DEVICE, num_of_d, nb_block, in_channels, K,
               nb_chev_filter, nb_time_filter, time_strides, adj_mx, adj_pa,
               n_gdc, num_for_predict, len_input, num_of_vertices, d_model, d_k, d_v, n_heads):
    '''

    :param DEVICE:
    :param nb_block:
    :param in_channels:
    :param K:
    :param nb_chev_filter:
    :param nb_time_filter:
    :param time_strides:
    :param num_for_predict:
    :param len_input
    :return:
    '''
    L_tilde = scaled_Laplacian(adj_mx)
    cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(DEVICE) for i in cheb_polynomial(L_tilde, K)]
    model = MHASTIGCN_submodule(DEVICE, num_of_d, nb_block, in_channels,
                             K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials,
                             adj_pa, n_gdc, num_for_predict, len_input, num_of_vertices, d_model, d_k, d_v, n_heads)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    return model
