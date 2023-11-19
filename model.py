import torch
import torch.nn as nn
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter
import torch.nn.functional as F
import numpy as np


def nconv(x, A):
    """Multiply x by adjacency matrix along source node axis"""
    return torch.einsum ('ncvl,vw->ncwl', (x, A)).contiguous ()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)


class FC(nn.Module):
    def __init__(self,c_in,c_out):
        super(FC,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)


class ScaledDotProductAttention (nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super ().__init__ ()
        self.temperature = temperature
        self.dropout = nn.Dropout (attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul (q / self.temperature, k.transpose (2, 3))

        if mask is not None:
            attn = attn.masked_fill (mask == 0, -1e9)

        attn = self.dropout (F.softmax (attn, dim=-1))
        output = torch.matmul (attn, v)

        return output, attn


class MultiHeadAttention (nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super ().__init__ ()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear (d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear (d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear (d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear (n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention (temperature=d_k ** 0.5)

        self.dropout = nn.Dropout (dropout)
        self.layer_norm = nn.LayerNorm (d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size (0), q.size (1), k.size (1), v.size (1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs (q).view (sz_b, len_q, n_head, d_k)
        k = self.w_ks (k).view (sz_b, len_k, n_head, d_k)
        v = self.w_vs (v).view (sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose (1, 2), k.transpose (1, 2), v.transpose (1, 2)

        if mask is not None:
            mask = mask.unsqueeze (1)  # For head axis broadcasting.

        q, attn = self.attention (q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose (1, 2).contiguous ().view (sz_b, len_q, -1)
        q = self.dropout (self.fc (q))
        q += residual

        q = self.layer_norm (q)

        return q, attn

class graphattention(nn.Module):
    def __init__(self,c_in,c_out,dropout,d=16, emb_length=0, aptonly=False, noapt=False):
        super(graphattention,self).__init__()
        self.d = d
        self.aptonly = aptonly
        self.noapt = noapt
        self.mlp = linear(c_in*2,c_out)
        self.dropout = dropout
        self.emb_length = emb_length
        if aptonly:
            self.qm = FC(self.emb_length, d)  # query matrix
            self.km = FC(self.emb_length, d)  # key matrix
        elif noapt:
            self.qm = FC(c_in, d)  # query matrix
            self.km = FC(c_in, d)  # key matrix
        else:
            self.qm = FC(c_in + self.emb_length, d)  # query matrix
            self.km = FC(c_in + self.emb_length, d)  # key matrix

    def forward(self,x,embedding):
        # x: [batch_size, D, nodes, time_step]
        # embedding = [10, num_nodes]
        out = [x]

        embedding = embedding.repeat((x.shape[0], x.shape[-1], 1, 1)) # embedding = [batch_size, time_step, 10, num_nodes]
        embedding = embedding.permute(0,2,3,1) # embedding = [batch_size, 16, num_nodes, time_step]

        if self.aptonly:
            x_embedding = embedding
            query = self.qm(x_embedding).permute(0, 3, 2, 1)
            key = self.km(x_embedding).permute(0, 3, 2, 1)  #
            # value = self.vm(x)
            attention = torch.matmul(query,key.permute(0, 1, 3, 2))  # attention=[batch_size, time_step, num_nodes, num_nodes]
            # attention = F.relu(attention)
            attention /= (self.d ** 0.5)
            attention = F.softmax(attention, dim=-1)
        elif self.noapt:
            x_embedding = x
            query = self.qm(x_embedding).permute(0, 3, 2, 1)  # query=[batch_size, time_step, num_nodes, d]
            key = self.km(x_embedding).permute(0, 3, 2, 1)  # key=[batch_size, time_step, num_nodes, d]
            attention = torch.matmul(query,key.permute(0, 1, 3, 2))  # attention=[batch_size, time_step, num_nodes, num_nodes]
            # attention = F.relu(attention)
            attention /= (self.d ** 0.5)
            attention = F.softmax(attention, dim=-1)
        else:
            x_embedding = torch.cat([x,embedding], axis=1) # x_embedding=[batch_size, D+10, num_nodes, time_step]
            query = self.qm(x_embedding).permute(0,3,2,1) # query=[batch_size, time_step, num_nodes, d]
            key = self.km(x_embedding).permute(0,3,2,1) # key=[batch_size, time_step, num_nodes, d]
            # query = F.relu(query)
            # key = F.relu(key)
            attention = torch.matmul(query, key.permute(0,1,3,2)) # attention=[batch_size, time_step, num_nodes, num_nodes]
            # attention = F.relu(attention)
            attention /= (self.d**0.5)
            attention = F.softmax(attention, dim=-1)

        x = torch.matmul(x.permute(0,3,1,2), attention).permute(0,2,3,1)
        out.append(x)

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h, 0#attention

class GraphConvNet (nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super ().__init__ ()
        c_in = (order * support_len + 1) * c_in
        self.final_conv = Conv2d (c_in, c_out, (1, 1), padding=(0, 0), stride=(1, 1), bias=True)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support: list):
        out = [x]
        for a in support:
            x1 = nconv (x, a)
            out.append (x1)
            for k in range (2, self.order + 1):
                x2 = nconv (x1, a)
                out.append (x2)
                x1 = x2

        h = torch.cat (out, dim=1)
        h = self.final_conv (h)
        h = F.dropout (h, self.dropout, training=self.training)
        return h


class STABC (nn.Module):
    def __init__(self, device, num_nodes, out_dim, target_range, dropout=0.3, supports=None, gat_bool=False,do_graph_conv=True,
                 addaptadj=True, aptinit=None, in_dim=1,
                 residual_channels=32, dilation_channels=32, cat_feat_gc=False,
                 skip_channels=64, end_channels=128, kernel_size=2, blocks=4, layers=2,
                 apt_size=10,n_head = 2, d_k = 20, d_v = 20):
        super ().__init__ ()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.do_graph_conv = do_graph_conv
        self.cat_feat_gc = cat_feat_gc
        self.addaptadj = addaptadj
        self.target_range = target_range
        self.d_model = out_dim
        self.gat_bool = gat_bool
        self.bn = nn.ModuleList()
        self.gat = nn.ModuleList()
        self.num_node = num_nodes
        self.out_dim = out_dim

        if self.cat_feat_gc:
            self.start_conv = nn.Conv2d (in_channels=1,  # hard code to avoid errors
                                         out_channels=residual_channels,
                                         kernel_size=(1, 1))
            self.cat_feature_conv = nn.Conv2d (in_channels=in_dim - 1,
                                               out_channels=residual_channels,
                                               kernel_size=(1, 1))
        else:
            self.start_conv = nn.Conv2d (in_channels=in_dim,
                                         out_channels=residual_channels,
                                         kernel_size=(1, 1))

        self.fixed_supports = supports or []
        receptive_field = 1

        self.supports_len = len (self.fixed_supports)
        if do_graph_conv and addaptadj:
            if aptinit is None:
                nodevecs = torch.randn (num_nodes, apt_size), torch.randn (apt_size, num_nodes)
            else:
                nodevecs = self.svd_init (apt_size, aptinit)
            self.supports_len += 1
            self.nodevec1, self.nodevec2 = [Parameter (n.to (device), requires_grad=True) for n in nodevecs]

        depth = list (range (blocks * layers))

        # 1x1 convolution for residual and skip connections (slightly different see docstring)
        self.residual_convs = ModuleList ([Conv1d (dilation_channels, residual_channels, (1, 1)) for _ in depth])
        self.skip_convs = ModuleList ([Conv1d (dilation_channels, skip_channels, (1, 1)) for _ in depth])
        self.bn = ModuleList ([BatchNorm2d (residual_channels) for _ in depth])
        self.graph_convs = ModuleList (
            [GraphConvNet (dilation_channels, residual_channels, dropout, support_len=self.supports_len)
             for _ in depth])

        self.filter_convs = ModuleList ()
        self.gate_convs = ModuleList ()
        for b in range (blocks):
            additional_scope = kernel_size - 1
            D = 1  # dilation
            for i in range (layers):
                # dilated convolutions
                self.filter_convs.append (Conv2d (residual_channels, dilation_channels, (1, kernel_size), dilation=D))
                self.gate_convs.append (Conv1d (residual_channels, dilation_channels, (1, kernel_size), dilation=D))
                D *= 2
                receptive_field += additional_scope
                additional_scope *= 2
        self.receptive_field = receptive_field

        self.end_conv_1 = Conv2d (skip_channels, end_channels, (1, 1), bias=True)
        self.end_conv_2 = Conv2d (end_channels, out_dim, (1, 1), bias=True)
        self.mutihead_attention = MultiHeadAttention(n_head, self.d_model, d_k, d_v, dropout=dropout)

    @staticmethod
    def svd_init(apt_size, aptinit):
        m, p, n = torch.svd (aptinit)
        nodevec1 = torch.mm (m[:, :apt_size], torch.diag (p[:apt_size] ** 0.5))
        nodevec2 = torch.mm (torch.diag (p[:apt_size] ** 0.5), n[:, :apt_size].t ())
        return nodevec1, nodevec2

    @classmethod
    def from_args(cls, args, device, supports, aptinit, **kwargs):
        defaults = dict (dropout=args.dropout, supports=supports,
                         do_graph_conv=args.do_graph_conv, addaptadj=args.addaptadj, aptinit=aptinit,
                         in_dim=args.in_dim, apt_size=args.apt_size, out_dim=args.horizon,
                         residual_channels=args.nhid, dilation_channels=args.nhid,
                         skip_channels=args.nhid * 8, end_channels=args.nhid * 16,
                         cat_feat_gc=args.cat_feat_gc, target_range=args.target_range)
        defaults.update (**kwargs)
        model = cls (device, args.num_nodes, **defaults)
        return model

    def load_checkpoint(self, state_dict):
        """It is assumed that ckpt was trained to predict a subset of timesteps."""
        bk, wk = ['end_conv_2.bias', 'end_conv_2.weight']  # only weights that depend on seq_length
        b, w = state_dict.pop (bk), state_dict.pop (wk)
        self.load_state_dict (state_dict, strict=False)
        cur_state_dict = self.state_dict ()
        cur_state_dict[bk][:b.shape[0]] = b
        cur_state_dict[wk][:w.shape[0]] = w
        self.load_state_dict (cur_state_dict)

    def find_target_lanes(self,x):
        if self.target_range != 0:
            for i in range (self.target_range[1], self.target_range[3] + 1):
                target_lane = x[:, :, int (i * self.target_range[0] + self.target_range[2]):int (
                    i * self.target_range[0] + self.target_range[4]) + 1, :]
                target_lanes = target_lane
                if i > self.target_range[1]:
                    target_lanes = torch.cat ([target_lanes,target_lane], dim=2)
        else:
            target_lanes = x
        return target_lanes

    def forward(self, x,attn_mask=None):
        # Input shape is (bs, features, n_nodes, n_timesteps)

        in_len = x.size (3)
        if in_len < self.receptive_field:
            x = nn.functional.pad (x, (self.receptive_field - in_len, 0, 0, 0))
        if self.cat_feat_gc:
            f1, f2 = x[:, [0]], x[:, 1:]
            x1 = self.start_conv (f1)
            x2 = F.leaky_relu (self.cat_feature_conv (f2))
            x = x1 + x2
        else:
            x = self.start_conv (x)
        skip = 0
        adjacency_matrices = self.fixed_supports
        # calculate the current adaptive adj matrix once per iteration
        if self.addaptadj:
            adp = F.softmax (F.relu (torch.mm (self.nodevec1, self.nodevec2)), dim=1)
            adjacency_matrices = self.fixed_supports + [adp]
        if self.gat_bool:
            w = torch.randn(self.bn,self.num_node,self.out_dim)
            x = x+ w*graphattention(x)

        # WaveNet layers
        for i in range (self.blocks * self.layers):
            residual = x
            # dilated convolution
            filter = torch.tanh (self.filter_convs[i] (residual))
            gate = torch.sigmoid (self.gate_convs[i] (residual))
            x = filter * gate
            # parametrized skip connection
            s = self.skip_convs[i] (x)
            try:  # if i > 0 this works
                skip = skip[:, :, :, -s.size (3):]  # TODO(SS): Mean/Max Pool?
            except:
                skip = 0
            skip = s + skip
            if i == (self.blocks * self.layers - 1):  # last X getting ignored anyway
                break
            if self.gat_bool:
                if self.addaptadj:
                    x, att = self.gat[i](x, self.embedding)

            if self.do_graph_conv:
                graph_out = self.graph_convs[i] (x, adjacency_matrices)
                x = x + graph_out if self.cat_feat_gc else graph_out
            else:
                x = self.residual_convs[i] (x)
            x = x + residual[:, :, :, -x.size (3):]  # TODO(SS): Mean/Max Pool?
            x = self.bn[i] (x)

        x = F.relu (skip)
        x = F.relu (self.end_conv_1 (x))
        x = self.end_conv_2 (x) # downsample to (bs, seq_length, num_nodes, nfeatures)
        target = self.find_target_lanes (x)
        x,target = x.squeeze().transpose(1, 2) ,target.squeeze().transpose(1, 2)
        # target,att = self.mutihead_attention(target,x,x,mask=attn_mask)

        return target.unsqueeze(3).transpose(1, 2)
