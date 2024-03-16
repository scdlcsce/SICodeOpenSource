"""Defines all graph embedding models"""
from functools import reduce
import random

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

import utils
# import feature_preprocess


class OrderEmbedder(nn.Module):
    def __init__(self, args):
        super(OrderEmbedder, self).__init__()
        self.emb_model = SkipLastGNN(args.feature_num, args.hidden_dim, args.hidden_dim, args)
        self.margin = args.margin
        self.use_intersection = False
        self.clf_model = nn.Sequential(nn.Linear(1, 2), nn.LogSoftmax(dim=-1))

    def forward(self, emb_as, emb_bs):
        return emb_as, emb_bs

    def predict(self, emb_as, emb_bs): #b 是否是 a 的子图
        return torch.sum(torch.max(torch.zeros_like(emb_as, device=emb_as.device), emb_bs - emb_as)**2, dim=1)


class SkipLastGNN(nn.Module):
    def __init__(self, feature_num, hidden_dim, output_dim, args):
        super(SkipLastGNN, self).__init__()
        self.dropout = args.dropout
        self.n_layers = args.n_layers

        self.feat_preprocess = None

        self.pre_mp = nn.Embedding(num_embeddings=feature_num, embedding_dim=hidden_dim)
        # self.pre_mp = nn.Sequential(nn.Linear(1, hidden_dim))

        conv_model = self.build_conv_model(args.conv_type, 1)
        self.convs = nn.ModuleList()

        self.learnable_skip = nn.Parameter(torch.ones(self.n_layers, self.n_layers))

        for l in range(args.n_layers):
            hidden_input_dim = hidden_dim * (l + 1)
            self.convs.append(conv_model(hidden_input_dim, hidden_dim))

        post_input_dim = hidden_dim * (args.n_layers + 1)
        self.post_mp = nn.Sequential(
            nn.Linear(post_input_dim, hidden_dim), nn.Dropout(args.dropout),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 256), nn.ReLU(),
            nn.Linear(256, hidden_dim))
        self.skip = args.skip
        self.conv_type = args.conv_type

    def build_conv_model(self, model_type, n_inner_layers):
        return lambda i, h: GINConv(nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h)))

    def forward(self, data):
        # if self.feat_preprocess is not None:
        #     if not hasattr(data, "preprocessed"):
        #         data = self.feat_preprocess(data)
        #         data.preprocessed = True
        x, edge_index, batch = data.node_feature, data.edge_index, data.batch
        x = self.pre_mp(x).squeeze(1)

        all_emb = x.unsqueeze(1)
        emb = x
        for i in range(len(self.convs)):
            skip_vals = self.learnable_skip[i,:i+1].unsqueeze(0).unsqueeze(-1)
            curr_emb = all_emb * torch.sigmoid(skip_vals)
            curr_emb = curr_emb.view(x.size(0), -1)
            x = self.convs[i](curr_emb, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            emb = torch.cat((emb, x), 1)
            all_emb = torch.cat((all_emb, x.unsqueeze(1)), 1)

        emb = pyg_nn.global_add_pool(emb, batch)
        emb = self.post_mp(emb)
        return emb

    def loss(self, pred, label):
        return F.nll_loss(pred, label)


class GINConv(pyg_nn.MessagePassing):
    def __init__(self, nn, eps=0, train_eps=False, **kwargs):
        super(GINConv, self).__init__(aggr='add', **kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        #reset(self.nn)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_index, edge_weight = pyg_utils.remove_self_loops(edge_index,
            edge_weight)
        out = self.nn((1 + self.eps) * x + self.propagate(edge_index, x=x,
            edge_weight=edge_weight))
        return out

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)

