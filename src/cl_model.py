import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv, GCNConv, global_max_pool, global_mean_pool
import sys
# import numpy as np

sys.path.append('..')
from utils import reset

# drug_num = 87
# cline_num = 55


class HgnnEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HgnnEncoder, self).__init__()
        self.conv1 = HypergraphConv(in_channels, 256)
        self.batch1 = nn.BatchNorm1d(256)
        self.conv2 = HypergraphConv(256, 256)
        self.batch2 = nn.BatchNorm1d(256)
        self.conv3 = HypergraphConv(256, out_channels)
        self.act = nn.ReLU()

    def forward(self, x, edge):
        x = self.batch1(self.act(self.conv1(x, edge)))
        x = self.batch2(self.act(self.conv2(x, edge)))
        x = self.act(self.conv3(x, edge))
        return x

class BioEncoder(nn.Module):
    def __init__(self, dim_drug, dim_cellline, output, use_GMP=True):
        super(BioEncoder, self).__init__()
        # -------drug_layer
        self.use_GMP = use_GMP
        self.conv1 = GCNConv(dim_drug, 128)
        self.batch_conv1 = nn.BatchNorm1d(128)
        self.conv2 = GCNConv(128, output)
        self.batch_conv2 = nn.BatchNorm1d(output)
        # -------cell line_layer
        self.fc_cell1 = nn.Linear(dim_cellline, 128)
        self.batch_cell1 = nn.BatchNorm1d(128)
        self.fc_cell2 = nn.Linear(128, output)
        self.reset_para()
        self.act = nn.ReLU()

    def reset_para(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return

    def forward(self, drug_feature, drug_adj, ibatch, gexpr_data):
        # -----drug_train
        x_drug = self.conv1(drug_feature, drug_adj)
        x_drug = self.batch_conv1(self.act(x_drug))
        x_drug = self.conv2(x_drug, drug_adj)
        x_drug = self.act(x_drug)
        x_drug = self.batch_conv2(x_drug)
        if self.use_GMP:
            x_drug = global_max_pool(x_drug, ibatch)
        else:
            x_drug = global_mean_pool(x_drug, ibatch)
        # ----cellline_train
        x_cellline = torch.tanh(self.fc_cell1(gexpr_data))
        x_cellline = self.batch_cell1(x_cellline)
        x_cellline = self.act(self.fc_cell2(x_cellline))
        return x_drug, x_cellline


class Decoder(torch.nn.Module):
    def __init__(self, in_channels):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // 2)
        self.batch1 = nn.BatchNorm1d(in_channels // 2)
        self.fc2 = nn.Linear(in_channels // 2, in_channels // 4)
        self.batch2 = nn.BatchNorm1d(in_channels // 4)
        self.fc3 = nn.Linear(in_channels // 4, 1)

        self.reset_parameters()
        self.drop_out = nn.Dropout(0.4)
        self.act = nn.Tanh()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, graph_embed, druga_id, drugb_id, cellline_id):
        h1 = torch.cat((graph_embed[druga_id, :], graph_embed[drugb_id, :], graph_embed[cellline_id, :]), 1)
        h = self.act(self.fc1(h1))
        h = self.batch1(h)
        h = self.drop_out(h)
        h = self.act(self.fc2(h))
        h = self.batch2(h)
        h = self.drop_out(h)
        h = self.fc3(h)
        return torch.sigmoid(h.squeeze(dim=1))
    
class KnowSynergyCL(torch.nn.Module):
    def __init__(self, bio_encoder, graph_encoder, decoder, projection_dim, tau):
        super(KnowSynergyCL, self).__init__()
        self.bio_encoder = bio_encoder
        self.graph_encoder = graph_encoder
        self.decoder = decoder
        self.projection = nn.Sequential(nn.Linear(projection_dim, projection_dim), nn.ReLU(inplace=True), nn.Linear(projection_dim, projection_dim))
        self.tau = tau
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.bio_encoder)
        reset(self.graph_encoder)
        reset(self.projection)
        reset(self.decoder)

    def forward(self, drug_feature, drug_adj, ibatch, gexpr_data, adj, druga_id, drugb_id, cellline_id):
        drug_embed, cellline_embed = self.bio_encoder(drug_feature, drug_adj, ibatch, gexpr_data)
        merge_embed = torch.cat((drug_embed, cellline_embed), 0)
        graph_embed = self.graph_encoder(merge_embed, adj)
        project_embed = self.projection(graph_embed)
        res = self.decoder(graph_embed, druga_id, drugb_id, cellline_id)
        return res, project_embed
    
    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def drug_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, DDI):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
                between_sim.diag() / (
                refl_sim.sum(1) - refl_sim.diag() 
                + between_sim.sum(1) + torch.mul(DDI, refl_sim).sum(1)
                )
            )
    
    def cell_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, CCI):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
                (between_sim.diag() + torch.mul(CCI, refl_sim).sum(1)) / (
                refl_sim.sum(1) - refl_sim.diag() + between_sim.sum(1) 
                )
            )
    
    def loss(self, drug_emb1, cline_emb1, drug_emb2, cline_emb2, DDI, CCI):
        l11 = self.drug_semi_loss(drug_emb1,drug_emb2,DDI)
        l12 = self.drug_semi_loss(drug_emb2,drug_emb1,DDI)

        l21 = self.cell_semi_loss(cline_emb1,cline_emb2,CCI)
        l22 = self.cell_semi_loss(cline_emb2,cline_emb1,CCI)

        ret1 = (l11 + l12) * 0.5
        ret2 = (l21 + l22) * 0.5

        return ret1.mean() + ret2.mean()

