import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class GraphConv(nn.Module):
    def __init__(self,input_dim=128, output_dim=128, gc_drop=0.1):
        super(GraphConv, self).__init__()
        weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        self.weight = nn.init.xavier_normal_(weight, gain=1.414)
        if gc_drop:
            self.gc_drop = nn.Dropout(gc_drop)
        else:
            self.gc_drop = lambda x: x
        self.act = nn.ELU()

    def forward(self, H, adj, activation=None):
        x_hidden = self.gc_drop(torch.mm(H, self.weight))
        H = torch.mm(adj, x_hidden)
        if activation is None:
            outputs = H
        else:
            outputs = self.act(H)
        return outputs

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size=64, output_size=1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=True)  
        self.fc2 = nn.Linear(hidden_size, int(hidden_size/2), bias=True)  
        self.fc3 = nn.Linear(int(hidden_size/2), int(hidden_size/4), bias=True)  
        self.fc4 = nn.Linear(int(hidden_size/4), output_size, bias=True)  
        self.act = nn.ReLU()
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))  
        x = self.act(self.fc3(x)) 
        x = self.fc4(x)
        return x/len(x)

class Model(nn.Module):
    def __init__(self, device, view_num, gnn, ncirc, ndis, hidden_dim=128, drop=0.1):
        super(Model, self).__init__()
        self.device = device
        self.ncirc = ncirc
        self.ndis = ndis
        self.hidden_dim = hidden_dim
        if gnn == 'gcn':
            self.gcn1 = GraphConv(hidden_dim, hidden_dim, drop)  
            self.gcn2 = GraphConv(hidden_dim, hidden_dim, drop)  
            self.gcn3 = GraphConv(hidden_dim, hidden_dim, drop)  
            self.gcn4 = GraphConv(hidden_dim, hidden_dim, drop)
        self.mlp = MLP(hidden_dim*2, hidden_dim, 1)  
        self.view_num = view_num  
        self.weight = torch.nn.Parameter(torch.FloatTensor(view_num, 1), requires_grad=True)  
        torch.nn.init.uniform_(self.weight,a = 0,b = 0.155)  
        self.weight2=torch.nn.Parameter(torch.FloatTensor(view_num, 1), requires_grad=True) 
        torch.nn.init.uniform_(self.weight2, a=0, b=0.155) 
        self.embedding = torch.nn.Embedding(self.ncirc + self.ndis, self.hidden_dim).to(self.device)
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, mA, features, i):

        u_i_embed = features.to(self.device)
        adj= torch.stack(mA).to_dense().to(self.device)

        adj_D = (adj * self.weight.view(-1, 1, 1)).sum(dim=0)
        D_embed1 = self.gcn1(u_i_embed, adj_D)  
        D_embed2 =  self.gcn2(D_embed1, adj_D)

        all = (D_embed1 + D_embed2) / 2

        for view in adj:
            embed1 = self.gcn3(u_i_embed, view) 
            embed2 = self.gcn4(embed1, view)  
            all = all + (embed1 + embed2) / 2

        all = all / (self.view_num + 1)

        u_embed, i_embed = torch.split(all, [self.ncirc, self.ndis], 0) 
        u_embed_expanded = u_embed.unsqueeze(1).expand(-1, self.ndis, -1)  
        i_embed_expanded = i_embed.unsqueeze(0).expand(self.ncirc, -1, -1)  
        user_item_pairs_tensor = torch.cat((u_embed_expanded, i_embed_expanded), dim=2) 
        user_item_pairs_tensor = user_item_pairs_tensor.reshape(-1, user_item_pairs_tensor.shape[2])
  
        scores = self.mlp(user_item_pairs_tensor)  
        scores = scores.reshape(self.ncirc, self.ndis)  
        
        return scores
    

    def loss(self, ls, score, pos_dict, neg_dict, temperature=0.1):
        score =  1.0 / (1.0 + torch.exp(-score))
        total_loss = 0

        if ls == 'info_nce':
            for user_id in range(self.ncirc):
                pos_items = pos_dict[user_id]
                neg_items = neg_dict[user_id]
                
                pos_scores = score[user_id, pos_items].to(self.device)
                neg_scores = score[user_id, neg_items].to(self.device)
                for j in pos_scores:
                    loss = self.info_nce(j.unsqueeze(0), neg_scores, temperature)
                total_loss += loss

            total_loss /= self.ncirc
        
        return total_loss
    
    def info_nce(self, pos_scores, neg_scores, temperature):
       
        logits = torch.cat([pos_scores.unsqueeze(0), neg_scores.unsqueeze(0)], dim=1).float()
        
        labels = torch.zeros(pos_scores.shape[0], dtype=torch.long).to(self.device)
        
        loss = F.cross_entropy(logits / temperature, labels)

        return loss
    