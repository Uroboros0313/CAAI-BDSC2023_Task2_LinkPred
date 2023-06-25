import torch as th
import torch.nn as nn
import torch.nn.functional as F
    

class DisMultGES(nn.Module):
    def __init__(self,
                 ent_vocab_size,
                 rel_vocab_size,
                 embedding_dim,
                 rel_attrs_enc_map,
                 margin,
                 device) -> None:
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.ent_table = nn.Embedding(ent_vocab_size, embedding_dim)
        self.rel_table = nn.Embedding(rel_vocab_size, embedding_dim)
        self.rel_attrs_enc_map = rel_attrs_enc_map
        
        
        for attr_name, map_ in rel_attrs_enc_map.items():
            setattr(self, f'{attr_name}_table', nn.Embedding(len(map_), embedding_dim))

        wi_dim = len(rel_attrs_enc_map) + 1
        self.wi = nn.parameter.Parameter(th.ones((wi_dim, 1), dtype=th.float32), requires_grad=False)
        
        self.margin = margin
        self.device = device
        self._reset_parameters()
    
    def _reset_parameters(self):
        border = 6 / self.embedding_dim
        
        nn.init.uniform_(self.ent_table.weight, -border, border)
        nn.init.uniform_(self.rel_table.weight, -border, border)
        nn.init.xavier_uniform_(self.wi.data)
        
        for attr_name, _ in self.rel_attrs_enc_map.items():
             nn.init.uniform_(getattr(self, f'{attr_name}_table').weight, -border, border)
            
    def forward(self, triplets, rel_attrs):
        heads, rels, tails = triplets[:, 0], triplets[:, 1], triplets[:, 2]
        he = self.forward_head(heads)
        re = self.forward_rel(rels, rel_attrs)
        te = self.forward_tail(tails)
        return th.sum(he * re * te, dim=-1)
    
    def forward_head(self, heads):
        return self.ent_table(heads)
    
    def forward_rel(self, rels, rel_attrs):
        rembs_list = [self.rel_table(rels)]
        
        for i, (attr_name, _) in enumerate(self.rel_attrs_enc_map.items()):
            rembs_list.append(getattr(self, f'{attr_name}_table')(rel_attrs[:, i]))
        
        rembs = th.stack(rembs_list, dim=2)
        re = th.einsum('ben, ni', 
                       rembs, 
                       F.dropout(self.wi, p=0.3, training=self.training)).squeeze(2)
        return re
    
    def forward_tail(self, tails):
        return self.ent_table(tails)
    
    def infer(self, tuplets, rel_attrs):
        heads, rels = tuplets[:, 0], tuplets[:, 1]
        he = self.forward_head(heads)
        re = self.forward_rel(rels, rel_attrs)
        return he * re
    
    def criterion(self, pos_distance, neg_distance):
        _const = th.tensor([1], dtype=th.long, device=self.device)
        loss = F.margin_ranking_loss(pos_distance, 
                                     neg_distance, 
                                     target=_const, 
                                     margin=self.margin,
                                     reduction='mean')
            
        return loss
        