
from transformers import BertTokenizer
from transformers import BertModel, AdamW
from transformers import get_linear_schedule_with_warmup

import torch
import torch.nn as nn
import torch.nn.functional as F

from loc_utils.model_util import Modality

from typing import Iterable, Dict, List
from torch import Tensor

# import sys
# import os
# sys.path.append(os.path.dirname(__file__)) 

# from mlp_pure import GLUMLP
from scattermoe.mlp import GLUMLP

class BertForSurvivalAnalysis(nn.Module):
    def __init__(self, pretrain_model_path, n_classes):
        # n_classes is the number of survival time bins
        super(BertForSurvivalAnalysis, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model_path)
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]  # The pooled output from the [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # Output time-bin predictions
        return logits
    

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context = None):
        B, N, C = x.shape
        if context is None:
            context = x
        # print(x.shape, context.shape)
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(context).reshape(B, context.shape[1], 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v  = kv.unbind(0) 

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, 
                 mlp_ratio=4., qkv_bias=False, 
                 drop=0., attn_drop=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # print(self.attn.q.capacity_per_expert, self.mlp.capacity_per_expert)

        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    
class CrossTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, 
                 mlp_ratio=4., qkv_bias=False, 
                 drop=0., attn_drop=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, ox):
        # print(self.attn.q.capacity_per_expert, self.mlp.capacity_per_expert)
        x = x + self.attn(self.norm1(x), ox)
        x = x + self.mlp(self.norm2(x))
        return x
    
class QFormerEncoder(nn.Module):
    def __init__(self, feature_size, num_patches, embed_dim, attn_drop=0.25, drop = 0.25, num_heads=4, qkv_bias=False, mlp_ratio = 4, act_layer=nn.GELU):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(num_patches, embed_dim))

        self.context_norm = nn.LayerNorm(feature_size)
        self.context_proj = nn.Linear(feature_size, embed_dim)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)

        self.mlp = Mlp(in_features=embed_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x:torch.Tensor):

        x = self.context_proj(self.context_norm(x))
        
        B, C, D = x.shape
        
        query = self.query_tokens.unsqueeze(0).repeat(B, 1, 1)
        x = query + self.attn(self.norm1(query), x)
        x = x + self.mlp(self.norm2(x))
        return x


    
class VanillaTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, 
                 mlp_ratio=4., qkv_bias=False, 
                 drop=0., attn_drop=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # print(self.attn.q.capacity_per_expert, self.mlp.capacity_per_expert)

        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = drop

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs)
        

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    
class SparseMoeBlock(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_local_experts, num_experts_per_tok, act_layer,n_modality = 2):
        super().__init__()
        self.hidden_dim = hidden_size
        self.ffn_dim = intermediate_size
        self.num_experts = num_local_experts
        self.top_k = num_experts_per_tok

        
        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        self.idx2feature = nn.ParameterDict(
           {f"{i}":nn.Parameter(torch.randn(hidden_size)) for i in range(n_modality)}
        )

        self.moe_mlp = GLUMLP(
            input_size=self.hidden_dim,
            hidden_size=self.ffn_dim,
            num_experts=self.num_experts,
            top_k=self.top_k,
            activation=act_layer()
        )
        self.cur_modality_idx = 0

    def forward(self, hidden_states: torch.Tensor):
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        # print(self.hidden_dim, hidden_states.shape, self.idx2feature[f'{self.cur_modality_idx}'].shape)

        # gate_input = torch.cat([hidden_states,self.idx2feature[f'{self.cur_modality_idx}'].repeat(hidden_states.shape[0], 1)], dim=-1)
        gate_input = hidden_states + self.idx2feature[f'{self.cur_modality_idx}']
        # print(gate_input.shape)
        router_logits = self.gate(gate_input)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)
        final_hidden_states = self.moe_mlp(hidden_states, routing_weights, selected_experts)
        final_hidden_states = final_hidden_states.view(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

def load_balancing_loss_func(gate_logits: torch.Tensor, num_experts = 8, top_k=2) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits (Union[`torch.Tensor`, Tuple[torch.Tensor]):
            Logits from the `gate`, should be a tuple of tensors. Shape: [batch_size, seqeunce_length, num_experts].
        num_experts (`int`, *optional*):
            Number of experts

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None:
        return 0

    if isinstance(gate_logits, tuple):
        # cat along the layers?
        gate_logits = torch.cat(gate_logits, dim=0)

    routing_weights, selected_experts = torch.topk(gate_logits, top_k, dim=-1)
    routing_weights = routing_weights.softmax(dim=-1)

    # cast the expert indices to int64, otherwise one-hot encoding will fail
    if selected_experts.dtype != torch.int64:
        selected_experts = selected_experts.to(torch.int64)

    if len(selected_experts.shape) == 2:
        selected_experts = selected_experts.unsqueeze(2)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    # For a given token, determine if it was routed to a given expert.
    expert_mask = torch.max(expert_mask, axis=-2).values

    # cast to float32 otherwise mean will fail
    expert_mask:torch.Tensor = expert_mask.to(torch.float32)
    tokens_per_group_and_expert = torch.mean(expert_mask, axis=-2)

    router_prob_per_group_and_expert = torch.mean(routing_weights, axis=-1)
    return torch.mean(tokens_per_group_and_expert * router_prob_per_group_and_expert.unsqueeze(-1)) * (num_experts**2)

class SparseTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, 
                 mlp_ratio=4., qkv_bias=False, 
                 drop=0., attn_drop=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_experts=16, topk=2, n_modality= 2):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.num_experts = num_experts
        self.topk = topk
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        # self.gate = nn.Linear(dim, num_experts, bias=False)
        self.mlp = SparseMoeBlock(
            dim, mlp_hidden_dim, num_experts, topk, act_layer, n_modality=n_modality
        )
        self.router_logits = []

    def calc_balance_loss(self):
        if len(self.router_logits) < 1:
            return
        blloss = load_balancing_loss_func(torch.cat(self.router_logits, dim=0), num_experts=self.num_experts, top_k=self.topk)
        self.router_logits = []
        return blloss

    def forward(self, x, modality_length=None):
        x = x + self.attn(self.norm1(x))
        if modality_length is None:
            raise ValueError("modality_length must be provided for SparseTransformerBlock forward.")
        x_list = x.split(modality_length, dim=1)
        mlp_x = []
        for i, item in enumerate(x_list):
            self.mlp.cur_modality_idx = i
            i_mlp_x, router_logits = self.mlp(self.norm2(item))
            self.router_logits.append(router_logits)
            mlp_x.append(i_mlp_x)
        mlp_x = torch.cat(mlp_x, dim=1)
        x = x + mlp_x
        return x
    
class EarlyFusion(nn.Module):
    def __init__(self, device, modalities: Dict[str, Modality], hidden_size, 
                 dropout_rate = 0.1, pred_dim = 20, 
                 mlp_ratio = 4, 
                 n_token = 16,
                 n_backbone = 3,
                 n_head = 4) -> None:
        super().__init__()
        self.device = device
        input_dim = sum([hidden_size for item in modalities])
        
        self.m_projector = nn.ModuleDict(
            {
                mm: nn.Sequential(
                    QFormerEncoder(modalities[mm].feature_dim, n_token, hidden_size)
                ) for mm in modalities
            }
        )
        
        self.backbone = nn.ModuleList(
            [VanillaTransformerBlock(hidden_size, n_head, mlp_ratio, drop=dropout_rate, attn_drop=dropout_rate
                                     ) for i in range(n_backbone)]
        )
        
        self.head = nn.Linear(hidden_size, pred_dim)
        
        self.logits_dim = pred_dim
    
    def forward(self, all_modalities):
        # early fusion
        input_data = [self.m_projector[item](all_modalities[item]) for item in all_modalities]
        
        input_data = torch.cat(input_data, dim=1)
        
        backbone_x = input_data
        for i, layer in enumerate(self.backbone):
            backbone_x = layer(backbone_x)
        
        logits = self.head(backbone_x.mean(dim=1))
        
        return logits
    
class LateFusion(nn.Module):
    def __init__(self, device, modalities: Dict[str, Modality], hidden_size, 
                 dropout_rate = 0.1, pred_dim = 20, mlp_ratio = 4,
                 n_token = 16,
                 n_backbone = 3,
                 n_head = 4) -> None:
        super().__init__()
        self.device = device
        input_dim = sum([hidden_size for item in modalities])
        
        self.m_projector = nn.ModuleDict(
            {
                mm: nn.Sequential(
                    QFormerEncoder(modalities[mm].feature_dim, n_token, hidden_size)
                ) for mm in modalities
            }
        )
        self.backbone = nn.ModuleDict(
            {
                mm: nn.Sequential(*[VanillaTransformerBlock(hidden_size, 
                                                           n_head, 
                                                           mlp_ratio, 
                                                           drop=dropout_rate, 
                                                           attn_drop=dropout_rate) for _ in range(n_backbone)]) for mm in modalities
            }
        )
        
        self.head = nn.Linear(input_dim, pred_dim)
        
        self.logits_dim = pred_dim
    
    def forward(self, all_modalities):
        # early fusion
        input_data = {item: self.m_projector[item](all_modalities[item]) for item in all_modalities}
        
        input_data = {mm: self.backbone[mm](input_data[mm]) for mm in input_data}
        input_list = [input_data[mm].mean(dim=1) for mm in input_data]
        input_data = torch.cat(input_list, dim=-1)
        
        logits = self.head(input_data)
        
        return logits
    
class CrossAttnFusion(nn.Module):
    def __init__(self, device, modalities: Dict[str, Modality], hidden_size, 
                 dropout_rate = 0.1, pred_dim = 20, mlp_ratio = 4,
                 n_token = 16,
                 n_backbone = 3,
                 n_head = 4) -> None:
        super().__init__()
        self.device = device
        input_dim = sum([hidden_size for item in modalities])
        
        self.m_projector = nn.ModuleDict(
            {
                mm: nn.Sequential(
                    QFormerEncoder(modalities[mm].feature_dim, n_token, hidden_size)
                ) for mm in modalities
            }
        )
        n_modality = len(modalities)
        self.cross_attn = nn.ModuleList(
            [CrossTransformerBlock(hidden_size, 4) for i in range(n_modality)]
        )
        
        self.head = nn.Linear(input_dim, pred_dim)
        
        self.logits_dim = pred_dim
    
    def forward(self, all_modalities):
        # early fusion
        input_data = [self.m_projector[item](all_modalities[item]) for item in all_modalities]
        
        tf_hidden = input_data
        for i, layer in enumerate(self.cross_attn):
            tf_tmp = []
            for k, m1 in enumerate(tf_hidden):
                cross_m2 = torch.cat(tf_hidden[:i]+tf_hidden[i+1:], dim=1)
                tf_tmp.append(
                    layer(tf_hidden[i], cross_m2)
                )
            tf_hidden = tf_tmp
        
        head_input = [item.mean(dim=1) for item in tf_hidden]
        head_input = torch.cat(head_input, dim=-1)
        logits = self.head(head_input)
        
        return logits

class MulTFusionAlign(nn.Module):
    def __init__(self, modalities, dim, num_heads, 
                 mlp_ratio=4., qkv_bias=False, 
                 drop=0., attn_drop=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.crossmodel_transformer = nn.ModuleDict(
            {f"{m1}_{m2}": CrossTransformerBlock(
                dim,
                num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                act_layer=act_layer,
                norm_layer=norm_layer) for m2 in modalities for m1 in modalities}
        )
        self.transformers = nn.ModuleDict({kk: nn.Sequential(*[VanillaTransformerBlock(
                dim * (len(modalities)-1),
                num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                act_layer=act_layer,
                norm_layer=norm_layer) for _ in range(3)]) for kk in modalities}
        )
        
        self.projector = nn.ModuleDict(
            {kk: nn.Linear(dim * (len(modalities)-1), dim * (len(modalities)-1)) for kk in modalities}
        )

    def forward(self, modalities):
        # print(self.attn.q.capacity_per_expert, self.mlp.capacity_per_expert)
        m_out = {}
        for m in modalities:
            m_cross = []
            for mm in modalities:
                if m == mm:
                    continue
                m_feature = modalities[m]
                layer = self.crossmodel_transformer[f"{m}_{mm}"]
                m_feature = layer(modalities[m], modalities[mm])
                m_cross.append(
                    m_feature
                )
            m_out[m] = self.projector[m](self.transformers[m](torch.cat(m_cross, dim=-1)))
        return m_out

class MultFusion(nn.Module):
    def __init__(self, device, modalities: Dict[str, Modality], hidden_size, 
                 dropout_rate = 0.1, pred_dim = 20, mlp_ratio = 4,
                 n_token = 16,
                 n_backbone = 3,
                 n_head = 4) -> None:
        
        super().__init__()
        self.device = device
        self.m_projector = nn.ModuleDict(
            {
                mm: nn.Sequential(
                    QFormerEncoder(modalities[mm].feature_dim, n_token, hidden_size)
                ) for mm in modalities
            }
        )
        
        self.encoder = MulTFusionAlign(
            modalities, hidden_size, n_head, mlp_ratio, drop=dropout_rate, attn_drop=dropout_rate
        )
        # input_dim = sum([hidden_size for item in modalities])
        self.head = nn.Linear(hidden_size * (len(modalities)-1) * len(modalities), pred_dim)
        self.logits_dim = pred_dim
        
    def forward(self, modalites):
        modalites = {item: self.m_projector[item](modalites[item]) for item in modalites}
        
        mult_out = self.encoder(modalites)
        multi_list = [mult_out[mm].mean(dim=1) for mm in mult_out]
        multi_list = torch.cat(multi_list, dim=-1)
        
        return self.head(multi_list)
    
class MAGGate(nn.Module):
    def __init__(self, inp1_size, inp2_size, inp3_size, dropout):
        super(MAGGate, self).__init__()

        self.fc12 = nn.Linear(inp1_size + inp2_size, 1)
        self.fc13 = nn.Linear(inp1_size + inp3_size, 1)
        
        self.fc2 = nn.Linear(inp2_size, inp1_size)
        self.fc3 = nn.Linear(inp3_size, inp1_size)
        
        self.beta = nn.Parameter(torch.randn((1,)))
        self.norm = nn.LayerNorm(inp1_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp1, inp2, inp3):
        w2 = torch.sigmoid(self.fc12(torch.cat([inp1, inp2], -1)))
        w3 = torch.sigmoid(self.fc13(torch.cat([inp1, inp3], -1)))
        
        adjust = self.fc2(w2 * inp2) + self.fc3(w3 * inp3)
        
        one = torch.tensor(1).type_as(adjust)
        alpha = torch.min(torch.norm(inp1) / torch.norm(adjust) * self.beta, one)
        output = inp1 + alpha * adjust
        output = self.dropout(self.norm(output))
        return output
    
class MAGFusion(nn.Module):
    def __init__(self, device, modalities: Dict[str, Modality], hidden_size, 
                 dropout_rate = 0.1, pred_dim = 20, mlp_ratio = 4,
                 n_token = 16,
                 n_backbone = 3,
                 n_head = 4) -> None:
        super(MAGFusion, self).__init__()
        self.device = device
        self.m_projector = nn.ModuleDict(
            {
                mm: nn.Sequential(
                    QFormerEncoder(modalities[mm].feature_dim, n_token, hidden_size)
                ) for mm in modalities
            }
        )
        self.central_modality = 'img'
        
        # self.mag_fusion = {}
        # for mm in modalities:
        #     if mm != self.central_modality:
        #         self.mag_fusion[mm] = MAGGate(hidden_size, hidden_size, dropout_rate)
        # self.mag_fusion = nn.ModuleDict(self.mag_fusion)
        self.mag_fusion = MAGGate(hidden_size, hidden_size, hidden_size, dropout_rate)
        self.backbone = nn.Sequential(*[VanillaTransformerBlock(hidden_size, 
                                                           n_head, 
                                                           mlp_ratio, 
                                                           drop=dropout_rate, 
                                                           attn_drop=dropout_rate) for _ in range(n_backbone)])
        
        self.head = nn.Linear(hidden_size, pred_dim)
        self.logits_dim = pred_dim
    
    def forward(self, all_modalities):
        # early fusion
        input_data = {item: self.m_projector[item](all_modalities[item]) for item in all_modalities}
        
        fused_feature = input_data[self.central_modality]
        other_modalities = []
        for mm in input_data:
            if mm != self.central_modality:
                other_modalities.append(input_data[mm])
        fused_feature = self.mag_fusion(fused_feature, other_modalities[0], other_modalities[1])

        input_data = self.backbone(fused_feature)
        input_data = input_data.mean(dim=1)
        
        logits = self.head(input_data)
        
        return logits
    
class TensorFusionOPT(nn.Module):
    """
    Implementation of TensorFusion Networks.
    
    See https://github.com/Justin1904/TensorFusionNetworks/blob/master/model.py for more and the original code.
    """
    def __init__(self):
        """Instantiates TensorFusion Network Module."""
        super().__init__()

    def forward(self, modalities):
        """
        Forward Pass of TensorFusion.
        
        :param modalities: An iterable of modalities to combine. 
        """
        if len(modalities) == 1:
            return modalities[0]

        mod0 = modalities[0]
        nonfeature_size = mod0.shape[:-1]

        m = torch.cat([torch.ones(*nonfeature_size, 1).type(mod0.dtype).to(mod0.device), mod0], dim=-1)
        for mod in modalities[1:]:
            mod = torch.cat([torch.ones(
                *nonfeature_size, 1).type(mod.dtype).to(mod.device), mod], dim=-1)
            fused = torch.einsum('...i,...j->...ij', m, mod)
            # print(fused.shape)
            m = fused.reshape([*nonfeature_size, -1])

        return m
    
class TensorFusion(nn.Module):
    def __init__(self, device, modalities: Dict[str, Modality], hidden_size, 
                 dropout_rate = 0.1, pred_dim = 20, mlp_ratio = 4,
                 n_token = 16,
                 n_backbone = 3,
                 n_head = 4) -> None:
        super(TensorFusion, self).__init__()
        self.device = device
        self.m_projector = nn.ModuleDict(
            {
                mm: nn.Sequential(
                    QFormerEncoder(modalities[mm].feature_dim, n_token, hidden_size)
                ) for mm in modalities
            }
        )
        self.central_modality = 'img'
        
        self.tensor_fusion_opt = TensorFusionOPT()
        self.tensor_fusion_norm = nn.LayerNorm((hidden_size + 1) ** len(modalities))
        self.tensor_fusion_projector = nn.Linear(
            (hidden_size + 1) ** len(modalities), hidden_size
        )
        
        self.backbone = nn.Sequential(*[VanillaTransformerBlock(hidden_size, 
                                                           n_head, 
                                                           mlp_ratio, 
                                                           drop=dropout_rate, 
                                                           attn_drop=dropout_rate) for _ in range(n_backbone)])
        
        self.head = nn.Linear(hidden_size, pred_dim)
        self.logits_dim = pred_dim
    
    def forward(self, all_modalities):
        # early fusion
        input_data = {item: self.m_projector[item](all_modalities[item]) for item in all_modalities}
        
        tf_input = [input_data[mm] for mm in input_data]
        tf_out = self.tensor_fusion_opt(tf_input)
        tf_out = self.tensor_fusion_projector(self.tensor_fusion_norm(tf_out))
        # # print(tf_out.shape)
        
        # input_data = self.backbone(tf_out)
        input_data = tf_out.mean(dim=1)
        
        logits = self.head(input_data)
        
        return logits
    
    
def contrastive_loss(modality_1, modality_2, temperature=0.5):
    if type(temperature) is torch.Tensor:
            with torch.no_grad(): 
                temperature.clamp_(0.001, 0.5)
    # Modality: Batch x Feature Dimension
    modality_1 = F.normalize(modality_1, dim=-1)
    modality_2 = F.normalize(modality_2, dim=-1)
    pos_idx = torch.arange(len(modality_1)).view(-1, 1).to(modality_1.device)
    pos_idx = torch.eq(pos_idx, pos_idx.t()).float()
    sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

    sim_1to2 = modality_1 @ modality_2.T / temperature
    sim_2to1 = modality_2 @ modality_1.T / temperature

    loss = -torch.sum(F.log_softmax(sim_1to2, dim=1) * sim_targets, dim=1).mean() - torch.sum(F.log_softmax(sim_2to1, dim=1) * sim_targets, dim=1).mean()

    return loss / 2
    
class LiMoEFusion(nn.Module):
    def __init__(self, device, modalities: Dict[str, Modality], hidden_size, 
                 dropout_rate=0.1, pred_dim=20, mlp_ratio=4,
                 n_token=16, n_backbone=3, n_head=4) -> None:
        super(LiMoEFusion, self).__init__()
        self.device = device
        self.modalities = list(modalities.keys())  # 保存顺序

        self.m_projector = nn.ModuleDict({
            mm: nn.Sequential(
                QFormerEncoder(modalities[mm].feature_dim, n_token, hidden_size)
            ) for mm in modalities
        })

        self.backbone = nn.Sequential(*[
            SparseTransformerBlock(
                hidden_size,
                n_head,
                mlp_ratio,
                drop=dropout_rate,
                attn_drop=dropout_rate,
                num_experts=4,
                n_modality=len(modalities)
            ) for _ in range(n_backbone)
        ])
        
        self.head = nn.Linear(hidden_size * len(modalities), pred_dim)
        self.logits_dim = pred_dim

    def calc_limoe_loss(self, modality_x_feature):
        loss_list = []
        for i in range(len(modality_x_feature)):
            for j in range(i + 1, len(modality_x_feature)):
                loss_list.append(contrastive_loss(
                    modality_x_feature[i].mean(dim=-2),  # mean over tokens
                    modality_x_feature[j].mean(dim=-2)
                ))
        return sum(loss_list)

    def get_router_loss(self):
        tmp_loss = 0
        for mn, mm in self.named_modules():
            if hasattr(mm, 'calc_balance_loss'):
                tmp_loss += mm.calc_balance_loss() or 0
        return tmp_loss

    def forward(self, all_modalities):
        # 1. 对每个模态做 QFormer 投影
        input_data = {
            mm: self.m_projector[mm](all_modalities[mm])
            for mm in all_modalities
        }

        # 2. 拼接所有模态，并记录每个模态的 token 数
        input_list = [input_data[mm] for mm in self.modalities]
        modality_lengths = [x.shape[1] for x in input_list]  # 每个模态的 token 数
        x_cat = torch.cat(input_list, dim=1)  # dim=1 表示时间维度拼接

        # 3. 送入 backbone，逐层传入 modality_lengths
        for blk in self.backbone:
            x_cat = blk(x_cat, modality_length=modality_lengths)

        # 4. 为 LiMoE 损失做准备：把拼接前的表示也保留
        limoe_loss = self.calc_limoe_loss(input_list)
        router_loss = self.get_router_loss()
        self.ret_loss = limoe_loss + router_loss

        # 5. 拼接所有模态表示的平均值（取 token 平均）
        x_split = torch.split(x_cat, modality_lengths, dim=1)
        pooled = [m.mean(dim=1) for m in x_split]
        fused = torch.cat(pooled, dim=-1)

        # 6. 输出
        logits = self.head(fused)
        return logits


class TFFusionAlign(nn.Module):
    def __init__(self, modalities, dim, num_heads, 
                 mlp_ratio=4., qkv_bias=False, 
                 drop=0., attn_drop=0., n_layer = 4,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_norm=True):
        super().__init__()
        self.n_layer = n_layer
        self.use_norm = use_norm
        if use_norm:
            self.norms = nn.ModuleDict(
                {kk: norm_layer(dim) for kk in modalities}
            )

        self.m_spc_transformer = nn.ModuleDict(
            {kk: nn.Sequential(*[VanillaTransformerBlock(
                dim,
                num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                act_layer=act_layer,
                norm_layer=norm_layer) for _ in range(n_layer)]) for kk in modalities}
        )


    def forward(self, modalities):
        # print(self.attn.q.capacity_per_expert, self.mlp.capacity_per_expert)
        m_out = {}
        for m in modalities:
            if self.use_norm:
                m_spc_in = self.norms[m](modalities[m])
            else:
                m_spc_in = modalities[m]
            m_out[m] = self.m_spc_transformer[m](m_spc_in)
        return m_out

class CancerMoE(nn.Module):
    def __init__(self, device, modalities: Dict[str, Modality], hidden_size, 
                 dropout_rate = 0.1, pred_dim = 20, mlp_ratio = 4,
                 n_token = 16,
                 n_backbone = 1,
                 n_head = 4,
                 num_experts = 4,
                 topk=2) -> None:
        super(CancerMoE, self).__init__()
        self.device = device
        self.m_projector = nn.ModuleDict(
            {
                mm: nn.Sequential(
                    QFormerEncoder(modalities[mm].feature_dim, n_token, hidden_size),
                    # nn.LayerNorm(hidden_size)
                ) for mm in modalities
            }
        )

        self.mult_align = TFFusionAlign(modalities, hidden_size, n_head, mlp_ratio, False, dropout_rate, dropout_rate, n_layer=1, use_norm = False)
        
        self.backbone = nn.ModuleList([
            SparseTransformerBlock(hidden_size, 
                                n_head, 
                                mlp_ratio, 
                                drop=dropout_rate, 
                                attn_drop=dropout_rate,
                                n_modality=len(modalities),
                                num_experts=num_experts,
                                topk=topk) if i//2==0 else 
            VanillaTransformerBlock(hidden_size, n_head,
                                mlp_ratio, drop=dropout_rate, 
                                attn_drop=dropout_rate,) 
            for i in range(n_backbone)
        ])

        
        self.head = nn.Linear(hidden_size * len(modalities), pred_dim)
        self.logits_dim = pred_dim
    
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Initialize weights with a Gaussian distribution (mean=0, std=0.1)
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                # Initialize biases with a Gaussian distribution (mean=0, std=0.1)
                if m.bias is not None:
                    nn.init.normal_(m.bias, mean=0.0, std=0.1)
    
    def calc_limoe_loss(self, modality_x_feature):
        loss_list = []
        for i in range(len(modality_x_feature)):
            for j in range(i+1, len(modality_x_feature)):
                feature_dim = modality_x_feature[i].shape[-1]
                loss_list.append(contrastive_loss(modality_x_feature[i].mean(dim=-2), modality_x_feature[j].mean(dim=-2)))
        
        return sum(loss_list)
    
    def get_router_loss(self):
        tmp_loss = 0
        for mn, mm in self.named_modules():
            if hasattr(mm, 'calc_balance_loss'):
                tmp_loss += mm.calc_balance_loss()
        return tmp_loss
    
    # def set_moe_modality(self, modality_length):
    #     for item in self.backbone:
    #         if type(item.mlp) == SparseMoeBlock:
    #             item.modality_length = modality_length

    
    def forward(self, all_modalities):
        # early fusion
        # for mm in all_modalities:
        #     print(mm, all_modalities[mm].min(), all_modalities[mm].max())
        input_data = {item: self.m_projector[item](all_modalities[item]) for item in all_modalities}
        
        input_data = self.mult_align(input_data)
        
        input_list = [input_data[mm] for mm in input_data]
        input_length = [input_data[mm].shape[1] for mm in input_data]
        modality_list = [mm for mm in input_data]
        
        backbone_input = torch.cat(input_list, dim=1)
        for blk in self.backbone:
            if isinstance(blk, SparseTransformerBlock):
                backbone_input = blk(backbone_input, modality_length=input_length)
            else:
                backbone_input = blk(backbone_input)

        
        limoe_loss = self.calc_limoe_loss(input_list)
        
        router_loss = self.get_router_loss()
        self.ret_loss = limoe_loss + router_loss
        
        # input_data = [input_data[mm].mean(dim=1) for mm in input_data]
        # input_data = torch.cat(input_data, dim=-1)
        input_data = backbone_input.mean(dim=1)
        input_data_list = backbone_input.split(input_length, dim=1)
        input_data_list = [item.mean(dim=1) for item in input_data_list]
        input_data = torch.cat(input_data_list, dim=-1)
        logits = self.head(input_data)
        
        return logits

class GatedFusion(nn.Module):
    def __init__(self, hidden_dim, num_modalities):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities
        self.gate = nn.Linear(hidden_dim * num_modalities, num_modalities)

    def forward(self, reps):
        reps_list = [r for r in reps if r is not None]
        if len(reps_list) == 1:
            return reps_list[0]  # only one valid modality

        all_reps = []
        masks = []
        for r in reps:
            if r is None:
                zero_tensor = torch.zeros_like(reps_list[0])
                all_reps.append(zero_tensor)
                masks.append(0)
            else:
                all_reps.append(r)
                masks.append(1)

        cat = torch.cat(all_reps, dim=-1)  # [B, hidden_dim * M]
        gate_logits = self.gate(cat)       # [B, M]

        # 设置无效模态的 gate 权重为 -inf
        mask_tensor = torch.tensor(masks, device=cat.device, dtype=torch.float32).unsqueeze(0)  # [1, M]
        inf_mask = (mask_tensor == 0).float() * (-1e9)
        gate_logits = gate_logits + inf_mask  # 无效模态 logits = -1e9

        gate_weights = torch.softmax(gate_logits, dim=-1)  # [B, M]
        gated = torch.stack(all_reps, dim=1)  # [B, M, D]
        gate_weights = gate_weights.unsqueeze(-1)  # [B, M, 1]
        return (gate_weights * gated).sum(dim=1)  # [B, D]

class SurvivalHead(nn.Module):  # ✅ 新增
    def __init__(self, input_dim, n_bins):
        super().__init__()
        self.hazard_layer = nn.Linear(input_dim, n_bins)

    def forward(self, x):
        hazard = self.hazard_layer(x) # [B, T]
        surv = torch.cumprod(1 - hazard, dim=1)        # [B, T]
        return hazard, surv

class MainModalityMoE(nn.Module):
    def __init__(self, device, modalities, hidden_size, dropout_rate=0.1, 
                 pred_dim=15, mlp_ratio=4, n_token=16, n_backbone=1, 
                 n_head=4, num_experts=4, topk=1, cancer_types=None):
        super(MainModalityMoE, self).__init__()
        self.device = device
        self.modalities = list(modalities.keys())
        self.cancer_types = cancer_types or ['Default']

        self.m_projector = nn.ModuleDict({
            mm: nn.Sequential(
                nn.Linear(modalities[mm].feature_dim, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ) for mm in self.modalities
        })

        self.fusion = GatedFusion(hidden_size, num_modalities=len(self.modalities))

        self.backbone = nn.Sequential(
            *[nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=n_head,
                dim_feedforward=hidden_size * mlp_ratio,
                dropout=dropout_rate,
                activation='gelu',
                batch_first=True
            ) for _ in range(n_backbone)]
        )

        # ✅ 替换：生存分析专用 head
        # self.surv_head = SurvivalHead(hidden_size, n_bins=pred_dim)
        self.surv_heads = nn.ModuleDict({
            ct: SurvivalHead(hidden_size, n_bins=pred_dim) for ct in self.cancer_types
        })
        self.logits_dim = pred_dim

    def forward(self, all_modalities, cancer_type='Default'):
        input_data = {}
        for mm in self.m_projector.keys():
            if mm not in all_modalities:
                input_data[mm] = None
                continue

            x = all_modalities[mm]
            if x.dim() > 2:
                x = x.mean(dim=1)

            mask_key = f"{mm}_mask"
            if mask_key in all_modalities and all_modalities[mask_key].sum() == 0:
                input_data[mm] = None
            else:
                input_data[mm] = self.m_projector[mm](x)

        input_list = [input_data.get(mm, None) for mm in self.m_projector.keys()]
        fused = self.fusion(input_list)  # [B, D]
        fused = fused.unsqueeze(1)       # [B, 1, D]

        backbone_out = self.backbone(fused)  # [B, 1, D]
        pooled = backbone_out.mean(dim=1)    # [B, D]

        hazards = []
        survs = []
        for i, ct in enumerate(cancer_type):
            out = self.surv_heads[ct](pooled[i].unsqueeze(0))  # [1, T]
            hazards.append(out[0])
            survs.append(out[1])

        hazard = torch.cat(hazards, dim=0)  # [B, T]
        surv = torch.cat(survs, dim=0)      # [B, T]
        return hazard, surv

class DeformableCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, num_points=4):
        super().__init__()
        self.num_heads = num_heads
        self.num_points = num_points
        self.embed_dim = embed_dim

        self.offset_mlp = nn.Linear(embed_dim, num_heads * num_points)
        self.attn_mlp = nn.Linear(embed_dim, num_heads * num_points)
        self.proj = nn.Linear(embed_dim * num_heads, embed_dim)

    def forward(self, query, key, value):
        B, Lq, C = query.shape
        if key is None or value is None:
            return query

        Lk = key.size(1)
        offset_logits = self.offset_mlp(query)
        attn_logits = self.attn_mlp(query)

        offset_logits = offset_logits.view(B, Lq, self.num_heads, self.num_points)
        attn_logits = attn_logits.view(B, Lq, self.num_heads, self.num_points)

        sampling_locations = offset_logits.sigmoid() * (Lk - 1)
        sampling_locations = sampling_locations.long().clamp(0, Lk - 1)

        sampled_keys = []
        sampled_values = []
        for b in range(B):
            sampled_keys.append(key[b][sampling_locations[b]])
            sampled_values.append(value[b][sampling_locations[b]])
        sampled_keys = torch.stack(sampled_keys, dim=0)
        sampled_values = torch.stack(sampled_values, dim=0)

        attn_scores = (query.unsqueeze(2).unsqueeze(3) * sampled_keys).sum(-1) / (C ** 0.5)
        attn_scores = attn_scores + attn_logits
        attn_weights = torch.softmax(attn_scores, dim=-1)

        attn_out = (attn_weights.unsqueeze(-1) * sampled_values).sum(dim=3)
        attn_out = attn_out.flatten(2)

        return self.proj(attn_out)

class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, num_points=4):
        super().__init__()
        self.self_attn = DeformableCrossAttention(d_model, nhead, num_points)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def forward(self, src):
        src2 = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class ImageProjector(nn.Module):
    def __init__(self, input_dim, n_token, hidden_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(n_token)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.pool(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x

class DenseCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value):
        if key is None or value is None:
            return query

        attn_output, _ = self.attn(query, key, value)
        return self.proj(attn_output)


class MainModalityDeformableMoE(nn.Module):
    def __init__(self, device, modalities, hidden_size, dropout_rate=0.1, pred_dim=20, mlp_ratio=4, 
                 n_token=16, n_backbone=1, n_head=4, num_experts=4, topk=2, num_points=8):
        super(MainModalityDeformableMoE, self).__init__()
        self.device = device
        self.modalities = list(modalities.keys())

        self.m_projector = nn.ModuleDict({
            mm: nn.Sequential(
                nn.Linear(modalities[mm].feature_dim, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ) for mm in self.modalities
        })

        self.fusion = GatedFusion(hidden_size, num_modalities=len(self.modalities))

        self.backbone = nn.Sequential(
            *[DeformableTransformerEncoderLayer(
                d_model=hidden_size,
                nhead=n_head,
                dim_feedforward=hidden_size * mlp_ratio,
                dropout=dropout_rate,
                num_points=num_points
            ) for _ in range(n_backbone)]
        )

        self.head = nn.Linear(hidden_size, pred_dim)  # pred_dim = number of discrete time bins
        self.logits_dim = pred_dim

    def forward(self, all_modalities):
        input_data = {}
        for mm in self.m_projector.keys():
            if mm not in all_modalities:
                input_data[mm] = None
                continue

            x = all_modalities[mm]
            if x.dim() > 2:
                x = x.mean(dim=1)
            mask_key = f"{mm}_mask"
            if mask_key in all_modalities and all_modalities[mask_key].sum() == 0:
                input_data[mm] = None
            else:
                input_data[mm] = self.m_projector[mm](x)

        input_list = [input_data.get(mm, None) for mm in self.m_projector.keys()]
        fused = self.fusion(input_list).unsqueeze(1)  # [B, 1, hidden_size]

        backbone_out = self.backbone(fused)           # [B, 1, hidden_size]
        pooled = backbone_out.mean(dim=1)             # [B, hidden_size]

        hazard_logits = self.head(pooled)             # [B, pred_dim]
        hazards = torch.sigmoid(hazard_logits)        # probability of event at each bin
        survival = torch.cumprod(1 - hazards, dim=1)  # [B, pred_dim]

        return hazards, survival



class DEMainModalityMILMoE(nn.Module):
    def __init__(self, device, modalities, hidden_size=256, dropout_rate=0.1, pred_dim=20, mlp_ratio=4, 
                 n_token=1024, n_backbone=6, n_head=4, top_k=128):
        super().__init__()
        self.device = device
        self.modalities = list(modalities.keys())
        self.hidden_size = hidden_size
        self.n_token = n_token
        self.top_k = top_k

        self.m_projector = nn.ModuleDict({
            mm: nn.Sequential(
                nn.Linear(modalities[mm].feature_dim, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ) for mm in modalities
        })

        self.missing_text_token = nn.Parameter(torch.randn(1, 1, hidden_size))

        self.cross_text = DenseCrossAttention(hidden_size, n_head)
        self.cross_rna = DenseCrossAttention(hidden_size, n_head)

        self.token_scorer = nn.Linear(hidden_size, 1)

        self.backbone = nn.Sequential(
            *[DeformableTransformerEncoderLayer(
                d_model=hidden_size,
                nhead=n_head,
                dim_feedforward=hidden_size * mlp_ratio,
                dropout=dropout_rate,
                num_points=8
            ) for _ in range(n_backbone)]
        )

        self.head = nn.Linear(hidden_size, pred_dim)  # ✅ 输出多时间段 hazard logits
        self.logits_dim = pred_dim

    def forward(self, all_modalities):
        img_feat = all_modalities['img']
        img_feat = self.m_projector['img'](img_feat)

        scores = self.token_scorer(img_feat).squeeze(-1)
        topk_scores, topk_indices = torch.topk(scores, self.top_k, dim=1)
        batch_indices = torch.arange(img_feat.size(0), device=img_feat.device).unsqueeze(-1).expand(-1, self.top_k)
        img_feat = img_feat[batch_indices, topk_indices]

        text_feat = all_modalities.get('text', None)
        if text_feat is not None:
            if text_feat.dim() > 2:
                text_feat = text_feat.mean(dim=1)
            text_feat = self.m_projector['text'](text_feat).unsqueeze(1)
        else:
            B = img_feat.size(0)
            text_feat = self.missing_text_token.expand(B, 1, -1)

        rna_feat = all_modalities.get('rna', None)
        if rna_feat is not None:
            if rna_feat.dim() > 2:
                rna_feat = rna_feat.mean(dim=1)
            rna_feat = self.m_projector['rna'](rna_feat).unsqueeze(1)

        img_feat = self.cross_text(img_feat, text_feat, text_feat)
        if rna_feat is not None:
            img_feat = self.cross_rna(img_feat, rna_feat, rna_feat)

        backbone_out = self.backbone(img_feat)
        pooled = backbone_out.mean(dim=1)

        hazard_logits = self.head(pooled)         # [B, pred_dim]
        hazards = torch.sigmoid(hazard_logits)    # [B, pred_dim]
        survival = torch.cumprod(1 - hazards, dim=1)  # [B, pred_dim]

        return hazards, survival




class CrossModalAttentionBlock(nn.Module):
    def __init__(self, dim, n_head):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_head, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: Tensor, context: Tensor) -> Tensor:
        attn_out, _ = self.attn(x, context, context)
        x = self.norm(x + attn_out)
        x = self.norm2(x + self.mlp(x))
        return x

class CrossAttnFusionWithLearnableMissing(nn.Module):
    def __init__(self, device, modalities: Dict[str, Modality], hidden_size,
                 dropout_rate=0.1, pred_dim=20, mlp_ratio=4,
                 n_token=16, n_backbone=3, n_head=4, mask_prob=0.1):
        super().__init__()
        self.device = device
        self.modalities = modalities
        self.hidden_size = hidden_size
        self.n_token = n_token
        self.m_names = list(modalities.keys())
        self.mask_prob = mask_prob  # ✨ 新增

        self.m_projector = nn.ModuleDict({
            mm: QFormerEncoder(
                feature_size=modalities[mm].feature_dim,
                num_patches=n_token,
                embed_dim=hidden_size,
                num_heads=n_head,
                mlp_ratio=mlp_ratio,
                attn_drop=dropout_rate,
                drop=dropout_rate
            ) for mm in modalities
        })

        self.learnable_missing_token = nn.ParameterDict({
            mm: nn.Parameter(torch.randn(n_token, hidden_size))
            for mm in modalities
        })

        self.cross_attn = nn.ModuleList([
            CrossTransformerBlock(
                dim=hidden_size,
                num_heads=n_head,
                mlp_ratio=mlp_ratio,
                qkv_bias=False,
                drop=dropout_rate,
                attn_drop=dropout_rate
            ) for _ in range(len(self.m_names))
        ])

        self.head = nn.Linear(hidden_size * len(self.m_names), pred_dim)
        self.logits_dim = pred_dim

    def forward(self, all_modalities: Dict[str, torch.Tensor], training=True):
        tf_hidden = {}
        B = next(iter(all_modalities.values())).shape[0]  # batch size

        for mm in self.m_names:
            valid_mask = all_modalities.get(f"{mm}_valid", None)

            # ✨ 随机 mask 一部分模态（仅在训练中启用）
            if training and valid_mask is not None:
                if torch.rand(1).item() < self.mask_prob:
                    valid_mask = torch.zeros_like(valid_mask).bool()

            if valid_mask is not None and valid_mask.all():
                tf_hidden[mm] = self.m_projector[mm](all_modalities[mm])
            else:
                token = self.learnable_missing_token[mm].unsqueeze(0).repeat(B, 1, 1)
                tf_hidden[mm] = token.to(self.device)

        for i, mm_i in enumerate(self.m_names):
            x = tf_hidden[mm_i]
            other_tokens = [tf_hidden[mm_j] for j, mm_j in enumerate(self.m_names) if mm_j != mm_i]
            if len(other_tokens) == 0:
                continue
            ox = torch.cat(other_tokens, dim=1)
            tf_hidden[mm_i] = self.cross_attn[i](x, ox)

        head_input = [tokens.mean(dim=1) for tokens in tf_hidden.values()]
        head_input = torch.cat(head_input, dim=-1)
        logits = self.head(head_input)

        return logits


class CrossAttnFusionDropMissing(nn.Module):
    def __init__(self, device, modalities: Dict[str, Modality], hidden_size,
                 dropout_rate=0.1, pred_dim=20, mlp_ratio=4,
                 n_token=16, n_backbone=3, n_head=4):
        super().__init__()
        self.device = device
        self.modalities = modalities
        self.hidden_size = hidden_size
        self.n_token = n_token
        self.m_names = list(modalities.keys())

        self.m_projector = nn.ModuleDict({
            mm: QFormerEncoder(
                feature_size=modalities[mm].feature_dim,
                num_patches=n_token,
                embed_dim=hidden_size,
                num_heads=n_head,
                mlp_ratio=mlp_ratio,
                attn_drop=dropout_rate,
                drop=dropout_rate
            ) for mm in modalities
        })

        self.cross_attn = nn.ModuleDict({
            mm: CrossTransformerBlock(
                dim=hidden_size,
                num_heads=n_head,
                mlp_ratio=mlp_ratio,
                qkv_bias=False,
                drop=dropout_rate,
                attn_drop=dropout_rate
            ) for mm in self.m_names
        })

        # ⛔ 不预设 head 输入维度，forward 时动态根据有效模态数量拼接
        self.pred_dim = pred_dim
        self.head = nn.Linear(hidden_size * len(self.m_names), pred_dim)  # placeholder, not used directly
        self.logits_dim = pred_dim

    def forward(self, all_modalities: Dict[str, torch.Tensor]):
        B = next(iter(all_modalities.values())).shape[0]
        tf_hidden = {}

        # Step 1: Project only valid modalities
        for mm in self.m_names:
            valid_mask = all_modalities.get(f"{mm}_valid", None)
            if valid_mask is not None and valid_mask.all():
                tf_hidden[mm] = self.m_projector[mm](all_modalities[mm])

        if len(tf_hidden) == 0:
            raise ValueError("All modalities are missing. Cannot proceed.")

        # Step 2: Cross-attention among valid modalities
        updated = {}
        for mm_i in tf_hidden:
            x = tf_hidden[mm_i]
            other_tokens = [tf_hidden[mm_j] for mm_j in tf_hidden if mm_j != mm_i]
            if other_tokens:
                ox = torch.cat(other_tokens, dim=1)
                updated[mm_i] = self.cross_attn[mm_i](x, ox)
            else:
                updated[mm_i] = x
        tf_hidden = updated

        # Step 3: Pool and fuse valid modalities
        head_input = [v.mean(dim=1) for v in tf_hidden.values()]
        head_input = torch.cat(head_input, dim=-1)

        # ⛳ Head prediction using dynamic input dim
        fusion_head = nn.Linear(head_input.shape[1], self.pred_dim).to(self.device)
        logits = fusion_head(head_input)

        return logits




