import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

def batched_index_select(input, dim, index):
    # input:(B, C, HW). index(B, N)
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    # import pdb; pdb.set_trace()
    return torch.gather(input, dim, index)      # (B,C, N)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., token_num=196):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # attn_mask: (B, N+1, N+1) input-dependent

        eps = 1e-6
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # (B, N+1, C) -> (B, N, 3C) -> (B, N+1, 3, H, C/H) -> (3, B, H, N+1, C/H)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)                 # (B, H, N+1, C/H)

        attn = (q @ k.transpose(-2, -1)) * self.scale       #  (B, H, N+1, C/H) @ (B, H, C/H, N+1) -> (B, H, N+1, N+1)

        # Apply key padding mask
        if key_padding_mask is not None:
            # Reshape to (batch_size, num_heads, sequence_length, sequence_length)
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            # Expand to the same size as the attention dimension
            key_padding_mask = key_padding_mask.expand(B, self.num_heads, N, N)
            # Set attention weights where key_padding_mask is true to negative infinity
            attn = attn.masked_fill(key_padding_mask, float('-inf'))

        # Key pruning (attention level) >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att

        attn = attn.to(torch.float32).exp_() * attn_mask.unsqueeze(1).to(torch.float32)     # (B, H, N+1, N+1)
        attn = (attn + eps/N) / (attn.sum(dim=-1, keepdim=True) + eps)          # (B, H, N+1, N+1)
        # attn = attn.softmax(dim=-1)                                           # (B, H, N+1, N+1)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)     # (B, H, N+1, N+1) * (B, H, N+1, C/H) -> (B, H, N+1, C/H) -> (B, N+1, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # save attention map
        cls_attn = attn[:, :, 0, 1:].sum(1) / self.num_heads                      # (B, H, N) -> (B, N)
        patch_attn = attn[:, :, 1:, 1:].sum(1) / self.num_heads                   # (B, H, N, N) -> (B, N, N)
        return x, cls_attn, patch_attn

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, token_keep_ratio=1.0, attn_keep_ratio=1.0, token_idx_agent=None, token_idx_map=None, token_idx_static=None, token_prune=False, attn_prune=False,
                attn_mask=None, key_padding_mask=None):

        # import pdb; pdb.set_trace()

        x_att, cls_attn, patch_attn = self.attn(self.norm1(x), attn_mask, key_padding_mask)
        # x: (B, N+1, C)
        # cls_attn: (B, N)      [cls] token, sum is 1
        # patch_attn: (B, N, N)     for each image patch
        x = x + self.drop_path(x_att)

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Token Prune >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        if token_prune:
            # print('keep_ratio',keep_ratio)
            x_cls_token = x[:, 0:1]  # (B, 1, C)
            B, N = cls_attn.shape  # N = 196
            agent_N = token_idx_agent.shape[1]
            map_N = token_idx_map.shape[1]
            static_N = token_idx_static.shape[1]
            
            cls_attn_agent = cls_attn[:,:agent_N]
            cls_attn_map = cls_attn[:, agent_N:agent_N+map_N]
            cls_attn_static = cls_attn[:, agent_N+map_N:]
            
            num_keep_node_agent = math.ceil(agent_N * token_keep_ratio)  # 196 r
            num_keep_node_map = math.ceil(map_N * token_keep_ratio)  # 196 r
            num_keep_node_static = math.ceil(static_N * token_keep_ratio)
            # num_keep_node_map = math.ceil(map_N)
            # num_keep_node_static = math.ceil(static_N)
            
            # attentive token
            token_idx_agent = cls_attn_agent.topk(num_keep_node_agent, dim=1)[1]
            token_idx_map = cls_attn_map.topk(num_keep_node_map, dim=1)[1] # (B, rN)        without gradient
            token_idx_static = cls_attn_static.topk(num_keep_node_static, dim=1)[1]
            
            x_attentive_agent = batched_index_select(x[:, 1:agent_N+1], 1, token_idx_agent)
            x_attentive_map = batched_index_select(x[:, agent_N+1:agent_N+1+map_N], 1, token_idx_map)# (B, N, C) -> (B, rN, C)
            x_attentive_static = batched_index_select(x[:, agent_N+1+map_N:], 1, token_idx_static)
            # x_attentive_map = x[:, agent_N + 1: agent_N + 1 + map_N]
            # x_attentive_static = x[:, agent_N + 1 + map_N:]

            x = torch.cat([x_cls_token, x_attentive_agent, x_attentive_map, x_attentive_static], dim=1)
            # x = torch.cat([x_cls_token, x_attentive], dim=1)  # (B, 1+rN, C)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Attention Prune >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        if attn_prune:
            # get attention map of pruned token
            patch_attn_agent = patch_attn[:,:agent_N,:agent_N]
            patch_attn_prune_agent = batched_index_select(patch_attn_agent, 1, token_idx_agent)  # (B, N, N) -> (B, rN, N)
            patch_attn_prune_agent = batched_index_select(patch_attn_prune_agent, 2, token_idx_agent)  # (B, rN, N) -> (B, rN, rN)
            
            #
            B, rN1, _ = x.shape
            rN = rN1 - 1
            num_keep_attn_agent = math.ceil(agent_N * attn_keep_ratio)  # rN * ra
            top_val, _ = patch_attn_prune_agent.topk(dim=2, k=num_keep_attn_agent)  # (B, rN, rN * ra)

            # import pdb; pdb.set_trace()

            attn_mask_p_agent = (patch_attn_prune_agent >= top_val[:, :, -1].unsqueeze(-1).expand(-1, -1,
                                                                                      num_keep_attn_agent)) + 0  # （B, rN, rN） without gradient     0/1 mask

            patch_attn_map= patch_attn[:, agent_N: agent_N+map_N, agent_N: agent_N+map_N]
            patch_attn_prune_map = batched_index_select(patch_attn_map, 1,
                                                          token_idx_map)  # (B, N, N) -> (B, rN, N)
            patch_attn_prune_map = batched_index_select(patch_attn_prune_map, 2,
                                                          token_idx_map)  # (B, rN, N) -> (B, rN, rN)
            # #
            B, rN1, _ = x.shape
            rN = rN1 - 1
            num_keep_attn_map = math.ceil(map_N * attn_keep_ratio)  # rN * ra
            # num_keep_attn_map = math.ceil(map_N)
            top_val, _ = patch_attn_prune_map.topk(dim=2, k=num_keep_attn_map)  # (B, rN, rN * ra)
            
            attn_mask_p_map = (patch_attn_prune_map >= top_val[:, :, -1].unsqueeze(-1).expand(-1, -1,
                                                                                        num_keep_attn_map)) + 0  # （B, rN, rN） without gradient     0/1 mask

            patch_attn_static= patch_attn[:, agent_N+map_N: , agent_N+map_N:]
            patch_attn_prune_static = batched_index_select(patch_attn_static, 1,
                                                          token_idx_static)  # (B, N, N) -> (B, rN, N)
            patch_attn_prune_static = batched_index_select(patch_attn_prune_static, 2,
                                                          token_idx_static)  # (B, rN, N) -> (B, rN, rN)
            # #
            B, rN1, _ = x.shape
            rN = rN1 - 1
            num_keep_attn_static = math.ceil(static_N * attn_keep_ratio)  # rN * ra
            # num_keep_attn_map = math.ceil(map_N)
            top_val, _ = patch_attn_prune_static.topk(dim=2, k=num_keep_attn_static)  # (B, rN, rN * ra)
            
            attn_mask_p_static = (patch_attn_prune_static >= top_val[:, :, -1].unsqueeze(-1).expand(-1, -1,
                                                                                        num_keep_attn_static)) + 0

            # attn_mask_p_map = attn_mask[:, num_keep_attn_agent + 1:, num_keep_attn_agent + 1:] # 285  # 44 48 44 

            # TODO: may add some random here

            attn_mask = torch.ones(B, rN1, rN1).to(x.device)  # (B, rN+1, rN+1) # 326 285+44+1 330 281+44+1
            attn_mask[:, 1:num_keep_attn_agent + 1, 1:num_keep_attn_agent + 1] = attn_mask_p_agent
            attn_mask[:, num_keep_attn_agent + 1: num_keep_attn_agent + 1 + num_keep_attn_map, num_keep_attn_agent + 1: num_keep_attn_agent + 1 + num_keep_attn_map] = attn_mask_p_map
            attn_mask[:, num_keep_attn_agent + 1 + num_keep_attn_map:, num_keep_attn_agent + 1 + num_keep_attn_map:] = attn_mask_p_static

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # x = x + self.drop_path(self.attn(self.norm1(x)))          # old form
        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, token_idx_agent, token_idx_map, token_idx_static, attn_mask
