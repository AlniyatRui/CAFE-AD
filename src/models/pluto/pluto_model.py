from copy import deepcopy
import math
import random
import json
import os 

import torch
import torch.nn as nn
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import (
    EgoTrajectoryTargetBuilder,
)

from src.feature_builders.pluto_feature_builder import PlutoFeatureBuilder

from .layers.fourier_embedding import FourierEmbedding
from .modules.agent_encoder import AgentEncoder
from .modules.agent_predictor import AgentPredictor
from .modules.map_encoder import MapEncoder
from .modules.static_objects_encoder import StaticObjectsEncoder
from .modules.planning_decoder import PlanningDecoder
from .layers.mlp_layer import MLPLayer
from .layers.prune_block import Block, batched_index_select


# no meaning, required by nuplan
trajectory_sampling = TrajectorySampling(num_poses=8, time_horizon=8, interval_length=1)

class ScenarioClassifier(nn.Module):
    def __init__(self, num_scenarios, dim):
        super(ScenarioClassifier, self).__init__()
        self.dim = dim
        self.featurizer = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(self.dim, num_scenarios)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.featurizer(x)
        x = x.squeeze(-1)
        x = self.classifier(x)
        return x 

class CAFE_Interpolation(nn.Module):
    def __init__(self, batch_size, num_scenarios):
        super(CAFE_Interpolation, self).__init__()
        self.batch_size = batch_size
        self.num_scenarios = num_scenarios

        self.threshold = 0.9
        self.threshold_upper_bound = 0.9
        self.threshold_lower_bound = 0.5
        self.threshold_change = 0.1
        self.step_to_change = 100
        
        self.power = 0.0

        self.uniform = torch.distributions.Uniform(self.power, 1)

        self.call_num = 0

        cur_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(cur_file_dir, "../../../"))
        above_avg_types_path = os.path.join(project_root, "preprocess", "above_avg_types.json")

        with open(above_avg_types_path, "r") as f:
            self.dominant_scenarios = json.load(f)
        
    def update_threshold(self):
        next_threshold = self.threshold - self.threshold_change
        if self.threshold == self.threshold_lower_bound:
            self.threshold = self.threshold_upper_bound
        elif next_threshold < self.threshold_lower_bound:
            self.threshold = self.threshold_lower_bound
        else:
            self.threshold = next_threshold

    def get_threshold(result, quantile):
        return torch.quantile(result, quantile)
        
    def sample_different_scenario(idx, scenario, scenario_target):
        y_size = scenario.size(0)
        for trial in range(y_size*4):
            current_idx = random.randrange(y_size)
            if scenario[current_idx] != scenario_target:
                return current_idx
        return idx 

    def get_feature_decomposition(self, scenario_im, feature):
        scenario_im =  torch.mean(scenario_im, dim=0, keepdim=True)
        scenario_thr = CAFE_Interpolation.get_threshold(scenario_im, self.threshold)

        sr_idx = scenario_im > scenario_thr
        sg_idx = scenario_im <= scenario_thr
        
        sr_mask = sr_idx
        sg_mask = sg_idx

        return feature*sr_mask, feature*sg_mask

    def forward(self, x, scenario, scenario_gradient):
        if (self.call_num % self.step_to_change == 0) and self.call_num != 0:
            self.update_threshold()

        B = x.size(0)
        result = torch.zeros(x.size()).to(x.device)

        sr = torch.zeros(x.size()).to(x.device)
        sg = torch.zeros(x.size()).to(x.device)

        self.average_change = 0
        for b in range(B):
            current_sgrad = scenario_gradient[b].to(x.device)
            current_feature = x[b, :, :]
            sr_f, sg_f = self.get_feature_decomposition(current_sgrad*current_feature, current_feature)
            
            sr[b] = sr_f
            sg[b] = sg_f
        
        mixup_strength = self.uniform.sample((B, 1))
        for b in range(B):
            if int(scenario[b].cpu().item()) in self.dominant_scenarios:
                scenario_label = scenario[b]
                diff_d = CAFE_Interpolation.sample_different_scenario(b, scenario, scenario_label)
                new_sr = mixup_strength[b][0] * sr[b] + (1-mixup_strength[b][0]) * sr[diff_d]
                result[b] = sg[b] + new_sr
            else:
                result[b] = sg[b] + sr[b]
                
        self.call_num += 1
  
        return result
    
class PlanningModel(TorchModuleWrapper):
    def __init__(
        self,
        dim=128,
        state_channel=6,
        polygon_channel=6,
        history_channel=9,
        history_steps=21,
        future_steps=80,
        encoder_depth=4,
        decoder_depth=4,
        drop_path=0.2,
        dropout=0.1,
        num_heads=8,
        num_modes=6,
        use_ego_history=False,
        state_attn_encoder=True,
        state_dropout=0.75,
        use_hidden_proj=False,
        cat_x=False,
        ref_free_traj=False,
        feature_builder: PlutoFeatureBuilder = PlutoFeatureBuilder(),
    ) -> None:
        super().__init__(
            feature_builders=[feature_builder],
            target_builders=[EgoTrajectoryTargetBuilder(trajectory_sampling)],
            future_trajectory_sampling=trajectory_sampling,
        )

        self.dim = dim
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.use_hidden_proj = use_hidden_proj
        self.num_modes = num_modes
        self.radius = feature_builder.radius
        self.ref_free_traj = ref_free_traj

        cur_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(cur_file_dir, "../../../"))
        type2id_path = os.path.join(project_root, "preprocess", "type2id.json")

        with open(type2id_path, "r") as f:
            scenario_label_dict = json.load(f)

        self.num_scenarios = len(scenario_label_dict.keys())
        self.scenario_classifier = ScenarioClassifier(num_scenarios=self.num_scenarios, dim=self.dim)
        self.mixup_module = CAFE_Interpolation(batch_size=32, num_scenarios=self.num_scenarios)

        self.pos_emb = FourierEmbedding(3, dim, 64)

        self.agent_encoder = AgentEncoder(
            state_channel=state_channel,
            history_channel=history_channel,
            dim=dim,
            hist_steps=history_steps,
            drop_path=drop_path,
            use_ego_history=use_ego_history,
            state_attn_encoder=state_attn_encoder,
            state_dropout=state_dropout,
        )

        self.map_encoder = MapEncoder(
            dim=dim,
            polygon_channel=polygon_channel,
            use_lane_boundary=True,
        )

        self.static_objects_encoder = StaticObjectsEncoder(dim=dim)
        
        drop_rate = 0.
        attn_drop_rate = 0.
        mlp_ratio = 4
        qkv_bias = False
        dpr = [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]
        
        self.blocks = nn.ModuleList([
            Block(
                dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], act_layer=nn.GELU, norm_layer=nn.LayerNorm)
            for i in range(encoder_depth)])
        self.pruning_loc = [1,3]
        
        self.norm = nn.LayerNorm(dim)

        self.agent_predictor = AgentPredictor(dim=dim, future_steps=future_steps)
        self.planning_decoder = PlanningDecoder(
            num_mode=num_modes,
            decoder_depth=decoder_depth,
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=4,
            dropout=dropout,
            cat_x=cat_x,
            future_steps=future_steps,
        )

        if use_hidden_proj:
            self.hidden_proj = nn.Sequential(
                nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)
            )

        if self.ref_free_traj:
            self.ref_free_decoder = MLPLayer(dim, 2 * dim, future_steps * 4)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def get_feature(self, data, warm_up=False):
        agent_pos = data["agent"]["position"][:, :, self.history_steps - 1]
        agent_heading = data["agent"]["heading"][:, :, self.history_steps - 1]
        agent_mask = data["agent"]["valid_mask"][:, :, : self.history_steps]
        agent_valid_mask = data["agent"]["valid_mask"][:, 1:]
        ego_valid_mask = data["agent"]["valid_mask"][:, 0:1]
        polygon_center = data["map"]["polygon_center"]
        polygon_mask = data["map"]["valid_mask"]
        
        current_state = torch.cat(
            [data["agent"]["position"][:, 1:, -1], data["agent"]["heading"][:, 1:, -1, None]],
            dim=-1,
        )

        ego_shape = data["agent"]["shape"][:, 0:1]
        agent_shape = data["agent"]["shape"][:, 1:]
        ego_category = data["agent"]["category"][:, 0:1]
        agent_category = data["agent"]["category"][:, 1:]

        all_token = torch.arange(0, agent_category.shape[1] + 1).unsqueeze(0).unsqueeze(-1).expand(agent_category.shape[0], -1, -1)
        all_token = all_token.to(agent_pos.device)
        ego_token = all_token[:, 0:1]
        agent_token = all_token[:, 1:]

        bs, A = agent_pos.shape[0:2]

        position = torch.cat([agent_pos, polygon_center[..., :2]], dim=1)
        angle = torch.cat([agent_heading, polygon_center[..., 2]], dim=1)
        angle = (angle + math.pi) % (2 * math.pi) - math.pi
        pos = torch.cat([position, angle.unsqueeze(-1)], dim=-1)

        agent_key_padding = ~(agent_mask.any(-1))
        polygon_key_padding = ~(polygon_mask.any(-1))
        key_padding_mask = torch.cat([agent_key_padding, polygon_key_padding], dim=-1)

        x_agent = self.agent_encoder(data)
        x_polygon = self.map_encoder(data)
        x_static, static_pos, static_key_padding = self.static_objects_encoder(data)

        x = torch.cat([x_agent, x_polygon, x_static], dim=1)

        pos = torch.cat([pos, static_pos], dim=1)
        pos_embed = self.pos_emb(pos)

        key_padding_mask = torch.cat([key_padding_mask, static_key_padding], dim=-1)
        x = x + pos_embed

        input = x
        mask = key_padding_mask
        # fusion encoder with token attention prune
        neighbors_future = data["agent"]["target"][:, 1:]
        ego_future = data["agent"]["target"][:, 0:1]
        
        ego_vel = data["agent"]["velocity"][:, 0:1]
        neighbors_vel = data["agent"]["velocity"][:, 1:]
        
        
        actors_mask = agent_key_padding[:, 1:]
        map_mask = polygon_key_padding.clone()
        static_mask = static_key_padding.clone()
        
        B = input.shape[0]
        N1 = input.shape[1]
        num_token_all = input.shape[1]  # N + 1
        num_agents = neighbors_future.shape[1]
        num_static = static_mask.shape[1]
        num_map = map_mask.shape[1]
        token_idx_agent = torch.arange(0, num_agents).long().unsqueeze(0).expand(B, -1).to(input.device)
        token_idx_map = torch.arange(0, num_map).long().unsqueeze(0).expand(B, -1).to(input.device)  # (B, N)      initial
        token_idx_static = torch.arange(0, num_static).long().unsqueeze(0).expand(B, -1).to(input.device)
        
        attn_mask = torch.ones(B, N1, N1).to(input.device)  # (B, N+1, N+1) 1
        
        if warm_up == False:
            keep_ratio = 0.9
        else:
            keep_ratio = 1.0

        ratio = attn_ratio = keep_ratio
        
        for i, blk in enumerate(self.blocks):
            if i in self.pruning_loc:
                input, token_idx_agent, token_idx_map, token_idx_static, attn_mask = blk(input, token_keep_ratio=ratio,
                                                                       token_idx_agent=token_idx_agent,
                                                                       token_idx_map=token_idx_map, 
                                                                       token_idx_static=token_idx_static,
                                                                       token_prune= True,
                                                                       attn_prune=True, attn_mask=attn_mask,
                                                                       key_padding_mask=mask,
                                                                       attn_keep_ratio=attn_ratio)
                ego_mask = agent_key_padding[:, 0:1]
                # import pdb; pdb.set_trace()
                map_mask = batched_index_select(map_mask, 1, token_idx_map)
                actors_mask = batched_index_select(actors_mask, 1, token_idx_agent)
                static_mask = batched_index_select(static_mask, 1, token_idx_static)
                # actors_mask = torch.eq(actors[:, :, -1].sum(-1), 0)
                mask = torch.cat([ego_mask, actors_mask, map_mask, static_mask], dim=-1)
                neighbors_future = batched_index_select(neighbors_future, 1, token_idx_agent)
                agent_valid_mask = batched_index_select(agent_valid_mask, 1, token_idx_agent)
                neighbors_vel = batched_index_select(neighbors_vel, 1, token_idx_agent)
                current_state = batched_index_select(current_state, 1, token_idx_agent)
                agent_shape = batched_index_select(agent_shape, 1, token_idx_agent)
                agent_category = batched_index_select(agent_category, 1, token_idx_agent)
                agent_token= batched_index_select(agent_token, 1, token_idx_agent)
            else:
                input, token_idx_agent, token_idx_map, token_idx_static, attn_mask = blk(input, token_keep_ratio=ratio,
                                                                       token_idx_agent=token_idx_agent,
                                                                       token_idx_map=token_idx_map, 
                                                                       token_idx_static=token_idx_static,
                                                                       token_prune=False,
                                                                       attn_prune=False, attn_mask=attn_mask,
                                                                       key_padding_mask=mask,
                                                                       attn_keep_ratio=attn_ratio)
        
        
        A = neighbors_future.shape[1]
        x = self.norm(input)
        
        return x
    
    def train_f(self, data, scenario, scenario_gradient, warm_up=False):
        agent_pos = data["agent"]["position"][:, :, self.history_steps - 1]
        agent_heading = data["agent"]["heading"][:, :, self.history_steps - 1]
        agent_mask = data["agent"]["valid_mask"][:, :, : self.history_steps]
        agent_valid_mask = data["agent"]["valid_mask"][:, 1:]
        ego_valid_mask = data["agent"]["valid_mask"][:, 0:1]
        polygon_center = data["map"]["polygon_center"]
        polygon_mask = data["map"]["valid_mask"]
        
        current_state = torch.cat(
            [data["agent"]["position"][:, 1:, -1], data["agent"]["heading"][:, 1:, -1, None]],
            dim=-1,
        )

        ego_shape = data["agent"]["shape"][:, 0:1]
        agent_shape = data["agent"]["shape"][:, 1:]
        ego_category = data["agent"]["category"][:, 0:1]
        agent_category = data["agent"]["category"][:, 1:]

        all_token = torch.arange(0, agent_category.shape[1] + 1).unsqueeze(0).unsqueeze(-1).expand(agent_category.shape[0], -1, -1)
        all_token = all_token.to(agent_pos.device)
        ego_token = all_token[:, 0:1]
        agent_token = all_token[:, 1:]

        bs, A = agent_pos.shape[0:2]

        position = torch.cat([agent_pos, polygon_center[..., :2]], dim=1)
        angle = torch.cat([agent_heading, polygon_center[..., 2]], dim=1)
        angle = (angle + math.pi) % (2 * math.pi) - math.pi
        pos = torch.cat([position, angle.unsqueeze(-1)], dim=-1)

        agent_key_padding = ~(agent_mask.any(-1))
        polygon_key_padding = ~(polygon_mask.any(-1))
        key_padding_mask = torch.cat([agent_key_padding, polygon_key_padding], dim=-1)

        x_agent = self.agent_encoder(data)
        x_polygon = self.map_encoder(data)
        x_static, static_pos, static_key_padding = self.static_objects_encoder(data)

        x = torch.cat([x_agent, x_polygon, x_static], dim=1)

        pos = torch.cat([pos, static_pos], dim=1)
        pos_embed = self.pos_emb(pos)

        key_padding_mask = torch.cat([key_padding_mask, static_key_padding], dim=-1)
        x = x + pos_embed

        input = x
        mask = key_padding_mask
        # fusion encoder with token attention prune
        neighbors_future = data["agent"]["target"][:, 1:]
        ego_future = data["agent"]["target"][:, 0:1]
        
        ego_vel = data["agent"]["velocity"][:, 0:1]
        neighbors_vel = data["agent"]["velocity"][:, 1:]
        
        
        actors_mask = agent_key_padding[:, 1:]
        map_mask = polygon_key_padding.clone()
        static_mask = static_key_padding.clone()
        
        B = input.shape[0]
        N1 = input.shape[1]
        num_token_all = input.shape[1]  # N + 1
        num_agents = neighbors_future.shape[1]
        num_static = static_mask.shape[1]
        num_map = map_mask.shape[1]
        token_idx_agent = torch.arange(0, num_agents).long().unsqueeze(0).expand(B, -1).to(input.device)
        token_idx_map = torch.arange(0, num_map).long().unsqueeze(0).expand(B, -1).to(input.device)  # (B, N)      initial
        token_idx_static = torch.arange(0, num_static).long().unsqueeze(0).expand(B, -1).to(input.device)
        
        attn_mask = torch.ones(B, N1, N1).to(input.device)  # (B, N+1, N+1) 1
        
        if warm_up == False:
            keep_ratio = 0.9
        else:
            keep_ratio = 1.0

        ratio = attn_ratio = keep_ratio
        
        for i, blk in enumerate(self.blocks):
            if i in self.pruning_loc:
                input, token_idx_agent, token_idx_map, token_idx_static, attn_mask = blk(input, token_keep_ratio=ratio,
                                                                       token_idx_agent=token_idx_agent,
                                                                       token_idx_map=token_idx_map, 
                                                                       token_idx_static=token_idx_static,
                                                                       token_prune= True,
                                                                       attn_prune=True, attn_mask=attn_mask,
                                                                       key_padding_mask=mask,
                                                                       attn_keep_ratio=attn_ratio)
                ego_mask = agent_key_padding[:, 0:1]
                # import pdb; pdb.set_trace()
                map_mask = batched_index_select(map_mask, 1, token_idx_map)
                actors_mask = batched_index_select(actors_mask, 1, token_idx_agent)
                static_mask = batched_index_select(static_mask, 1, token_idx_static)
                # actors_mask = torch.eq(actors[:, :, -1].sum(-1), 0)
                mask = torch.cat([ego_mask, actors_mask, map_mask, static_mask], dim=-1)
                neighbors_future = batched_index_select(neighbors_future, 1, token_idx_agent)
                agent_valid_mask = batched_index_select(agent_valid_mask, 1, token_idx_agent)
                neighbors_vel = batched_index_select(neighbors_vel, 1, token_idx_agent)
                current_state = batched_index_select(current_state, 1, token_idx_agent)
                agent_shape = batched_index_select(agent_shape, 1, token_idx_agent)
                agent_category = batched_index_select(agent_category, 1, token_idx_agent)
                agent_token= batched_index_select(agent_token, 1, token_idx_agent)
            else:
                input, token_idx_agent, token_idx_map, token_idx_static, attn_mask = blk(input, token_keep_ratio=ratio,
                                                                       token_idx_agent=token_idx_agent,
                                                                       token_idx_map=token_idx_map, 
                                                                       token_idx_static=token_idx_static,
                                                                       token_prune=False,
                                                                       attn_prune=False, attn_mask=attn_mask,
                                                                       key_padding_mask=mask,
                                                                       attn_keep_ratio=attn_ratio)
        
        neighbors_future = torch.cat([ego_future, neighbors_future], dim=1)
        agent_valid_mask = torch.cat([ego_valid_mask, agent_valid_mask], dim=1)
        neighbors_vel = torch.cat([ego_vel, neighbors_vel], dim=1)
        agent_shape = torch.cat([ego_shape, agent_shape], dim=1)
        agent_category = torch.cat([ego_category, agent_category], dim=1)
        agent_token = torch.cat([ego_token, agent_token], dim=1)
        
        A = neighbors_future.shape[1]
        x = self.norm(input)

        x = self.mixup_module.forward(x, scenario, scenario_gradient)

        prediction = self.agent_predictor(x[:, 1:A])

        ref_line_available = data["reference_line"]["position"].shape[1] > 0

        if ref_line_available:
            trajectory, probability = self.planning_decoder(
                data, {"enc_emb": x, "enc_key_padding_mask": mask}
            )
        else:
            trajectory, probability = None, None

        out = {
            "trajectory": trajectory,
            "probability": probability,  # (bs, R, M)
            "prediction": prediction,  # (bs, A-1, T, 2)
            "neighbors_future": neighbors_future,
            "agent_valid_mask": agent_valid_mask,
            "neighbors_vel": neighbors_vel,
            "current_state": current_state,
            "agent_shape": agent_shape,
            "agent_category": agent_category,
            "agent_token": agent_token,
        }

        if self.use_hidden_proj:
            out["hidden"] = self.hidden_proj(x[:, 0])

        if self.ref_free_traj:
            ref_free_traj = self.ref_free_decoder(x[:, 0]).reshape(
                bs, self.future_steps, 4
            )
            out["ref_free_trajectory"] = ref_free_traj

        if not self.training:
            if self.ref_free_traj:
                ref_free_traj_angle = torch.arctan2(
                    ref_free_traj[..., 3], ref_free_traj[..., 2]
                )
                ref_free_traj = torch.cat(
                    [ref_free_traj[..., :2], ref_free_traj_angle.unsqueeze(-1)], dim=-1
                )
                out["output_ref_free_trajectory"] = ref_free_traj

            output_prediction = torch.cat(
                [
                    prediction[..., :2] + agent_pos[:, 1:A, None],
                    torch.atan2(prediction[..., 3], prediction[..., 2]).unsqueeze(-1)
                    + agent_heading[:, 1:A, None, None],
                    prediction[..., 4:6],
                ],
                dim=-1,
            )
            out["output_prediction"] = output_prediction

            if trajectory is not None:
                r_padding_mask = ~data["reference_line"]["valid_mask"].any(-1)
                probability.masked_fill_(r_padding_mask.unsqueeze(-1), -1e6)

                angle = torch.atan2(trajectory[..., 3], trajectory[..., 2])
                out_trajectory = torch.cat(
                    [trajectory[..., :2], angle.unsqueeze(-1)], dim=-1
                )

                bs, R, M, T, _ = out_trajectory.shape
                flattened_probability = probability.reshape(bs, R * M)
                best_trajectory = out_trajectory.reshape(bs, R * M, T, -1)[
                    torch.arange(bs), flattened_probability.argmax(-1)
                ]

                out["output_trajectory"] = best_trajectory
                out["candidate_trajectories"] = out_trajectory
            else:
                out["output_trajectory"] = out["output_ref_free_trajectory"]
                out["probability"] = torch.zeros(1, 0, 0)
                out["candidate_trajectories"] = torch.zeros(
                    1, 0, 0, self.future_steps, 3
                )

        return out
    
    def forward(self, data, warm_up=False):
        agent_pos = data["agent"]["position"][:, :, self.history_steps - 1]
        agent_heading = data["agent"]["heading"][:, :, self.history_steps - 1]
        agent_mask = data["agent"]["valid_mask"][:, :, : self.history_steps]
        agent_valid_mask = data["agent"]["valid_mask"][:, 1:]
        ego_valid_mask = data["agent"]["valid_mask"][:, 0:1]
        polygon_center = data["map"]["polygon_center"]
        polygon_mask = data["map"]["valid_mask"]
        
        current_state = torch.cat(
            [data["agent"]["position"][:, 1:, -1], data["agent"]["heading"][:, 1:, -1, None]],
            dim=-1,
        )

        ego_shape = data["agent"]["shape"][:, 0:1]
        agent_shape = data["agent"]["shape"][:, 1:]
        ego_category = data["agent"]["category"][:, 0:1]
        agent_category = data["agent"]["category"][:, 1:]

        all_token = torch.arange(0, agent_category.shape[1] + 1).unsqueeze(0).unsqueeze(-1).expand(agent_category.shape[0], -1, -1)
        all_token = all_token.to(agent_pos.device)
        ego_token = all_token[:, 0:1]
        agent_token = all_token[:, 1:]

        bs, A = agent_pos.shape[0:2]

        position = torch.cat([agent_pos, polygon_center[..., :2]], dim=1)
        angle = torch.cat([agent_heading, polygon_center[..., 2]], dim=1)
        angle = (angle + math.pi) % (2 * math.pi) - math.pi
        pos = torch.cat([position, angle.unsqueeze(-1)], dim=-1)

        agent_key_padding = ~(agent_mask.any(-1))
        polygon_key_padding = ~(polygon_mask.any(-1))
        key_padding_mask = torch.cat([agent_key_padding, polygon_key_padding], dim=-1)

        x_agent = self.agent_encoder(data)
        x_polygon = self.map_encoder(data)
        x_static, static_pos, static_key_padding = self.static_objects_encoder(data)

        x = torch.cat([x_agent, x_polygon, x_static], dim=1)

        pos = torch.cat([pos, static_pos], dim=1)
        pos_embed = self.pos_emb(pos)

        key_padding_mask = torch.cat([key_padding_mask, static_key_padding], dim=-1)
        x = x + pos_embed

        input = x
        mask = key_padding_mask
        # fusion encoder with token attention prune
        neighbors_future = data["agent"]["target"][:, 1:]
        ego_future = data["agent"]["target"][:, 0:1]
        
        ego_vel = data["agent"]["velocity"][:, 0:1]
        neighbors_vel = data["agent"]["velocity"][:, 1:]
        
        
        actors_mask = agent_key_padding[:, 1:]
        map_mask = polygon_key_padding.clone()
        static_mask = static_key_padding.clone()
        
        B = input.shape[0]
        N1 = input.shape[1]
        num_token_all = input.shape[1]  # N + 1
        num_agents = neighbors_future.shape[1]
        num_static = static_mask.shape[1]
        num_map = map_mask.shape[1]
        token_idx_agent = torch.arange(0, num_agents).long().unsqueeze(0).expand(B, -1).to(input.device)
        token_idx_map = torch.arange(0, num_map).long().unsqueeze(0).expand(B, -1).to(input.device)  # (B, N)      initial
        token_idx_static = torch.arange(0, num_static).long().unsqueeze(0).expand(B, -1).to(input.device)
        
        attn_mask = torch.ones(B, N1, N1).to(input.device)  # (B, N+1, N+1) 1
        
        if warm_up == False:
            keep_ratio = 0.9
        else:
            keep_ratio = 1.0

        ratio = attn_ratio = keep_ratio
        
        for i, blk in enumerate(self.blocks):
            input, token_idx_agent, token_idx_map, token_idx_static, attn_mask = blk(input, token_keep_ratio=ratio,
                                                                   token_idx_agent=token_idx_agent,
                                                                   token_idx_map=token_idx_map, 
                                                                   token_idx_static=token_idx_static,
                                                                   token_prune=False,
                                                                   attn_prune=False, attn_mask=attn_mask,
                                                                   key_padding_mask=mask,
                                                                   attn_keep_ratio=attn_ratio)



        neighbors_future = torch.cat([ego_future, neighbors_future], dim=1)
        agent_valid_mask = torch.cat([ego_valid_mask, agent_valid_mask], dim=1)
        neighbors_vel = torch.cat([ego_vel, neighbors_vel], dim=1)
        agent_shape = torch.cat([ego_shape, agent_shape], dim=1)
        agent_category = torch.cat([ego_category, agent_category], dim=1)
        agent_token = torch.cat([ego_token, agent_token], dim=1)
        
        A = neighbors_future.shape[1]
        x = self.norm(input)

        prediction = self.agent_predictor(x[:, 1:A])

        ref_line_available = data["reference_line"]["position"].shape[1] > 0

        if ref_line_available:
            trajectory, probability = self.planning_decoder(
                data, {"enc_emb": x, "enc_key_padding_mask": mask}
            )
        else:
            trajectory, probability = None, None

        out = {
            "trajectory": trajectory,
            "probability": probability,  # (bs, R, M)
            "prediction": prediction,  # (bs, A-1, T, 2)
            "neighbors_future": neighbors_future,
            "agent_valid_mask": agent_valid_mask,
            "neighbors_vel": neighbors_vel,
            "current_state": current_state,
            "agent_shape": agent_shape,
            "agent_category": agent_category,
            "agent_token": agent_token,
        }

        if self.use_hidden_proj:
            out["hidden"] = self.hidden_proj(x[:, 0])

        if self.ref_free_traj:
            ref_free_traj = self.ref_free_decoder(x[:, 0]).reshape(
                bs, self.future_steps, 4
            )
            out["ref_free_trajectory"] = ref_free_traj

        if not self.training:
            if self.ref_free_traj:
                ref_free_traj_angle = torch.arctan2(
                    ref_free_traj[..., 3], ref_free_traj[..., 2]
                )
                ref_free_traj = torch.cat(
                    [ref_free_traj[..., :2], ref_free_traj_angle.unsqueeze(-1)], dim=-1
                )
                out["output_ref_free_trajectory"] = ref_free_traj

            output_prediction = torch.cat(
                [
                    prediction[..., :2] + agent_pos[:, 1:A, None],
                    torch.atan2(prediction[..., 3], prediction[..., 2]).unsqueeze(-1)
                    + agent_heading[:, 1:A, None, None],
                    prediction[..., 4:6],
                ],
                dim=-1,
            )
            out["output_prediction"] = output_prediction

            if trajectory is not None:
                r_padding_mask = ~data["reference_line"]["valid_mask"].any(-1)
                probability.masked_fill_(r_padding_mask.unsqueeze(-1), -1e6)

                angle = torch.atan2(trajectory[..., 3], trajectory[..., 2])
                out_trajectory = torch.cat(
                    [trajectory[..., :2], angle.unsqueeze(-1)], dim=-1
                )

                bs, R, M, T, _ = out_trajectory.shape
                flattened_probability = probability.reshape(bs, R * M)
                best_trajectory = out_trajectory.reshape(bs, R * M, T, -1)[
                    torch.arange(bs), flattened_probability.argmax(-1)
                ]

                out["output_trajectory"] = best_trajectory
                out["candidate_trajectories"] = out_trajectory
            else:
                out["output_trajectory"] = out["output_ref_free_trajectory"]
                out["probability"] = torch.zeros(1, 0, 0)
                out["candidate_trajectories"] = torch.zeros(
                    1, 0, 0, self.future_steps, 3
                )
      
        return out
