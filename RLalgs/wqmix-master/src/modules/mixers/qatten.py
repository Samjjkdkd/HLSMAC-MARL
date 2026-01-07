# From https://github.com/wjh720/QPLEX/, added here for convenience.
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle as pkl
class QattenMixer(nn.Module):
    def __init__(self, args):
        super(QattenMixer, self).__init__()

        self.name = 'qatten'
        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.unit_dim = args.unit_dim
        self.n_actions = args.n_actions
        self.sa_dim = self.state_dim + self.n_agents * self.n_actions
        self.n_head = args.n_head  # attention head num

        self.embed_dim = args.mixing_embed_dim
        self.attend_reg_coef = args.attend_reg_coef

        # ResQ (Residual Q) parameters
        self.use_resq = getattr(args, 'use_resq', False)
        if self.use_resq:
            self.residual_embed_dim = getattr(args, 'residual_embed_dim', 32)
            self.residual_layers = getattr(args, 'residual_layers', 2)
            self.resq_init_scale = getattr(args, 'resq_init_scale', 0.1)
            
            # Build residual network Qr
            self.residual_network = self._build_residual_network()
            
            # Consistency loss coefficient
            self.consistency_loss_coef = getattr(args, 'consistency_loss_coef', 1.0)

        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()

        if getattr(args, "hypernet_layers", 1) == 1:
            for i in range(self.n_head):  # multi-head attention
                self.selector_extractors.append(nn.Linear(self.state_dim, self.embed_dim, bias=False))  # query
                if self.args.nonlinear:  # add qs
                    self.key_extractors.append(nn.Linear(self.unit_dim + 1, self.embed_dim, bias=False))  # key
                else:
                    self.key_extractors.append(nn.Linear(self.unit_dim, self.embed_dim, bias=False))  # key
            if self.args.weighted_head:
                self.hyper_w_head = nn.Linear(self.state_dim, self.n_head)
                # self.hyper_w_head = nn.Linear(self.state_dim, self.embed_dim * self.n_head)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            for i in range(self.n_head):  # multi-head attention
                selector_nn = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                            nn.ReLU(),
                                            nn.Linear(hypernet_embed, self.embed_dim, bias=False))
                self.selector_extractors.append(selector_nn)  # query
                if self.args.nonlinear:  # add qs
                    self.key_extractors.append(nn.Linear(self.unit_dim + 1, self.embed_dim, bias=False))  # key
                else:
                    self.key_extractors.append(nn.Linear(self.unit_dim, self.embed_dim, bias=False))  # key
            if self.args.weighted_head:
                self.hyper_w_head = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                                  nn.ReLU(),
                                                  nn.Linear(hypernet_embed, self.n_head))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 embednet layers is not implemented!")
        else:
            raise Exception("Error setting number of embednet layers.")

        if self.args.state_bias:
            # V(s) instead of a bias for the last layers
            self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states, actions):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        unit_states = states[:, : self.unit_dim * self.n_agents]  # get agent own features from state
        unit_states = unit_states.reshape(-1, self.n_agents, self.unit_dim)
        unit_states = unit_states.permute(1, 0, 2)

        agent_qs = agent_qs.view(-1, 1, self.n_agents)  # agent_qs: (batch_size, 1, agent_num)

        if self.args.nonlinear:
            unit_states = th.cat((unit_states, agent_qs.permute(2, 0, 1)), dim=2)
        # states: (batch_size, state_dim)
        all_head_selectors = [sel_ext(states) for sel_ext in self.selector_extractors]
        # all_head_selectors: (head_num, batch_size, embed_dim)
        # unit_states: (agent_num, batch_size, unit_dim)
        all_head_keys = [[k_ext(enc) for enc in unit_states] for k_ext in self.key_extractors]
        # all_head_keys: (head_num, agent_num, batch_size, embed_dim)

        # calculate attention per head
        head_qs = []
        head_attend_logits = []
        head_attend_weights = []
        for curr_head_keys, curr_head_selector in zip(all_head_keys, all_head_selectors):
            # curr_head_keys: (agent_num, batch_size, embed_dim)
            # curr_head_selector: (batch_size, embed_dim)

            # (batch_size, 1, embed_dim) * (batch_size, embed_dim, agent_num)
            attend_logits = th.matmul(curr_head_selector.view(-1, 1, self.embed_dim),
                                      th.stack(curr_head_keys).permute(1, 2, 0))
            # attend_logits: (batch_size, 1, agent_num)
            # scale dot-products by size of key (from Attention is All You Need)
            scaled_attend_logits = attend_logits / np.sqrt(self.embed_dim)
            if self.args.mask_dead:
                # actions: (episode_batch, episode_length - 1, agent_num, 1)
                actions = actions.reshape(-1, 1, self.n_agents)
                # actions: (batch_size, 1, agent_num)
                scaled_attend_logits[actions == 0] = -99999999  # action == 0 means the unit is dead
            attend_weights = F.softmax(scaled_attend_logits, dim=2)  # (batch_size, 1, agent_num)
            
            # (batch_size, 1, agent_num) * (batch_size, 1, agent_num)
            head_q = (agent_qs * attend_weights).sum(dim=2)
            head_qs.append(head_q)
            head_attend_logits.append(attend_logits)
            head_attend_weights.append(attend_weights)
        if self.args.state_bias:
            # State-dependent bias
            v = self.V(states).view(-1, 1)  # v: (bs, 1)
            # head_qs: [head_num, bs, 1]
            if self.args.weighted_head:
                w_head = th.abs(self.hyper_w_head(states))  # w_head: (bs, head_num)
                w_head = w_head.view(-1, self.n_head, 1)  # w_head: (bs, head_num, 1)
                y = th.stack(head_qs).permute(1, 0, 2)  # head_qs: (head_num, bs, 1); y: (bs, head_num, 1)
                y = (w_head * y).sum(dim=1) + v  # y: (bs, 1)
            else:
                y = th.stack(head_qs).sum(dim=0) + v  # y: (bs, 1)
        else:
            if self.args.weighted_head:
                w_head = th.abs(self.hyper_w_head(states))  # w_head: (bs, head_num)
                w_head = w_head.view(-1, self.n_head, 1)  # w_head: (bs, head_num, 1)
                y = th.stack(head_qs).permute(1, 0, 2)  # head_qs: (head_num, bs, 1); y: (bs, head_num, 1)
                y = (w_head * y).sum(dim=1)  # y: (bs, 1)
            else:
                y = th.stack(head_qs).sum(dim=0)  # y: (bs, 1)
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        # regularize magnitude of attention logits
        attend_mag_regs = self.attend_reg_coef * sum((logit ** 2).mean() for logit in head_attend_logits)
        head_entropies = [(-((probs + 1e-8).log() * probs).squeeze(dim=1).sum(1).mean()) for probs in head_attend_weights]
        return q_tot, attend_mag_regs, head_entropies

    def _build_residual_network(self):
        """Build the residual network Qr for ResQ"""
        layers = []
        
        # Input dimension: state + agent_qs + actions
        input_dim = self.state_dim + self.n_agents + self.n_agents * self.n_actions
        
        for i in range(self.residual_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, self.residual_embed_dim))
            elif i == self.residual_layers - 1:
                layers.append(nn.Linear(self.residual_embed_dim, 1))
            else:
                layers.append(nn.Linear(self.residual_embed_dim, self.residual_embed_dim))
            
            if i < self.residual_layers - 1:
                layers.append(nn.ReLU())
        
        residual_net = nn.Sequential(*layers)
        
        # Initialize with small weights for stability
        for layer in residual_net.modules():
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight, -self.resq_init_scale, self.resq_init_scale)
                if layer.bias is not None:
                    nn.init.uniform_(layer.bias, -self.resq_init_scale, self.resq_init_scale)
        
        return residual_net

    def _compute_mask_function(self, chosen_actions, max_actions):
        """Compute mask function wr(τ, u)"""
        # wr = 0 if u == ũ (max_actions), wr = 1 otherwise
        mask = (chosen_actions != max_actions).float()
        return mask

    def _forward_with_resq(self, agent_qs, states, actions, chosen_actions=None, max_actions=None):
        """Forward pass with ResQ residual network"""
        # First compute main Q_tot using original Qatten
        q_tot_main, attend_mag_regs, head_entropies = self.forward(agent_qs, states, actions)
        
        if not self.use_resq:
            return q_tot_main, attend_mag_regs, head_entropies, None, None
        
        # Compute residual Q value
        batch_size = agent_qs.size(0)
        agent_qs_flat = agent_qs.view(batch_size, -1)  # [batch_size, n_agents]
        states_flat = states.view(batch_size, -1)  # [batch_size, state_dim]
        
        if chosen_actions is not None and max_actions is not None:
            # Compute mask function wr
            mask_wr = self._compute_mask_function(chosen_actions, max_actions)  # [batch_size, 1]
            
            # Prepare input for residual network
            # Concatenate: states + agent_qs + actions_one_hot
            actions_one_hot = F.one_hot(chosen_actions.squeeze(-1), num_classes=self.n_actions).float()
            actions_one_hot = actions_one_hot.view(batch_size, -1)  # [batch_size, n_agents * n_actions]
            
            residual_input = th.cat([states_flat, agent_qs_flat, actions_one_hot], dim=1)
            
            # Forward through residual network
            q_residual = self.residual_network(residual_input)  # [batch_size, 1]
            
            # Ensure residual is always non-positive (Q_r ≤ 0)
            q_residual = -F.relu(q_residual)
            
            # Compute final Q with residual
            mask_wr_expanded = mask_wr.unsqueeze(-1)  # [batch_size, 1, 1]
            q_tot_final = q_tot_main + mask_wr_expanded * q_residual
            
            # Compute consistency loss term
            consistency_loss = self.consistency_loss_coef * (q_tot_final - q_tot_main - mask_wr_expanded * q_residual).pow(2).mean()
            
            return q_tot_final, attend_mag_regs, head_entropies, q_residual, consistency_loss
        
        else:
            # If no actions provided, return main Q only
            return q_tot_main, attend_mag_regs, head_entropies, None, None
