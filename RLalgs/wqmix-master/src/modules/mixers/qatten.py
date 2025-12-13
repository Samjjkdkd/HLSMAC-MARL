# From https://github.com/wjh720/QPLEX/, added here for convenience.
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle as pkl


class GraphLearner(nn.Module):
    def __init__(self, args):
        super(GraphLearner, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.unit_dim = args.unit_dim
        self.hidden_dim = args.graph_hidden_dim
        
        # 构建边预测器网络
        self.edge_predictor = nn.Sequential(
            nn.Linear(2 * self.unit_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
        
        # 可选：添加dropout层以防止过拟合
        self.dropout = nn.Dropout(args.graph_dropout_rate)
    
    def forward(self, state, unit_states):
        # state: (batch_size, state_dim)
        # unit_states: (agent_num, batch_size, unit_dim)
        batch_size = state.size(0)
        
        # 将unit_states转换为(batch_size, agent_num, unit_dim)格式
        unit_states = unit_states.permute(1, 0, 2)
        
        # 准备所有智能体对的特征
        # 将unit_states复制为(agent_num, batch_size, agent_num, unit_dim)
        unit_states_i = unit_states.unsqueeze(2).repeat(1, 1, self.n_agents, 1)
        unit_states_j = unit_states.unsqueeze(1).repeat(1, self.n_agents, 1, 1)
        
        # 连接每对智能体的特征
        pair_features = th.cat([unit_states_i, unit_states_j], dim=-1)
        pair_features = pair_features.reshape(-1, 2 * self.unit_dim)
        
        # 应用dropout
        pair_features = self.dropout(pair_features)
        
        # 预测边的权重
        edge_logits = self.edge_predictor(pair_features)
        edge_logits = edge_logits.reshape(batch_size, self.n_agents, self.n_agents)
        
        # 使用sigmoid将边权重归一化到[0, 1]范围
        adjacency_matrix = F.sigmoid(edge_logits)
        
        # 可选：添加自环
        if self.args.graph_add_self_loop:
            adjacency_matrix = adjacency_matrix + th.eye(self.n_agents, device=adjacency_matrix.device).unsqueeze(0)
        
        return adjacency_matrix


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
        
        # 初始化图学习模块
        if args.use_graph_learner:
            self.graph_learner = GraphLearner(args)

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

        # 使用图学习模块生成邻接矩阵E
        adjacency_matrix = None
        if hasattr(self, 'graph_learner'):
            adjacency_matrix = self.graph_learner(states, unit_states)
            # adjacency_matrix: (batch_size, agent_num, agent_num)

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
            
            # 应用图学习得到的邻接矩阵调整注意力权重
            if adjacency_matrix is not None:
                # 对每个智能体，只考虑其邻接矩阵中的对应行
                # adjacency_matrix: (batch_size, agent_num, agent_num)
                # attend_weights: (batch_size, 1, agent_num)
                # 这里我们假设所有头共享同一个邻接矩阵，也可以为每个头学习不同的矩阵
                adjusted_attend_weights = attend_weights * adjacency_matrix.mean(dim=1).unsqueeze(1)
                # 重新归一化以确保权重和为1
                adjusted_attend_weights = adjusted_attend_weights / (adjusted_attend_weights.sum(dim=2, keepdim=True) + 1e-8)
                attend_weights = adjusted_attend_weights
            
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
        return q_tot, attend_mag_regs, head_entropies, adjacency_matrix
