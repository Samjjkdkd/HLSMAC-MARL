from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


# This multi-agent controller uses different parameters for different unit types
class GMZZBasicMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        
        # 从观测中提取单位类型信息
        obs = ep_batch["obs"][:, t]
        bs = ep_batch.batch_size
        
        # 假设单位类型信息在观测的最后几个维度
        # 根据sc2_tactics_env.py中的get_obs_agent方法，单位类型在own_feats的最后几个维度
        # 这里需要根据实际的观测结构调整
        # 假设unit_type_bits=3（对应gmzz中的3种单位类型）
        unit_type_bits = 3
        type_one_hot = obs[:, :, -unit_type_bits:]
        
        # 将one-hot转换为type_id
        type_ids = th.argmax(type_one_hot, dim=-1).view(bs * self.n_agents)
        
        # 将type_id=2映射到type_id=1，让两种Depot共享参数
        type_ids = th.where(type_ids == 2, 1, type_ids)
        
        # 初始化输出和隐藏状态
        agent_outs = th.zeros(bs * self.n_agents, self.args.n_actions, device=ep_batch.device)
        new_hidden_states = th.zeros_like(self.hidden_states)
        
        # 为每种单位类型分别进行前向计算
        for type_id in self.agents.keys():
            # 找到该类型的所有agent
            mask = (type_ids == type_id)
            if not mask.any():
                continue
            
            # 提取该类型agent的输入和隐藏状态
            inputs = agent_inputs[mask]
            hidden = self.hidden_states.view(bs * self.n_agents, -1)[mask]
            
            # 使用对应的agent实例进行前向计算
            out, new_hidden = self.agents[type_id](inputs, hidden)
            
            # 保存结果
            agent_outs[mask] = out
            new_hidden_states.view(bs * self.n_agents, -1)[mask] = new_hidden
        
        # 更新隐藏状态
        self.hidden_states = new_hidden_states

        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(bs * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(bs, self.n_agents, -1)

    def init_hidden(self, batch_size):
        # 使用第一种agent类型来初始化隐藏状态形状
        first_agent = next(iter(self.agents.values()))
        self.hidden_states = first_agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        params = []
        for agent in self.agents.values():
            params.extend(agent.parameters())
        return params

    def load_state(self, other_mac):
        for type_id in self.agents.keys():
            self.agents[type_id].load_state_dict(other_mac.agents[type_id].state_dict())

    def cuda(self):
        for agent in self.agents.values():
            agent.cuda()

    def save_models(self, path):
        for type_id, agent in self.agents.items():
            th.save(agent.state_dict(), "{}/agent_{}.th".format(path, type_id))

    def load_models(self, path):
        for type_id, agent in self.agents.items():
            agent.load_state_dict(th.load("{}/agent_{}.th".format(path, type_id), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        # 创建不同兵种的agent实例，type_id为0的普通单位和type_id为1/2的Supply Depot
        # type_id=1和2的Depot共享参数
        self.agents = {
            0: agent_REGISTRY[self.args.agent](input_shape, self.args),  # 普通单位（如Marine）
            1: agent_REGISTRY[self.args.agent](input_shape, self.args)   # Supply Depot（type_id=1和2共享此参数）
        }

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
