# From https://github.com/wjh720/QPLEX/, added here for convenience.
import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.qatten import QattenMixer
import torch as th
from torch.optim import RMSprop
from torch.optim import Adam
import numpy as np


class QattenLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0
        
        # Initialize TD parameters
        self.lam = getattr(args, 'lambda', 0.0)  # Get lambda from args, default to 0.0 (TD(0))
        self.n_step = getattr(args, 'n_step', 1)  # Get n_step from args, default to 1 (TD(0))

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            elif args.mixer == "qatten":
                self.mixer = QattenMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        if args.optimiser == 'rmsprop':
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        elif args.optimiser == 'adam':
            self.optimiser = Adam(
                params=self.params,
                lr=args.lr,
                betas=(args.adam_beta1, args.adam_beta2),
                eps=args.adam_eps,
                weight_decay=args.adam_weight_decay
            )

        # ResQ specific optimiser parameters
        self.use_resq = getattr(args, 'use_resq', False)
        self.main_params = self.params  # Default assignment
        self.main_optimiser = self.optimiser  # Default assignment
        self.residual_optimiser = None  # Default assignment
        
        if self.use_resq:
            self.residual_weight_decay = getattr(args, 'residual_weight_decay', 0.0)
            if self.residual_weight_decay > 0 and hasattr(self.mixer, 'residual_network'):
                # Separate optimizer for residual network with weight decay
                self.residual_params = list(self.mixer.residual_network.parameters())
                self.main_params = [p for p in self.params if p not in self.residual_params]
                
                self.main_optimiser = Adam(
                    params=self.main_params,
                    lr=args.lr,
                    betas=(args.adam_beta1, args.adam_beta2),
                    eps=args.adam_eps,
                    weight_decay=0.0  # No weight decay for main network
                )
                self.residual_optimiser = Adam(
                    params=self.residual_params,
                    lr=args.lr,
                    betas=(args.adam_beta1, args.adam_beta2),
                    eps=args.adam_eps,
                    weight_decay=self.residual_weight_decay
                )

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, show_demo=False, save_data=None):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        x_mac_out = mac_out.clone().detach()
        x_mac_out[avail_actions == 0] = -9999999
        max_action_qvals, max_action_index = x_mac_out[:, :-1].max(dim=3)

        max_action_index = max_action_index.detach().unsqueeze(3)
        is_max_action = (max_action_index == actions).int().float()

        if show_demo:
            q_i_data = chosen_action_qvals.detach().cpu().numpy()
            q_data = (max_action_qvals - chosen_action_qvals).detach().cpu().numpy()

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]
            cur_max_actions = target_mac_out.max(dim=3, keepdim=True)[1]  # get the indices
            # cur_max_actions: (episode_batch, episode_length - 1, agent_num, 1)
        target_next_actions = cur_max_actions.detach()  # actions are also inputs for mixer network

        # Mix
        if self.mixer is not None:
            if self.mixer.name == 'qatten':
                if self.use_resq:
                    # Use ResQ enhanced mixer
                    chosen_action_qvals, q_attend_regs, head_entropies, q_residual, consistency_loss = \
                        self.mixer._forward_with_resq(chosen_action_qvals, batch["state"][:, :-1], actions, 
                                                     actions, max_action_index)
                    # For target mixer, we don't need ResQ (no residual computation)
                    target_max_qvals, _, _ = self.target_mixer(target_max_qvals, batch["state"][:, 1:], target_next_actions)
                else:
                    # Use standard Qatten mixer
                    chosen_action_qvals, q_attend_regs, head_entropies = self.mixer(chosen_action_qvals, batch["state"][:, :-1], actions)
                    target_max_qvals, _, _ = self.target_mixer(target_max_qvals, batch["state"][:, 1:], target_next_actions)
                    q_residual = None
                    consistency_loss = None
            else:
                chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
                target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])
                q_residual = None
                consistency_loss = None
        else:
            q_residual = None
            consistency_loss = None

        # Calculate targets based on selected TD method
        # Priority: n_step > lambda (if n_step > 1, use n_step TD regardless of lambda)

        gamma = self.args.gamma
        rewards_t = rewards            # [B, L, 1]  where L = T-1
        terminated_t = terminated      # [B, L, 1]
        V_tp1 = target_max_qvals       # [B, L, ...]  (after mixer usually [B, L, 1])

        B, L = rewards_t.size(0), rewards_t.size(1)
        targets = th.zeros_like(rewards_t)  # [B, L, 1]

        if self.n_step > 1:
            n = self.n_step

            for t in range(L):
                # accumulate sum_{i=0}^{n-1} gamma^i r_{t+i}, stop if terminated happens
                G = th.zeros_like(rewards_t[:, t])  # [B, 1]
                alive = th.ones_like(terminated_t[:, t])  # [B, 1], 1 means not terminated yet

                for i in range(n):
                    ti = t + i
                    if ti >= L:
                        break
                    # only add reward if episode still alive before taking r_{ti}
                    G = G + (gamma ** i) * alive * rewards_t[:, ti]
                    # after adding reward at ti, update alive for next step (if terminated at ti, future rewards/bootstrap shouldn't count)
                    alive = alive * (1.0 - terminated_t[:, ti])

                # bootstrap term: gamma^n * V(s_{t+n}) if we still alive through step t+n-1
                bootstrap_idx = t + n - 1  # because V_tp1[:, k] == V(s_{k+1})
                if bootstrap_idx < L:
                    G = G + (gamma ** n) * alive * V_tp1[:, bootstrap_idx]

                targets[:, t] = G
                
        elif self.lam == 0:
            # TD(0): r_t + gamma * (1 - done_t) * V(s_{t+1})
            targets = rewards_t + gamma * (1.0 - terminated_t) * V_tp1

        else:
            # TD(lambda) backward recursion:
            # G_t^λ = r_t + gamma*(1-done_t)*[(1-λ)*V(s_{t+1}) + λ*G_{t+1}^λ]
            lam = self.lam

            G = rewards_t[:, -1] + gamma * (1.0 - terminated_t[:, -1]) * V_tp1[:, -1]
            targets[:, -1] = G

            for t in reversed(range(L - 1)):
                V_next = V_tp1[:, t]  # V(s_{t+1})
                G = rewards_t[:, t] + gamma * (1.0 - terminated_t[:, t]) * (
                    (1.0 - lam) * V_next + lam * G
                )
                targets[:, t] = G

        if show_demo:
            tot_q_data = chosen_action_qvals.detach().cpu().numpy()
            tot_target = targets.detach().cpu().numpy()
            print('action_pair_%d_%d' % (save_data[0], save_data[1]), np.squeeze(q_data[:, 0]),
                  np.squeeze(q_i_data[:, 0]), np.squeeze(tot_q_data[:, 0]), np.squeeze(tot_target[:, 0]))
            self.logger.log_stat('action_pair_%d_%d' % (save_data[0], save_data[1]),
                                 np.squeeze(tot_q_data[:, 0]), t_env)
            return

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        if self.mixer.name == 'qatten':
            if self.use_resq:
                # Include consistency loss for ResQ
                total_loss = (masked_td_error ** 2).sum() / mask.sum() + q_attend_regs
                if consistency_loss is not None:
                    total_loss += consistency_loss
                loss = total_loss
            else:
                loss = (masked_td_error ** 2).sum() / mask.sum() + q_attend_regs
        else:
            loss = (masked_td_error ** 2).sum() / mask.sum()

        masked_hit_prob = th.mean(is_max_action, dim=2) * mask
        hit_prob = masked_hit_prob.sum() / mask.sum()

        # Optimise
        if self.use_resq and self.residual_optimiser is not None:
            # Use separate optimizers for main and residual networks
            self.main_optimiser.zero_grad()
            self.residual_optimiser.zero_grad()
            loss.backward()
            
            # Clip gradients separately if needed
            grad_norm_main = th.nn.utils.clip_grad_norm_(self.main_params, self.args.grad_norm_clip)
            grad_norm_residual = th.nn.utils.clip_grad_norm_(self.residual_params, self.args.grad_norm_clip)
            
            self.main_optimiser.step()
            self.residual_optimiser.step()
            
            grad_norm = max(grad_norm_main, grad_norm_residual)
        else:
            # Use single optimizer for all parameters
            self.optimiser.zero_grad()
            loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
            self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("hit_prob", hit_prob.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            if self.mixer.name == 'qatten':
                # 监控头熵
                [self.logger.log_stat('head_{}_entropy'.format(h_i), ent.item(), t_env) for h_i, ent in enumerate(head_entropies)]
            if self.use_resq and q_residual is not None:
                self.logger.log_stat("residual_q_mean", q_residual.mean().item(), t_env)
                if consistency_loss is not None:
                    self.logger.log_stat("consistency_loss", consistency_loss.item(), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        
        # Save optimizers appropriately
        if self.use_resq and self.residual_optimiser is not None:
            th.save(self.main_optimiser.state_dict(), "{}/main_opt.th".format(path))
            th.save(self.residual_optimiser.state_dict(), "{}/residual_opt.th".format(path))
        else:
            th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        
        # Load optimizers appropriately
        if self.use_resq and self.residual_optimiser is not None:
            try:
                self.main_optimiser.load_state_dict(th.load("{}/main_opt.th".format(path), map_location=lambda storage, loc: storage))
                self.residual_optimiser.load_state_dict(th.load("{}/residual_opt.th".format(path), map_location=lambda storage, loc: storage))
            except FileNotFoundError:
                # Fallback to single optimizer if separate files don't exist
                self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
                self.main_optimiser = self.optimiser
                self.residual_optimiser = None
        else:
            self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
