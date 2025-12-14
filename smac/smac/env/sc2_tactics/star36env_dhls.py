from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from smac.env.sc2_tactics.maps import get_map_params
from smac.env.sc2_tactics.utils import map_specific_utils
from smac.env.sc2_tactics.utils import common_utils

import atexit
from warnings import warn
from operator import attrgetter
from copy import deepcopy
import numpy as np
import enum
import math
from absl import logging

from pysc2 import maps
from pysc2 import run_configs
from pysc2.lib import protocol

from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import raw_pb2 as r_pb
from s2clientprotocol import debug_pb2 as d_pb

import smac.env.sc2_tactics.sc2_tactics_env as te

actions = {
    "move": 16,  # target: PointOrUnit
    "attack": 23,  # target: PointOrUnit
    "stop": 4,  # target: None
    "NydusCanalLoad": 1437,
    "NydusCanalUnload": 1438,
}

class Direction(enum.IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3


class SC2TacticsDHLSEnv(te.SC2TacticsEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load = {}
        print("----------------------")
        print("You create a DHLS env!")
        print("----------------------")
    
    def get_agent_action(self, a_id, action):
        """Construct the action for agent a_id."""
        avail_actions = self.get_avail_agent_actions(a_id)
        assert (
            avail_actions[action] == 1
        ), "Agent {} cannot perform action {}".format(a_id, action)

        unit = self.get_unit_by_id(a_id)
        tag = unit.tag
        x = unit.pos.x
        y = unit.pos.y

        if action == 0:
            # no-op (valid only when dead)
            assert unit.health == 0, "No-op only available for dead agents."
            if self.debug:
                logging.debug("Agent {}: Dead".format(a_id))
            return None
        
        elif a_id in self.load:
            return None

        elif action == 1:
            # stop
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["stop"],
                unit_tags=[tag],
                queue_command=False,
            )
            if self.debug:
                logging.debug("Agent {}: Stop".format(a_id))

        elif action == 2:
            # move north
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x, y=y + self._move_amount
                ),
                unit_tags=[tag],
                queue_command=False,
            )
            if self.debug:
                logging.debug("Agent {}: Move North".format(a_id))

        elif action == 3:
            # move south
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x, y=y - self._move_amount
                ),
                unit_tags=[tag],
                queue_command=False,
            )
            if self.debug:
                logging.debug("Agent {}: Move South".format(a_id))

        elif action == 4:
            # move east
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x + self._move_amount, y=y
                ),
                unit_tags=[tag],
                queue_command=False,
            )
            if self.debug:
                logging.debug("Agent {}: Move East".format(a_id))

        elif action == 5:
            # move west
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(
                    x=x - self._move_amount, y=y
                ),
                unit_tags=[tag],
                queue_command=False,
            )
            if self.debug:
                logging.debug("Agent {}: Move West".format(a_id))

        elif action == 6 and unit.unit_type == self.rlunit_ids.get("nydusNetwork"):
            # Load all units in the sight range
            target_tag = 0
            for t_id, t_unit in self.agents.items():
                if ((t_unit.unit_type == self.rlunit_ids.get("roach") or
                    t_unit.unit_type == self.rlunit_ids.get("zergling")) and 
                    self.distance(unit.pos.x, unit.pos.y, t_unit.pos.x, t_unit.pos.y)
                    <= self.unit_sight_range(t_id) and
                    t_id not in self.load):
                    target_tag = t_unit.tag
                    self.load[t_id] = t_unit
                    break
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["NydusCanalLoad"],
                target_unit_tag=target_tag,
                unit_tags=[tag],
                queue_command=False,
            )
            if self.debug:
                logging.debug("Agent {}: Load Unit {}".format(a_id, t_id))

        elif action == 6 and unit.unit_type == self.rlunit_ids.get("nydusCanal"):
            # Unload all agents
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["NydusCanalUnload"],
                unit_tags=[tag],
                queue_command=False,
            )
            if self.debug:
                logging.debug("Agent {}: Unload ALL Units".format(a_id))

        else:
            # attack/heal units that are in range
            target_id = action - self.n_actions_no_attack
            target_unit = self.enemies[target_id]
            action_name = "attack"

            action_id = actions[action_name]
            target_tag = target_unit.tag

            cmd = r_pb.ActionRawUnitCommand(
                ability_id=action_id,
                target_unit_tag=target_tag,
                unit_tags=[tag],
                queue_command=False,
            )

            if self.debug:
                logging.debug(
                    "Agent {} {}s unit # {}".format(
                        a_id, action_name, target_id
                    )
                )

        sc_action = sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd))
        return sc_action
    
    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        unit = self.get_unit_by_id(agent_id)
        if unit.health > 0:
            # cannot choose no-op when alive
            avail_actions = [0] * self.n_actions

            if unit.unit_type == self.rlunit_ids.get("hatchery"):
                avail_actions[1] = 1
                return avail_actions

            if (unit.unit_type != self.rlunit_ids.get("roach") and
                unit.unit_type != self.rlunit_ids.get("zergling")):
                # the structures in dhls can only stop
                avail_actions[1] = 1

                if unit.unit_type == self.rlunit_ids.get("nydusNetwork"):
                # check if roach can enter the NydusNetwork
                    for t_id, t_unit in self.agents.items():
                        if ((t_unit.unit_type == self.rlunit_ids.get("roach") or
                            t_unit.unit_type == self.rlunit_ids.get("zergling")) and
                            t_id not in self.load):
                            dist = self.distance(
                                unit.pos.x, unit.pos.y, t_unit.pos.x, t_unit.pos.y
                            )
                            sight_range = self.unit_sight_range(agent_id)
                            if dist <= sight_range:
                                avail_actions[6] = 1
                                break
                
                if unit.unit_type == self.rlunit_ids.get("nydusCanal") and self.load != {}:
                    avail_actions[6] = 1
                
                return avail_actions

            # stop should be allowed
            avail_actions[1] = 1

            # see if we can move
            if self.can_move(unit, Direction.NORTH):
                avail_actions[2] = 1
            if self.can_move(unit, Direction.SOUTH):
                avail_actions[3] = 1
            if self.can_move(unit, Direction.EAST):
                avail_actions[4] = 1
            if self.can_move(unit, Direction.WEST):
                avail_actions[5] = 1

            # Can attack only alive units that are alive in the shooting range
            shoot_range = self.unit_shoot_range(agent_id)

            target_items = self.enemies.items()

            for t_id, t_unit in target_items:
                if t_unit.health > 0:
                    dist = self.distance(
                        unit.pos.x, unit.pos.y, t_unit.pos.x, t_unit.pos.y
                    )
                    if dist <= shoot_range:
                        avail_actions[t_id + self.n_actions_no_attack] = 1

            return avail_actions

        else:
            # only no-op allowed
            return [1] + [0] * (self.n_actions - 1)

    def get_unit_type_id(self, unit, ally):
        """Returns the ID of unit type in the given scenario."""
        if ally:  # use new SC2 unit types
            if unit.unit_type == 86:
                type_id = 0
            elif unit.unit_type == 95:
                type_id = 1
            elif unit.unit_type == 142:
                type_id = 2
            elif unit.unit_type == 110:
                type_id = 3
            elif unit.unit_type == 105:
                type_id = 4
        else:
            if unit.unit_type == 48:
                type_id = 0
            elif unit.unit_type == 33:
                type_id = 1
            elif unit.unit_type == 18:
                type_id = 2
        return type_id
    
    def update_units(self):
        """Update units after an environment step.
        This function assumes that self._obs is up-to-date.
        """
        n_ally_alive = 0
        n_enemy_alive = 0

        # Store previous state
        self.previous_ally_units = deepcopy(self.agents)
        self.previous_enemy_units = deepcopy(self.enemies)

        # Check if roach is unloaded in dhls
        self.clean_load()

        for al_id, al_unit in self.agents.items():
            updated = False
            for unit in self._obs.observation.raw_data.units:
                if al_unit.tag == unit.tag:
                    self.agents[al_id] = unit
                    updated = True
                    n_ally_alive += 1
                    break


            updated = self.check_load(al_unit.tag, updated)

            if not updated:  # dead
                al_unit.health = 0

        for e_id, e_unit in self.enemies.items():
            updated = False
            for unit in self._obs.observation.raw_data.units:
                if e_unit.tag == unit.tag:
                    self.enemies[e_id] = unit
                    updated = True
                    n_enemy_alive += 1
                    break

            if not updated:  # dead
                e_unit.health = 0

        if (
            n_ally_alive == 0
            and n_enemy_alive > 0
            or self.check_end_code(ally=True)
        ):
            return -1  # lost
        if (
            n_ally_alive > 0
            and n_enemy_alive == 0
            or self.check_end_code(ally=False)
        ):
            return 1  # won
        if n_ally_alive == 0 and n_enemy_alive == 0:
            return 0

        return None
        
    def _kill_all_units(self):
        """Kill all units on the map."""
        units_alive = [
            unit.tag for unit in self.agents.values() if unit != None and unit.health > 0
        ] + [unit.tag for unit in self.enemies.values() if unit.health > 0] + [
            unit.tag for unit in self.load.values() if unit.health > 0
        ]
        self.load = {}
        debug_command = [
            d_pb.DebugCommand(kill_unit=d_pb.DebugKillUnit(tag=units_alive))
        ]
        self._controller.debug(debug_command)

    def _init_assign_aliases(self, min_unit_type):
        self._min_unit_type = min_unit_type
        self.rlunit_ids = common_utils.generate_unit_aliases_pure(self.map_name, min_unit_type)
        print(self.rlunit_ids)

    def check_structure(self, ally = True):
        """Check if the enemy's Nexus unit is killed."""
        if ally == True:
            for a in self.agents.values():
                if a.unit_type == self.rlunit_ids.get("hatchery") and a.health <= 0:
                    return True
        
        if ally == False:
            for e in self.enemies.values():
                if e.unit_type == 18 and e.health <= 0:
                    return True
        return False
    
    def check_unit_killed(self, ally = True):
        """Check if all the enemy's units are killed, except buildings"""
        if ally == False:
            for e in self.enemies.values():
                if e.unit_type != 18 and e.health > 0:
                    return False
            return True
        
        if ally == True:
            for a in self.agents.values():
                if ((a.unit_type == self.rlunit_ids.get("roach") or
                     a.unit_type == self.rlunit_ids.get("zergling")) and a.health > 0):
                    return False
            return True
        
    def clean_load(self):
        """If roach is closed enough to Nydus Canal, which means it is just unloaded,
           remove it from self.load"""
        for a_id, a_unit in self.agents.items():
            if a_unit.unit_type == self.rlunit_ids.get("nydusCanal"):
                del_id = []
                for t_id in self.load:
                    t_unit = self.get_unit_by_id(t_id)
                    dist = self.distance(a_unit.pos.x, a_unit.pos.y, t_unit.pos.x, t_unit.pos.y)
                    if dist <= 5:
                        del_id.append(t_id)
                for t_id in del_id:
                    del self.load[t_id]
                return
        return
    
    def check_load(self, a_tag, updated):
        """make sure the roach in load is not dead"""
        for t_unit in self.load.values():
            if a_tag == t_unit.tag:
                return True
        return updated

    def reset(self):
        result = super().reset()
        # init enemy base pos
        for e_id, e_unit in self.enemies.items():
            if e_unit.unit_type == 18:
                self.enemy_base_id = e_id
                self.enemy_base_pos = (e_unit.pos.x, e_unit.pos.y)
                break
        # calculate map diagonale distance
        self.map_diagonal_distance = math.sqrt(
            (self.map_x ** 2) + (self.map_y ** 2)
        )
        return result

    def reward_battle(self):
        """Reward function when self.reward_spare==False.
        Returns accumulative hit/shield point damage dealt to the enemy
        + reward_death_value per enemy unit killed, and, in case
        self.reward_only_positive == False, - (damage dealt to ally units
        + reward_death_value per ally unit killed) * self.reward_negative_scale
        """
        if self.reward_sparse:
            return 0

        reward = 0
        delta_deaths = 0
        delta_ally = 0
        delta_enemy = 0

        neg_scale = self.reward_negative_scale

        # update deaths
        for al_id, al_unit in self.agents.items():
            if self.check_unit_condition(al_unit, al_id):
                # did not die so far
                prev_health = 0
                if self.previous_ally_units[al_id] == None:
                    prev_health = al_unit.health + al_unit.shield
                else:
                    prev_health = (
                        self.previous_ally_units[al_id].health
                        + self.previous_ally_units[al_id].shield
                    )
                if al_unit.health == 0:
                    # just died
                    self.death_tracker_ally[al_id] = 1
                    if not self.reward_only_positive:
                        delta_deaths -= self.reward_death_value * neg_scale
                    delta_ally += prev_health * neg_scale
                else:
                    # still alive
                    delta_ally += neg_scale * (
                        prev_health - al_unit.health - al_unit.shield
                    )

        for e_id, e_unit in self.enemies.items():
            if e_unit != None and not self.death_tracker_enemy[e_id]:
                prev_health = (
                    self.previous_enemy_units[e_id].health
                    + self.previous_enemy_units[e_id].shield
                )
                if e_unit.health == 0:
                    self.death_tracker_enemy[e_id] = 1
                    delta_deaths += self.reward_death_value
                    delta_enemy += prev_health
                else:
                    delta_enemy += prev_health - e_unit.health - e_unit.shield

        
        # TODO Custom reward
        custom_reward = 0

        # reward for enemy units moving far away from base
        alpha = 0.2
        R_far = self.map_diagonal_distance * 0.4
        if self.ratio_enemies_attracted_by_roach() >= 0.8:
            # min enemy distance to base
            prev_dist_list = []
            curr_dist_list = []
            for e_id, e_unit in self.enemies.items():
                if e_unit == None or self.death_tracker_enemy[e_id]:
                    continue
                if e_unit.unit_type == 18 or e_unit.health <= 0:
                    continue
                prev_dist = math.sqrt(
                    (self.previous_enemy_units[e_id].pos.x - self.enemy_base_pos[0]) ** 2
                    + (self.previous_enemy_units[e_id].pos.y - self.enemy_base_pos[1]) ** 2
                )
                curr_dist = math.sqrt(
                    (e_unit.pos.x - self.enemy_base_pos[0]) ** 2
                    + (e_unit.pos.y - self.enemy_base_pos[1]) ** 2
                )
                prev_dist_list.append(prev_dist)
                curr_dist_list.append(curr_dist)
            if prev_dist_list and curr_dist_list:
                mean_prev_dist = np.mean(prev_dist_list)
                mean_curr_dist = np.mean(curr_dist_list)
                if (
                    math.isfinite(mean_prev_dist)
                    and math.isfinite(mean_curr_dist)
                ):
                    phi_prev = min(mean_prev_dist, R_far)
                    phi_curr = min(mean_curr_dist, R_far)
                    custom_reward += alpha * (phi_curr - phi_prev)

        # reward for zergling attack base and avoid fighting
        gamma = 0.3
        prev_base_hp = self.previous_enemy_units[self.enemy_base_id].health
        curr_base_hp = self.enemies[self.enemy_base_id].health
        base_damage = max(0.0, prev_base_hp - curr_base_hp)
        all_attackers_cnt = 0
        zergling_attackers_cnt = 0
        for a_id, a_unit in self.agents.items():
            if not self.check_unit_condition(a_unit, a_id):
                continue
            if not self.is_attacking_base(a_id):
                continue
            all_attackers_cnt += 1
            if(self.has_enemy_around(a_id)):
                continue
            if a_unit.unit_type == self.rlunit_ids.get("zergling"):
                zergling_attackers_cnt += 1
        if zergling_attackers_cnt > 0:
            custom_reward += gamma * base_damage * float(zergling_attackers_cnt / all_attackers_cnt)

        # TODO END

        if self.reward_only_positive:
            reward = abs(delta_enemy + delta_deaths + custom_reward)  # shield regeneration
        else:
            reward = delta_enemy + delta_deaths - delta_ally + custom_reward

        self.delta_enemy, self.delta_deaths, self.delta_ally = delta_enemy, delta_deaths, delta_ally

        return reward

    def is_attacking_base(self, al_id):
        """Check if an ally is attacking the enemy base."""
        action = int(np.argmax(self.last_action[al_id]))
        if action < self.n_actions_no_attack:
            return False
        target_id = action - self.n_actions_no_attack
        target_unit = self.enemies[target_id]
        if target_unit.unit_type == 18:
            return True
        return False
    
    def ratio_enemies_attracted_by_roach(self):
        """Calculate the ratio of enemy units that are within sight range of any roach."""
        n_attracted = 0
        n_total = 0
        for e_id, e_unit in self.enemies.items():
            if e_unit == None or self.death_tracker_enemy[e_id]:
                continue
            if e_unit.unit_type == 18 or e_unit.health <= 0:
                continue
            n_total += 1
            for a_id, a_unit in self.agents.items():
                if a_unit.unit_type != self.rlunit_ids.get("roach"):
                    continue
                if not self.check_unit_condition(a_unit, a_id):
                    continue
                dist = self.distance(
                    e_unit.pos.x, e_unit.pos.y, a_unit.pos.x, a_unit.pos.y
                )
                if dist <= self.unit_sight_range(a_id):
                    n_attracted += 1
                    break
        if n_total == 0:
            return 0.0
        return n_attracted / n_total

    def has_enemy_around(self, a_id):
        """Check if there is any enemy unit around the ally unit."""
        ally_unit = self.get_unit_by_id(a_id)
        for e_id, e_unit in self.enemies.items():
            if e_unit == None or self.death_tracker_enemy[e_id]:
                continue
            if e_unit.unit_type == 18 or e_unit.health <= 0:
                continue
            dist = self.distance(
                ally_unit.pos.x, ally_unit.pos.y, e_unit.pos.x, e_unit.pos.y
            )
            if dist <= self.unit_sight_range(a_id):
                return True
        return False