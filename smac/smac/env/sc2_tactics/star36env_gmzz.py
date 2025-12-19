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
    "DepotLower": 556,
    "DepotRaise": 558,
}

class Direction(enum.IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3


class SC2TacticsGMZZEnv(te.SC2TacticsEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("----------------------")
        print("You create a GMZZ env!")
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
        elif unit.unit_type == self.rlunit_ids.get("Depot") and action == 6:
            # lower the supply depot
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["DepotLower"],
                unit_tags=[tag],
                queue_command=False,
            )
            if self.debug:
                logging.debug("Agent {}: Lower Supply Depot".format(a_id))
        elif unit.unit_type == self.rlunit_ids.get("DepotLowered") and action == 6:
            # Raise the supply depot
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["DepotRaise"],
                unit_tags=[tag],
                queue_command=False,
            )
            if self.debug:
                logging.debug("Agent {}: Raise Supply Depot".format(a_id))
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

            # stop should be allowed
            avail_actions[1] = 1

            if (unit.unit_type == self.rlunit_ids.get("Depot") or 
                unit.unit_type == self.rlunit_ids.get("DepotLowered")):
                avail_actions[6] = 1
                return avail_actions

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
            if unit.unit_type == 48:
                type_id = 0
            elif unit.unit_type == 19:
                type_id = 1
            elif unit.unit_type == 47:
                type_id = 2
            else:
                type_id = 99
                if self.debug:
                    logging.debug("Agent has unknown type: {}".format(unit.unit_type))
        else:
            if unit.unit_type == 105:
                type_id = 0
            elif unit.unit_type == 98:
                type_id = 1
            else:
                type_id = 99
                if self.debug:
                    logging.debug("Enemy has unknown type: {}".format(unit.unit_type))
        return type_id

    def _init_assign_aliases(self, min_unit_type):
        self._min_unit_type = min_unit_type
        self.rlunit_ids = common_utils.generate_unit_aliases_pure(self.map_name, min_unit_type)
        print(self.rlunit_ids)
    
    def check_unit_killed(self, ally = True):
        """Check if all the enemy's units are killed, except buildings"""
        if ally == False:
            for e in self.enemies.values():
                if e.unit_type != 98 and e.health > 0:
                    return False
            return True
        if ally == True:
            for a in self.agents.values():
                if (a.unit_type != self.rlunit_ids.get("Depot") and
                    a.unit_type != self.rlunit_ids.get("DepotLowered")
                    and a.health > 0):
                    return False
            return True
        return False
    
    def reset(self):
        """Reset the environment."""
        obs = super().reset()
        self.height_lower = []
        self.height_upper = []
        self.depot_posX = []
        self.depot_posY = []
        for a_id, a_unit in self.agents.items():
            if a_unit.unit_type == self.rlunit_ids.get("Depot") or a_unit.unit_type == self.rlunit_ids.get("DepotLowered"):
                posX = int(a_unit.pos.x)
                posY = int(a_unit.pos.y)
                self.height_upper.append(self.terrain_height[posX, posY])
                self.depot_posX.append(posX)
                self.depot_posY.append(posY)
        for e_id, e_unit in self.enemies.items():
            if e_unit.unit_type == 105:
                posX = int(e_unit.pos.x)
                posY = int(e_unit.pos.y)
                self.height_lower.append(self.terrain_height[posX, posY])
        self.height_lower = np.mean(self.height_lower)
        self.height_upper = np.mean(self.height_upper)
        self.depot_posX = np.mean(self.depot_posX)
        self.depot_posY = np.mean(self.depot_posY)
        # print("Height lower:", self.height_lower)
        # print("Height upper:", self.height_upper)
        self.ally_has_beento_lower = []
        self.enemy_has_beento_upper = []
        self.enemy_has_goback = []
        self.map_diagonal_distance = math.sqrt(
            (self.map_x ** 2) + (self.map_y ** 2)
        )
        return obs
    
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

        # ========Custom Reward===========
        custom_reward = 0
        
        # reward for depot sychronize
        depot_raised = 0
        for al_id, al_unit in self.agents.items():
            if not self.check_unit_condition(al_unit, al_id):
                continue
            if al_unit.unit_type == self.rlunit_ids.get("Depot"):
                depot_raised += 1
        # if depot_raised == 0 or depot_raised == 3:
        #     custom_reward += 3


        # one time mass negative reward for ally goto lower height
        for al_id, al_unit in self.agents.items():
            if not self.check_unit_condition(al_unit, al_id):
                continue
            if al_id in self.ally_has_beento_lower:
                continue
            posX = int(al_unit.pos.x)
            posY = int(al_unit.pos.y)
            height = self.terrain_height[posX, posY]
            if height < self.height_upper - 1e-3:
                self.ally_has_beento_lower.append(al_id)
                custom_reward -= 150

        # one time mass reward for enemy goto upper height and negative reward for go back to lower height
        for e_id, e_unit in self.enemies.items():
            if e_unit == None or self.death_tracker_enemy[e_id]:
                continue
            posX = int(e_unit.pos.x)
            posY = int(e_unit.pos.y)
            height = self.terrain_height[posX, posY]
            dist_to_depot = self.distance(e_unit.pos.x, e_unit.pos.y, self.depot_posX, self.depot_posY)
            dist_to_depot /= self.map_diagonal_distance
            if e_id in self.enemy_has_beento_upper:
                if e_id in self.enemy_has_goback:
                    continue
                if height < self.height_upper - 1e-2:
                    self.enemy_has_goback.append(e_id)
                    custom_reward -= 40
            else:
                if height >= self.height_upper and dist_to_depot > 0.1:
                    self.enemy_has_beento_upper.append(e_id)
                    custom_reward += 20

        
        # reward for lower the depot while zergling is below and raise the depot while zergling is above
        if len(self.enemy_has_beento_upper) < 8 and depot_raised == 3:
            custom_reward -= 10
        if len(self.enemy_has_beento_upper) >= 8 and depot_raised < 3:
            custom_reward -= 30

        
        # reward for ally stay away from depot
        for al_id, al_unit in self.agents.items():
            if not self.check_unit_condition(al_unit, al_id):
                continue
            if al_unit.unit_type == self.rlunit_ids.get("Depot") or al_unit.unit_type == self.rlunit_ids.get("DepotLowered"):
                continue
            dist_to_depot = self.distance(al_unit.pos.x, al_unit.pos.y, self.depot_posX, self.depot_posY)
            dist_to_depot /= self.map_diagonal_distance
            posX = int(al_unit.pos.x)
            posY = int(al_unit.pos.y)
            height = self.terrain_height[posX, posY]
            # if dist_to_depot > 0.2 and height >= self.height_upper:
            #     custom_reward += 1
            if dist_to_depot <= 0.15 and len(self.enemy_has_beento_upper) < 8:
                delta_enemy = 0  # no reward for damage when too close to depot and not all enemies have gone up

        # =============End================

        if self.reward_only_positive:
            reward = abs(delta_enemy + delta_deaths)  # shield regeneration
        else:
            reward = delta_enemy + delta_deaths - delta_ally
        
        reward += custom_reward

        self.delta_enemy, self.delta_deaths, self.delta_ally = delta_enemy, delta_deaths, delta_ally

        return reward
    
    def approximatelly_equal(self, a, b, tol=1e-3):
        return abs(a - b) <= tol