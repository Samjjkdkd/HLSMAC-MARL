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
    "move": 16,
    "attack": 23,
    "stop": 4,
    "BurrowDown": 1390,
    "BurrowUp": 1392,
}

class Direction(enum.IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3

class SC2TacticsADCCEnv(te.SC2TacticsEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_actions += 1
        self.n_actions_no_attack += 1
        print("----------------------")
        print("You create a ADCC env!")
        print("----------------------")
        
        # Initialize exploration history for each agent
        self.exploration_history = {}
        for agent_id in range(self.n_agents):
            self.exploration_history[agent_id] = set()
        
        # Initialize variables to track enemy base discovery for each agent
        self.enemy_base_discovered = [False for _ in range(self.n_agents)]
        self.enemy_base_reward_given = [False for _ in range(self.n_agents)]
    
    def get_agent_action(self, a_id, action):
        """Construct the action for agent a_id."""
        avail_actions = self.get_avail_agent_actions(a_id)
        assert (
            avail_actions[action] == 1
        ), "Agent {} cannot perform action {}".format(a_id, action)

        unit = self.get_unit_by_id(a_id)
        if unit == None:
            return None
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

        elif action == 6:
            # burrowDown
            if unit.unit_type == self.rlunit_ids.get("zergling"):
                cmd = r_pb.ActionRawUnitCommand(
                    ability_id=actions["BurrowDown"],
                    unit_tags=[tag],
                    queue_command=False,
                )
                if self.debug:
                    logging.debug("Agent {}: burrowDown".format(a_id))
            elif unit.unit_type == self.rlunit_ids.get("zerglingBurrowed"):
                cmd = r_pb.ActionRawUnitCommand(
                    ability_id=actions["BurrowUp"],
                    unit_tags=[tag],
                    queue_command=False,
                )
                if self.debug:
                    logging.debug("Agent {}: burrowUp".format(a_id))
            else:
                if self.debug:
                    logging.debug("Agent {} with type {} makes illegal burrow action".format(a_id, unit.unit_type))

        else:
            # attack units that are in range
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
    
    def get_unit_type_id(self, unit, ally):
        """Returns the ID of unit type in the given scenario."""
        if ally:  # use new SC2 unit types
            if unit.unit_type == 86:
                type_id = 0 # hatchery
            elif unit.unit_type == 105:
                type_id = 1 # zergling
            elif unit.unit_type == 119:
                type_id = 2 # zerglingBurrowed
            else:
                type_id = 99
                if self.debug:
                    logging.debug("Agent has unknown type: {}".format(unit.unit_type))
        else:  # use default SC2 unit types
            if unit.unit_type == 18:
                type_id = 0 # commandCenter
            elif unit.unit_type == 484:
                type_id = 1 # HellionTank
            else:
                type_id = 99
                if self.debug:
                    logging.debug("Enemy has unknown type: {}".format(unit.unit_type))
        return type_id
    
    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        unit = self.get_unit_by_id(agent_id)
        if unit.health > 0:
            # cannot choose no-op when alive
            avail_actions = [0] * self.n_actions

            # stop should be allowed
            avail_actions[1] = 1

            # see if the unit is Hatchery
            if unit.unit_type == self.rlunit_ids.get("hatchery"):
                return avail_actions
            
            # see if the unit is zerglingBurrowed
            if unit.unit_type == self.rlunit_ids.get("zerglingBurrowed"):
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

            avail_actions[6] = 1

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
    
    def _init_assign_aliases(self, min_unit_type):
        self._min_unit_type = min_unit_type
        self.rlunit_ids = common_utils.generate_unit_aliases_pure(self.map_name, min_unit_type)
        print(self.rlunit_ids)

    def check_structure(self, ally = True):
        """Check if the enemy's CommandCenter or the agent's Hatchery is destroyed."""
        if ally == False:
            for e in self.enemies.values():
                if e.unit_type == 18 and e.health <= 0:
                    return True
        if ally == True:
            for a in self.agents.values():
                if a.unit_type == self.rlunit_ids.get("hatchery") and a.health <= 0:
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
                if a.unit_type != self.rlunit_ids.get("hatchery") and a.health > 0:
                    return False
            return True
        return False
    
    def reset(self):
        """Reset the environment and clear exploration history."""
        result = super().reset()
        
        # Clear exploration history
        for agent_id in range(self.n_agents):
            self.exploration_history[agent_id] = set()
        
        # Reset enemy base discovery flags for each agent
        self.enemy_base_discovered = [False for _ in range(self.n_agents)]
        self.enemy_base_reward_given = [False for _ in range(self.n_agents)]
        
        return result
    
    def reward_battle(self):
        """Custom reward function for ADCC scenario.
        This overrides the base class reward_battle method to provide map-specific rewards.
        """
        if self.reward_sparse:
            return 0

        reward = 0
        delta_deaths = 0
        delta_ally = 0
        delta_enemy = 0

        neg_scale = self.reward_negative_scale

        # 1. Exploration reward: Encourage agents to explore new areas, prioritize distant areas from hatchery
        exploration_reward = 0
        
        # Find our hatchery position
        hatchery_pos = None
        for al_id, al_unit in self.agents.items():
            if al_unit.unit_type == self.rlunit_ids.get("hatchery") and al_unit.health > 0:
                hatchery_pos = (al_unit.pos.x, al_unit.pos.y)
                break
        
        for al_id, al_unit in self.agents.items():
            if al_unit.health > 0 and al_unit.unit_type != self.rlunit_ids.get("hatchery"):  # Only reward alive non-hatchery agents
                # Quantize position to a grid to track explored areas
                grid_size = 4.0  # Increased for 80x80 map scale
                grid_x = int(al_unit.pos.x // grid_size)
                grid_y = int(al_unit.pos.y // grid_size)
                grid_pos = (grid_x, grid_y)
                
                # If this is a new grid position for this agent
                if grid_pos not in self.exploration_history[al_id]:
                    self.exploration_history[al_id].add(grid_pos)
                    
                    if hatchery_pos is not None:
                        # Calculate distance from hatchery
                        dist_from_hatchery = self.distance(
                            al_unit.pos.x, al_unit.pos.y, hatchery_pos[0], hatchery_pos[1]
                        )
                        
                        # Scale reward based on distance from hatchery for 80x80 map
                        # Base reward: 5, maximum additional reward: 25 (total 30 for far areas)
                        # Reward increases with distance, capped at 30
                        # For 80x80 map, max possible distance is ~113, so denominator 4 gives good scaling
                        distance_bonus = min(25, dist_from_hatchery / 4)  # Adjusted for 80x80 map
                        area_reward = 5 + distance_bonus
                    else:
                        # If no hatchery found, use base reward
                        area_reward = 5
                    
                    exploration_reward += area_reward
        reward += exploration_reward
        
        # 2. Enemy base detection reward: High reward for each agent that first sees enemy CommandCenter
        for al_id, al_unit in self.agents.items():
            if al_unit.health > 0:
                x = al_unit.pos.x
                y = al_unit.pos.y
                sight_range = self.unit_sight_range(al_id)
                
                # Check if any enemy unit is a CommandCenter and within sight range
                for e_id, e_unit in self.enemies.items():
                    if e_unit.health > 0 and e_unit.unit_type == 18:  # 18 is CommandCenter
                        dist = self.distance(x, y, e_unit.pos.x, e_unit.pos.y)
                        if dist < sight_range:
                            self.enemy_base_discovered[al_id] = True
                            break
                
                # Give a large one-time reward when this agent first detects enemy base
                if self.enemy_base_discovered[al_id] and not self.enemy_base_reward_given[al_id]:
                    self.enemy_base_reward_given[al_id] = True
                    reward += 100  # Very large reward for finding enemy base
        
        # 3. Enemy base destruction reward: Increased reward for destroying enemy CommandCenter
        enemy_command_center_destroyed = self.check_structure(ally=False)
        if enemy_command_center_destroyed:
            reward += 500  # Massive reward for destroying enemy base
        
        # 4. Original battle rewards (damage and kills)
        # Update deaths
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

        # Add original battle rewards
        if self.reward_only_positive:
            battle_reward = abs(delta_enemy + delta_deaths)  # shield regeneration
        else:
            battle_reward = delta_enemy + delta_deaths - delta_ally
        
        reward += battle_reward

        self.delta_enemy, self.delta_deaths, self.delta_ally = delta_enemy, delta_deaths, delta_ally
        #print("adcc_reward:", reward, "delta_enemy:", delta_enemy, "delta_deaths:", delta_deaths, "delta_ally:", delta_ally,
        #      "exploration_reward:", exploration_reward, "battle_reward:", battle_reward)
        return reward
