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
}

class Direction(enum.IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3


class SC2TacticsWWJZEnv(te.SC2TacticsEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("----------------------")
        print("You create a WWJZ env!")
        print("----------------------")

        # Initialize exploration history for each agent
        # self.exploration_history = {}
        # for agent_id in range(self.n_agents):
        #     self.exploration_history[agent_id] = set()
        
        # # Initialize variables to track enemy base discovery for each agent
        # self.enemy_base_discovered = [False for _ in range(self.n_agents)]
        # self.enemy_base_reward_given = [False for _ in range(self.n_agents)]
    
    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        unit = self.get_unit_by_id(agent_id)
        if unit.unit_type == self.rlunit_ids.get("nexus") and unit.health > 0:
            avail_actions = [0] * self.n_actions
            avail_actions[1] = 1
            return avail_actions
        else:
            return super().get_avail_agent_actions(agent_id)
        
    def _init_assign_aliases(self, min_unit_type):
        self._min_unit_type = min_unit_type
        self.rlunit_ids = common_utils.generate_unit_aliases_pure(self.map_name, min_unit_type)
        print(self.rlunit_ids)

    def get_unit_type_id(self, unit, ally):
        """Returns the ID of unit type in the given scenario."""
        if ally:  # use new SC2 unit types
            if unit.unit_type == 59:
                type_id = 0 # nexus
            elif unit.unit_type == 73:
                type_id = 1 # nexus
            else:
                type_id = 99 # error
        else:  # use default SC2 unit types
            if unit.unit_type == 18:
                type_id = 0 # commandCenter
            elif unit.unit_type == 48:
                type_id = 1 # marine
            elif unit.unit_type == 53:
                type_id = 1 # Hellion
            else:
                type_id = 99 # error
        return type_id

    def check_structure(self, ally = True):
        """Check if the enemy's Nexus unit is killed."""
        if ally == True:
            for a in self.agents.values():
                if a.unit_type == self.rlunit_ids.get("nexus") and a.health <= 0:
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
                if a.unit_type != self.rlunit_ids.get("nexus") and a.health > 0:
                    return False
            return True
    
    # def is_in_combat(self, agent_id):
    #     """Check if the agent is in combat (has visible enemy units within sight range)"""
    #     unit = self.get_unit_by_id(agent_id)
    #     if not self.check_unit_condition(unit, agent_id):
    #         return False
        
    #     # Skip nexus (building) units
    #     if unit.unit_type == self.rlunit_ids.get("nexus"):
    #         return False
        
    #     x = unit.pos.x
    #     y = unit.pos.y
    #     sight_range = self.unit_sight_range(agent_id)
        
    #     # Check if any enemy unit is visible and alive within sight range
    #     for e_id, e_unit in self.enemies.items():
    #         if e_unit != None and e_unit.health > 0:
    #             e_x = e_unit.pos.x
    #             e_y = e_unit.pos.y
    #             dist = self.distance(x, y, e_x, e_y)
    #             if dist < sight_range:
    #                 return True
        
    #     return False
    
    # def are_any_in_combat(self):
    #     """Check if any of the ally agents are in combat"""
    #     for al_id, al_unit in self.agents.items():
    #         # Skip nexus (building) units
    #         if al_unit.unit_type != self.rlunit_ids.get("nexus"):
    #             if self.is_in_combat(al_id):
    #                 return True
    #     return False
    
    # def get_agent_center(self):
    #     """Calculate the center position of all alive agents (excluding buildings)"""
    #     alive_agents = []
    #     for al_id, al_unit in self.agents.items():
    #         # Skip buildings (nexus) and dead units
    #         if al_unit.unit_type != self.rlunit_ids.get("nexus") and al_unit.health > 0:
    #             alive_agents.append(al_unit)
        
    #     if not alive_agents:
    #         return None
        
    #     total_x = sum(agent.pos.x for agent in alive_agents)
    #     total_y = sum(agent.pos.y for agent in alive_agents)
        
    #     return (total_x / len(alive_agents), total_y / len(alive_agents))
    
    # def reset(self):
    #     """Reset the environment and clear enemy base discovery flags."""
    #     result = super().reset()
        
    #     # Clear exploration history
    #     for agent_id in range(self.n_agents):
    #         self.exploration_history[agent_id] = set()
        
    #     # Reset enemy base discovery flags for each agent
    #     self.enemy_base_discovered = [False for _ in range(self.n_agents)]
    #     self.enemy_base_reward_given = [False for _ in range(self.n_agents)]
        
    #     return result
    
    # def reward_battle(self):
    #     """Custom reward function for WWJZ scenario.
    #     This overrides the base class reward_battle method to provide map-specific rewards.
    #     """
    #     if self.reward_sparse:
    #         return 0

    #     reward = 0
    #     delta_deaths = 0
    #     delta_ally = 0
    #     delta_enemy = 0
        
    #     # Initialize reward components for debugging
    #     formation_reward = None
    #     exploration_reward = None

    #     neg_scale = self.reward_negative_scale

    #     map_size_x = self.map_x
    #     map_size_y = self.map_y
    #     map_size_mean = (map_size_x + map_size_y) / 2
    #     #print("map_size_x:", map_size_x, "map_size_y:", map_size_y)
        
    #     # # 1. Enemy base proximity reward: Encourage movable units to approach enemy base before discovery
    #     # enemy_base_proximity_reward = 0
        
    #     # # Find enemy base (CommandCenter) position
    #     # enemy_base_pos = None
    #     # for e_id, e_unit in self.enemies.items():
    #     #     if e_unit.health > 0 and e_unit.unit_type == 18:  # 18 is CommandCenter
    #     #         enemy_base_pos = (e_unit.pos.x, e_unit.pos.y)
    #     #         break
        
    #     # # If enemy base is found, calculate proximity reward for each movable unit that hasn't discovered it yet
    #     # if enemy_base_pos is not None:
    #     #     for al_id, al_unit in self.agents.items():
    #     #         # Only apply to alive, movable units (not nexus) that haven't discovered enemy base
    #     #         if (al_unit.health > 0 and 
    #     #             al_unit.unit_type != self.rlunit_ids.get("nexus") and 
    #     #             not self.enemy_base_discovered[al_id]):
                    
    #     #             # Calculate distance from this unit to enemy base
    #     #             dist_to_enemy_base = self.distance(
    #     #                 al_unit.pos.x, al_unit.pos.y, enemy_base_pos[0], enemy_base_pos[1]
    #     #             )
                    
    #     #             # Reward scale factors
    #     #             max_reward = 10.0  # Maximum reward when very close to enemy base
    #     #             max_distance = map_size_mean * 0.8  # Distance beyond which no reward is given
                    
    #     #             # Calculate reward based on distance (inverse relationship)
    #     #             if dist_to_enemy_base < max_distance:
    #     #                 # Linear scaling: closer = higher reward
    #     #                 proximity_reward = max_reward * (1.0 - dist_to_enemy_base / max_distance)
    #     #                 enemy_base_proximity_reward += proximity_reward
        
    #     # reward += enemy_base_proximity_reward
    #     # exploration_reward = enemy_base_proximity_reward

    #     # 2. Enemy base detection reward: High reward for each agent that first sees enemy CommandCenter
    #     for al_id, al_unit in self.agents.items():
    #         if al_unit.health > 0:
    #             x = al_unit.pos.x
    #             y = al_unit.pos.y
    #             sight_range = self.unit_sight_range(al_id)
                
    #             # Check if any enemy unit is a CommandCenter and within sight range
    #             for e_id, e_unit in self.enemies.items():
    #                 if e_unit.health > 0 and e_unit.unit_type == 18:  # 18 is CommandCenter
    #                     dist = self.distance(x, y, e_unit.pos.x, e_unit.pos.y)
    #                     if dist < sight_range:
    #                         self.enemy_base_discovered[al_id] = True
    #                         break
                
    #             # Give a large one-time reward when this agent first detects enemy base
    #             if self.enemy_base_discovered[al_id] and not self.enemy_base_reward_given[al_id]:
    #                 self.enemy_base_reward_given[al_id] = True
    #                 reward += 100  # Very large reward for finding enemy base
        
    #     # 3. Original battle rewards (damage and kills)
    #     # Update deaths and damage for ally units
    #     for al_id, al_unit in self.agents.items():
    #         if self.check_unit_condition(al_unit, al_id):
    #             # did not die so far
    #             prev_health = 0
    #             if self.previous_ally_units[al_id] == None:
    #                 prev_health = al_unit.health + al_unit.shield
    #             else:
    #                 prev_health = (
    #                     self.previous_ally_units[al_id].health
    #                     + self.previous_ally_units[al_id].shield
    #                 )
    #             if al_unit.health == 0:
    #                 # just died
    #                 self.death_tracker_ally[al_id] = 1
    #                 if not self.reward_only_positive:
    #                     delta_deaths -= self.reward_death_value * neg_scale
    #                 delta_ally += prev_health * neg_scale
    #             else:
    #                 # still alive
    #                 delta_ally += neg_scale * (
    #                     prev_health - al_unit.health - al_unit.shield
    #                 )

    #     # Update deaths and damage for enemy units, with bonus for non-base units
    #     bonus_multiplier = 2  # Bonus multiplier for non-base enemy attacks
    #     for e_id, e_unit in self.enemies.items():
    #         if e_unit != None and not self.death_tracker_enemy[e_id]:
    #             prev_health = (
    #                 self.previous_enemy_units[e_id].health
    #                 + self.previous_enemy_units[e_id].shield
    #             )
                
    #             # Determine if this is a base unit (CommandCenter)
    #             is_base_unit = (e_unit.unit_type == 18)
                
    #             # Calculate multiplier: bonus for non-base units
    #             multiplier = 1.0 + (bonus_multiplier if not is_base_unit else 0.0)
                
    #             if e_unit.health == 0:
    #                 # just died
    #                 self.death_tracker_enemy[e_id] = 1
    #                 delta_deaths += self.reward_death_value
    #                 delta_enemy += prev_health * multiplier
    #             else:
    #                 # still alive, calculate damage dealt
    #                 current_health = e_unit.health + e_unit.shield
    #                 damage_dealt = prev_health - current_health
    #                 delta_enemy += damage_dealt * multiplier
        
    #     # 4. Add original battle rewards to total reward
    #     if self.reward_only_positive:
    #         battle_reward = abs(delta_enemy + delta_deaths)  # shield regeneration
    #     else:
    #         battle_reward = delta_enemy + delta_deaths - delta_ally
        
    #     reward += battle_reward
        
    #     # 5. Add team formation reward: keep agents together when not in combat
    #     if not self.are_any_in_combat():
    #         # Calculate formation reward only when no agent is in combat
    #         agent_center = self.get_agent_center()
    #         if agent_center is not None:
    #             center_x, center_y = agent_center
    #             alive_agents = [u for u in self.agents.values() if u.unit_type != self.rlunit_ids.get("nexus") and u.health > 0]
    #             if len(alive_agents) > 1:  # Only apply formation reward when there are multiple agents
    #                 # Calculate average distance from each agent to center
    #                 total_distance = 0.0
    #                 for agent in alive_agents:
    #                     dist = self.distance(agent.pos.x, agent.pos.y, center_x, center_y)
    #                     total_distance += dist
                    
    #                 avg_distance = total_distance / len(alive_agents)
                    
    #                 # print("center_x", center_x, "center_y", center_y, "avg_distance", avg_distance)
    #                 # Reward agents for staying close to each other
    #                 # The closer they are, the higher the reward
    #                 # Use a exponential decay function to reward proximity
    #                 formation_reward = max(0, map_size_mean/8 - avg_distance) # Scale down to avoid overwhelming other rewards
    #                 formation_reward = min(formation_reward, map_size_mean/12)
    #                 reward += formation_reward

    #     #print("battle_reward", battle_reward, "formation_reward", formation_reward, "exploration_reward", exploration_reward)

    #     # Store these values for potential debugging
    #     self.delta_enemy, self.delta_deaths, self.delta_ally = delta_enemy, delta_deaths, delta_ally

    #     return reward