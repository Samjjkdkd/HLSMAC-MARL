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


class SC2TacticsYQGZEnv(te.SC2TacticsEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("----------------------")
        print("You create a YQGZ env!")
        print("----------------------")

    def _init_assign_aliases(self, min_unit_type):
        self._min_unit_type = min_unit_type
        self.rlunit_ids = common_utils.generate_unit_aliases_pure(self.map_name, min_unit_type)
        print(self.rlunit_ids)

    def get_unit_type_id(self, unit, ally):
        """Returns the ID of unit type in the given scenario."""
        if ally:  # use new SC2 unit types
            if unit.unit_type == 105:
                type_id = 0 # zergling
            else:
                type_id = 99 # error
        else:  # use default SC2 unit types
            if unit.unit_type == 48:
                type_id = 0 # marine
            elif unit.unit_type == 33:
                type_id = 1 # siegeTank
            elif unit.unit_type == 32:
                type_id = 2 # siegeTankSieged
            else:
                type_id = 99 # error
        return type_id