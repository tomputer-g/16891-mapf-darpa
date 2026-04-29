"""
Canonical enum and constant definitions for the DARPA exploration simulation.

All display symbols, agent types, and simulation state enums live here so that
maps.py, agents.py, tasks.py, and visualizer.py share a single source of truth.

  ObservationState  – per-cell visibility / content (defines ASCII symbols)
  AgentType         – drone vs. ground-vehicle (defines ASCII symbols)
  PATH_SYMBOL       – ASCII symbol for planned-path waypoints
  AgentStatus       – operational state of a single agent
  EventType         – discrete event types produced during simulation
"""

from enum import Enum, auto


# ===========================================================================
# Map / cell types
# ===========================================================================

class ObservationState(Enum):
    """Visibility / content state of a single map cell.

    Each member's ASCII symbol is returned by .symbol() and is the canonical
    source of truth for all display and file-format conventions:

        '-'  UNKNOWN           — never observed by any agent
        '.'  FREE              — observed, passable open ground
        '@'  OBSTACLE          — observed, impassable
        'o'  OBJECTIVE         — observed, standalone objective (not in a building)
        '#'  BUILDING          — observed, unoccupied building (passable)
        'Q'  OCCUPIED_BUILDING — observed, building containing an objective
    """
    UNKNOWN           = auto()   # never seen by any agent
    FREE              = auto()   # observed, passable
    OBSTACLE          = auto()   # observed, impassable
    OBJECTIVE         = auto()   # observed, standalone objective (see ObjectiveTask)
    BUILDING          = auto()   # observed, unoccupied building (passable)
    OCCUPIED_BUILDING = auto()   # observed, building containing an objective

    # EXTEND: HAZARD, CONTESTED, EXPLORED_STALE, COMMS_RELAY, …

    def symbol(self) -> str:
        return {
            ObservationState.UNKNOWN:           '-',
            ObservationState.FREE:              '.',
            ObservationState.OBSTACLE:          '@',
            ObservationState.OBJECTIVE:         'o',
            ObservationState.BUILDING:          '#',
            ObservationState.OCCUPIED_BUILDING: 'Q',
        }[self]


class AgentType(Enum):
    """Type of agent; value is the canonical ASCII display symbol.

        'D'  DRONE  — aerial agent
        'G'  GROUND — ground vehicle / quadruped
    """
    DRONE  = 'D'
    GROUND = 'G'

    def symbol(self) -> str:
        return self.value


# Symbol used to render planned-path waypoints in ASCII maps.
PATH_SYMBOL = '*'


# ===========================================================================
# Agent / simulation state
# ===========================================================================

class AgentStatus(Enum):
    """Operational state of a single agent."""
    IDLE       = auto()   # no task currently assigned
    NAVIGATING = auto()   # following a planned path toward a task target
    REPLANNING = auto()   # needs to (re)compute a path

    # EXTEND: EXECUTING_TASK, WAITING_FOR_TEAM, DOCKING, CHARGING, …


class EventType(Enum):
    """Events exchanged in the simulation loop."""
    TASK_ASSIGNED = auto()   # auctioneer gave an agent a new task
    TASK_COMPLETE = auto()   # agent's current task is finished
    PATH_BLOCKED  = auto()   # next waypoint turned out to be an obstacle
    STEP_COMPLETE = auto()   # normal single-step advance

    # EXTEND: AGENT_JOINED, COMMS_RECEIVED, BATTERY_LOW, OBJECTIVE_FOUND, …
