"""
Tasks and task queue for the DARPA exploration simulation.

  Task            – abstract base class for all assignable work items
  ExplorationTask – navigate to and observe an UNKNOWN frontier cell
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Set, Tuple

from sim_types import ObservationState

if TYPE_CHECKING:
    from agents import Agent
    from maps import KnownMap


# ===========================================================================
# Tasks
# ===========================================================================

class Task(ABC):
    """
    Abstract base class for all units of work assigned to agents.

    Subclasses must implement:
      target_loc         – where the agent should navigate toward
      check_completion() – return True the first time the task is done

    EXTEND:
      ObjectiveTask    – collect a reward item at a known location
      SurveillanceTask – loiter at a waypoint for N timesteps
      RelayTask        – establish a comms relay between two positions
      RescueTask       – reach a survivor cell and wait for handoff
    """

    _id_counter: int = 0

    def __init__(self, priority: float = 1.0) -> None:
        Task._id_counter += 1
        self.task_id:      int           = Task._id_counter
        self.priority:     float         = priority
        self.completed:    bool          = False
        self.assigned_to:  Optional[int] = None   # agent id

    @property
    @abstractmethod
    def target_loc(self) -> Tuple[int, int]: ...

    @abstractmethod
    def check_completion(self, known_map: "KnownMap") -> bool:
        """
        Called every tick by the simulation loop.
        Returns True (and sets self.completed) the first time the task
        transitions to the done state.
        Subclasses may inspect known_map to evaluate sensor-based conditions.
        """

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}("
                f"id={self.task_id}, loc={self.target_loc}, done={self.completed})")


class ExplorationTask(Task):
    """
    Frontier exploration task.

    Target    : a specific UNKNOWN cell adjacent to the known frontier.
    Completion: the target cell is no longer UNKNOWN in KnownMap, meaning
                some agent's sensor footprint has swept over it.  The agent
                does not need to stand on the cell — proximity sufficient to
                observe it is enough.
    """

    def __init__(self, loc: Tuple[int, int], priority: float = 1.0) -> None:
        super().__init__(priority)
        self._target_loc = loc

    @property
    def target_loc(self) -> Tuple[int, int]:
        return self._target_loc

    def check_completion(self, known_map: "KnownMap") -> bool:
        if not self.completed:
            r, c = self._target_loc
            if known_map.state[r][c] != ObservationState.UNKNOWN:
                self.completed = True
                return True
        return False

