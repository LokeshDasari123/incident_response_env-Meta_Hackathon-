"""
envs/base_env.py
----------------
Abstract base environment. Defines the OpenEnv interface.
"""

from abc import ABC, abstractmethod
from models.action      import IncidentAction
from models.observation import IncidentObservation
from models.state       import IncidentState


class BaseIncidentEnv(ABC):

    @abstractmethod
    def reset(self, task_id: str = "easy") -> IncidentObservation: ...

    @abstractmethod
    def step(self, action: IncidentAction): ...

    @abstractmethod
    def state(self) -> IncidentState: ...

    @abstractmethod
    def close(self) -> None: ...
