# base.py - Abstract Base for Trace Storage
# ============================================================================
# FILE: orchestra/recorder/base.py
# ============================================================================

import json
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

class OrchestraEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle complex objects like UUID and datetime.
    """
    def default(self, obj):
        if isinstance(obj, uuid.UUID):
            return str(obj)
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)

@dataclass
class TraceRecord:
    id: str
    started_at: float
    ended_at: Optional[float]
    input_params: Optional[Dict[str, Any]]
    output_result: Optional[Dict[str, Any]]
    total_cost: float
    status: str
    metadata: Dict[str, Any]

@dataclass
class StepRecord:
    id: str
    trace_id: str
    node_name: str
    input_state: Optional[Dict[str, Any]]
    output_state: Optional[Dict[str, Any]]
    input_diff: Optional[Dict[str, Any]]
    output_diff: Optional[Dict[str, Any]]
    started_at: float
    ended_at: Optional[float]
    cost: float
    status: str

class BaseStorage(ABC):
    """Abstract base class for storage backends."""
    
    @abstractmethod
    def create_trace(self, trace_id: str, input_params: Any, metadata: Optional[Dict] = None):
        pass

    @abstractmethod
    def update_trace(self, trace_id: str, **kwargs):
        pass

    @abstractmethod
    def create_step(self, step_id: str, trace_id: str, node_name: str, input_state: Any, input_diff: Optional[Any] = None):
        pass

    @abstractmethod
    def update_step(self, step_id: str, **kwargs):
        pass

    @abstractmethod
    def list_traces(self, limit: int = 10, offset: int = 0) -> List[TraceRecord]:
        pass

    @abstractmethod
    def get_trace_steps(self, trace_id: str) -> List[StepRecord]:
        pass

    @abstractmethod
    def cleanup_old_traces(self, days: int):
        """Delete traces older than X days."""
        pass
