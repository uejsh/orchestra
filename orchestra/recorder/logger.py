# logger.py - Core Orchestra Recorder API
# ============================================================================
# FILE: orchestra/recorder/logger.py
# Main entry point for recording traces and steps
# ============================================================================

import uuid
import time
import logging
import contextvars
import json
import threading
import queue
from typing import Optional, Dict, Any, List
from contextlib import contextmanager

from .base import BaseStorage, OrchestraEncoder
from .storage import SQLiteStorage

logger = logging.getLogger(__name__)

# Thread-local storage for current trace context
_current_trace_id = contextvars.ContextVar("current_trace_id", default=None)
_current_step_id = contextvars.ContextVar("current_step_id", default=None)


class BackgroundStorageWorker:
    """
    Worker thread that processes storage operations from a queue.
    Ensures that main execution path is not blocked by I/O.
    """
    def __init__(self, storage: BaseStorage):
        self.storage = storage
        self.queue = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.running = True
        self.thread.start()

    def _run(self):
        while self.running or not self.queue.empty():
            try:
                op, args, kwargs = self.queue.get(timeout=1.0)
                try:
                    method = getattr(self.storage, op)
                    method(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in background storage op '{op}': {e}")
                finally:
                    self.queue.task_done()
            except queue.Empty:
                continue

    def enqueue(self, op: str, *args, **kwargs):
        self.queue.put((op, args, kwargs))

    def stop(self):
        self.running = False
        self.thread.join(timeout=5.0)

class OrchestraRecorder:
    """
    Singleton-ish class to handle recording of traces.
    """
    _instance = None

    def __init__(self, storage: Optional[BaseStorage] = None, use_background: bool = True):
        self.storage = storage or SQLiteStorage()
        self.use_background = use_background
        
        if self.use_background:
            self.worker = BackgroundStorageWorker(self.storage)
            # Integrate with Lifecycle Manager for graceful shutdown
            try:
                from ..lifecycle import register_shutdown_handler
                register_shutdown_handler(self.worker.stop)
                logger.debug("Registered recorder worker for graceful shutdown")
            except ImportError:
                logger.warning("Could not register shutdown handler (lifecycle module missing)")
        else:
            self.worker = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = OrchestraRecorder()
        return cls._instance

    def _exec(self, op: str, *args, **kwargs):
        """Execute a storage operation either locally or in background."""
        if self.use_background and self.worker:
            self.worker.enqueue(op, *args, **kwargs)
        else:
            method = getattr(self.storage, op)
            method(*args, **kwargs)

    @contextmanager
    def trace(self, input_params: Any, metadata: Optional[Dict] = None):
        """
        Context manager to record a full execution trace.
        """
        trace_id = str(uuid.uuid4())
        token = _current_trace_id.set(trace_id)
        
        try:
            self._exec("create_trace", trace_id, input_params, metadata)
            
            yield trace_id
            
            # If successful completion
            self._exec(
                "update_trace",
                trace_id=trace_id, 
                status="SUCCESS"
            )
            
        except Exception as e:
            self._exec(
                "update_trace",
                trace_id=trace_id,
                status="ERROR",
                output_result={"error": str(e)}
            )
            raise e
        finally:
            _current_trace_id.reset(token)

    @contextmanager
    def step(self, node_name: str, input_state: Any):
        """
        Context manager to record a single step (node execution).
        """
        trace_id = _current_trace_id.get()
        if not trace_id:
            yield None
            return

        step_id = str(uuid.uuid4())
        step_token = _current_step_id.set(step_id)
        
        try:
            self._exec(
                "create_step",
                step_id=step_id,
                trace_id=trace_id,
                node_name=node_name,
                input_state=input_state
            )
            
            yield step_id
            
        except Exception as e:
            self._exec(
                "update_step",
                step_id=step_id, 
                status="ERROR",
                output_state={"error": str(e)}
            )
            raise e
        finally:
            _current_step_id.reset(step_token)

    def finish_step(self, step_id: str, output_state: Any, cost: float = 0.0):
        """
        Mark a step as finished.
        """
        if not step_id: return
        
        self._exec(
            "update_step",
            step_id=step_id,
            output_state=output_state,
            cost=cost,
            status="SUCCESS"
        )
        
    def finish_trace(self, trace_id: str, output_result: Any, total_cost: float = 0.0):
        """
        Update trace with final result.
        """
        self._exec(
            "update_trace",
            trace_id=trace_id,
            output_result=output_result,
            total_cost=total_cost,
            status="SUCCESS"
        )

    @staticmethod
    def compute_diff(state_a: Any, state_b: Any) -> Dict:
        """
        Compute a lightweight diff between two states (dicts).
        """
        if not isinstance(state_a, dict) or not isinstance(state_b, dict):
            return {"old": str(state_a), "new": str(state_b)}
            
        diff = {}
        all_keys = set(state_a.keys()) | set(state_b.keys())
        
        for k in all_keys:
            if k not in state_a:
                diff[k] = {"action": "added", "value": state_b[k]}
            elif k not in state_b:
                diff[k] = {"action": "removed", "value": state_a[k]}
            elif state_a[k] != state_b[k]:
                diff[k] = {"action": "changed", "old": state_a[k], "new": state_b[k]}
                
        return diff
