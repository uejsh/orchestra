# event_listener.py - LangGraph Event Processing for Node-Level Observability
# ============================================================================
# FILE: orchestra/recorder/event_listener.py
# Processes LangGraph streaming events to capture node-level execution details
# ============================================================================

import asyncio
import time
import logging
from typing import Any, Dict, Optional

from .logger import OrchestraRecorder

logger = logging.getLogger(__name__)


class LangGraphEventListener:
    """
    Processes LangGraph astream_events to capture node-level execution.
    
    This solves the "blackbox problem" by recording:
    - Each node's input state
    - Each node's output state
    - State diffs between nodes
    - Timing for each node
    """
    
    def __init__(self, recorder: OrchestraRecorder, trace_id: str):
        self.recorder = recorder
        self.trace_id = trace_id
        self.prev_state: Dict[str, Any] = {}
        self.active_steps: Dict[str, Dict] = {}  # run_id -> step info
    
    async def process_graph_stream(
        self,
        graph,
        input_data: Dict[str, Any],
        config: Optional[Dict] = None,
        **kwargs
    ) -> Any:
        """
        Execute graph using astream_events and capture all node events.
        
        Args:
            graph: Compiled LangGraph
            input_data: Graph input
            config: LangGraph config
            
        Returns:
            Final graph output
        """
        final_output = None
        
        try:
            async for event in graph.astream_events(input_data, config=config, version="v2", **kwargs):
                event_type = event.get("event", "")
                event_name = event.get("name", "")
                run_id = event.get("run_id", "")
                data = event.get("data", {})
                
                # Filter to only process node-level events (not internal LangChain events)
                # LangGraph nodes typically emit "on_chain_start" and "on_chain_end"
                
                if event_type == "on_chain_start" and self._is_graph_node(event):
                    self._handle_node_start(event_name, run_id, data)
                    
                elif event_type == "on_chain_end" and self._is_graph_node(event):
                    output = self._handle_node_end(event_name, run_id, data)
                    if output is not None:
                        final_output = output
                        
                elif event_type == "on_chain_stream":
                    # Intermediate streaming - could capture partial outputs
                    pass
                    
        except Exception as e:
            logger.error(f"Error during graph streaming: {e}")
            raise
            
        return final_output
    
    def _is_graph_node(self, event: Dict) -> bool:
        """
        Determine if this event is from a graph node (vs internal LangChain).
        
        LangGraph node events typically have metadata indicating they're graph nodes,
        not nested LLM/chain calls.
        """
        metadata = event.get("metadata", {})
        tags = event.get("tags", [])
        
        # LangGraph nodes are tagged with "graph:step" or similar
        # Also check if it's a top-level node (not nested)
        langgraph_tags = [t for t in tags if "graph" in t.lower() or "node" in t.lower()]
        
        # If no special tags, check if parent_ids is empty (top-level)
        parent_ids = metadata.get("parent_ids", [])
        
        # Heuristic: Consider it a graph node if:
        # 1. It has langgraph-related tags, OR
        # 2. It has no parent (top-level chain)
        return bool(langgraph_tags) or len(parent_ids) <= 1
    
    def _handle_node_start(self, node_name: str, run_id: str, data: Dict):
        """Record the start of a node execution."""
        input_state = data.get("input", {})
        
        # Compute diff from previous state
        input_diff = OrchestraRecorder.compute_diff(self.prev_state, input_state)
        
        # Create step record
        step_id = self.recorder.storage.create_step(
            step_id=run_id,
            trace_id=self.trace_id,
            node_name=node_name,
            input_state=input_state,
            input_diff=input_diff if input_diff else None
        )
        
        # Track this step
        self.active_steps[run_id] = {
            "node_name": node_name,
            "input_state": input_state,
            "started_at": time.time()
        }
        
        logger.debug(f"📍 Node START: {node_name}")
    
    def _handle_node_end(self, node_name: str, run_id: str, data: Dict) -> Optional[Any]:
        """Record the end of a node execution."""
        output_state = data.get("output", {})
        
        step_info = self.active_steps.pop(run_id, None)
        if not step_info:
            logger.warning(f"Node END without matching START: {node_name}")
            return output_state
        
        input_state = step_info.get("input_state", {})
        
        # Compute output diff
        output_diff = OrchestraRecorder.compute_diff(input_state, output_state)
        
        # Update step record
        self.recorder.storage.update_step(
            step_id=run_id,
            output_state=output_state,
            output_diff=output_diff if output_diff else None,
            status="SUCCESS",
            ended_at=time.time()
        )
        
        # Update prev_state for next node
        if isinstance(output_state, dict):
            self.prev_state.update(output_state)
        else:
            self.prev_state = {"output": output_state}
        
        duration = time.time() - step_info["started_at"]
        logger.debug(f"✅ Node END: {node_name} ({duration:.3f}s)")
        
        return output_state


def run_graph_with_recording(
    graph,
    input_data: Dict[str, Any],
    recorder: OrchestraRecorder,
    trace_id: str,
    config: Optional[Dict] = None,
    **kwargs
) -> Any:
    """
    Synchronous wrapper to run a graph with node-level recording.
    
    Uses asyncio.run() internally to handle the async streaming.
    """
    listener = LangGraphEventListener(recorder, trace_id)
    
    async def _run():
        return await listener.process_graph_stream(graph, input_data, config, **kwargs)
    
    # Check if we're already in an async context
    try:
        loop = asyncio.get_running_loop()
        # Already in async context - create a new task
        # This is tricky; for simplicity, we'll use nest_asyncio pattern
        # or just return a coroutine for the caller to await
        raise RuntimeError("Cannot use sync wrapper inside async context. Use process_graph_stream directly.")
    except RuntimeError:
        # No running loop - safe to use asyncio.run
        return asyncio.run(_run())


async def run_graph_with_recording_async(
    graph,
    input_data: Dict[str, Any],
    recorder: OrchestraRecorder,
    trace_id: str,
    config: Optional[Dict] = None,
    **kwargs
) -> Any:
    """
    Async version for use in async contexts.
    """
    listener = LangGraphEventListener(recorder, trace_id)
    return await listener.process_graph_stream(graph, input_data, config, **kwargs)
