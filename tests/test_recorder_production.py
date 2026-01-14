import os
import time
import uuid
from datetime import datetime
from orchestra.recorder.logger import OrchestraRecorder
from orchestra.recorder.storage import SQLiteStorage

def test_background_writer_and_serialization():
    db_path = "test_prod.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        
    storage = SQLiteStorage(db_path=db_path)
    # Important: singleton might already exist, so we create a fresh one for the test
    recorder = OrchestraRecorder(storage=storage, use_background=True)
    
    unique_input = {"test_id": str(uuid.uuid4()), "time": datetime.now()}
    
    print("Starting Trace...")
    with recorder.trace(input_params=unique_input) as trace_id:
        with recorder.step("test_node", input_state={"step": 1}) as step_id:
            recorder.finish_step(step_id, output_state={"result": "ok"})
            
    # Wait for background worker to process
    print("Waiting for background worker...")
    time.sleep(1.0)
    
    # Check DB
    traces = storage.list_traces()
    print(f"Found {len(traces)} traces")
    assert len(traces) >= 1
    found = False
    for t in traces:
        if t.id == trace_id:
            found = True
            assert t.status == "SUCCESS"
            # Verify serialization of datetime (via OrchestraEncoder)
            assert "time" in t.input_params
            print("Trace verification passed")
            break
    assert found
    
    steps = storage.get_trace_steps(trace_id)
    assert len(steps) == 1
    assert steps[0].node_name == "test_node"
    print("Step verification passed")
    
    # Test Pruning
    storage.cleanup_old_traces(days=-1) # Force cleanup everything
    assert len(storage.list_traces()) == 0
    print("Cleanup verification passed")

    # Stop worker to close connections
    if recorder.worker:
        recorder.worker.stop()

    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except Exception as e:
            print(f"Cleanup warning: could not remove {db_path}: {e}")

if __name__ == "__main__":
    test_background_writer_and_serialization()
    print("\n✅ Verification passed!")
