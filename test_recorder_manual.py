
import os
import shutil
from orchestra.recorder.logger import OrchestraRecorder

def test_recorder():
    # Setup clean db
    if os.path.exists(".orchestra"):
        shutil.rmtree(".orchestra")
    
    rec = OrchestraRecorder.get_instance()
    
    print("Starting Trace...")
    with rec.trace(input_params={"q": "hello"}) as trace_id:
        print(f"Trace ID: {trace_id}")
        
        with rec.step("node_1", input_state={"foo": "bar"}) as step_id:
            print(f"Step ID: {step_id}")
            # Simulate work
            output = {"foo": "baz", "new": 123}
            # Diff check
            diff = rec.compute_diff({"foo": "bar"}, output)
            print(f"Diff: {diff}")
            
            rec.finish_step(step_id, output, cost=0.01)
            
    # Verify reading
    traces = rec.storage.list_traces()
    print(f"Stored Traces: {len(traces)}")
    if traces:
        print(f"Trace Status: {traces[0].status}")
        steps = rec.storage.get_trace_steps(traces[0].id)
        print(f"Stored Steps: {len(steps)}")
        if steps:
            print(f"Step Output: {steps[0].output_state}")

if __name__ == "__main__":
    test_recorder()
