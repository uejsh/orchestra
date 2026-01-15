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

def test_postgres_storage_mocked():
    """
    Test PostgresStorage logic without a real database using mocks.
    """
    from unittest.mock import MagicMock
    import sys
    
    # 1. Force Mock modules into sys.modules
    # We do this BEFORE importing PostgresStorage to ensure it picks up mocks
    mock_psycopg2 = MagicMock()
    mock_pool_module = MagicMock()
    
    # Save original modules if they exist
    original_psycopg2 = sys.modules.get("psycopg2")
    original_pool = sys.modules.get("psycopg2.pool")
    
    sys.modules["psycopg2"] = mock_psycopg2
    sys.modules["psycopg2.pool"] = mock_pool_module
    mock_psycopg2.pool = mock_pool_module

    try:
        from orchestra.recorder.storage import PostgresStorage
        
        # Setup specific return values
        mock_pool_instance = MagicMock()
        mock_conn = MagicMock()
        mock_cursor_ctx = MagicMock() 
        mock_cursor = MagicMock()
        
        # Determine what PostgresStorage calls
        # It calls psycopg2.pool.ThreadedConnectionPool(...)
        mock_pool_module.ThreadedConnectionPool.return_value = mock_pool_instance
        
        # It calls pool.getconn()
        mock_pool_instance.getconn.return_value = mock_conn
        
        # It calls conn.cursor() -> ctx -> cursor
        mock_conn.cursor.return_value = mock_cursor_ctx
        mock_cursor_ctx.__enter__.return_value = mock_cursor
        
        # Initialize
        storage = PostgresStorage(dsn="postgresql://user:pass@localhost/db")
        
        # 1. Test Initialization
        # Expected: Create Traces (1), Create Steps (1), Indices (2) = 4 execute calls
        assert mock_cursor.execute.call_count >= 2
        calls = [str(call) for call in mock_cursor.execute.mock_calls]
        assert any("CREATE TABLE IF NOT EXISTS traces" in c for c in calls)
        assert any("CREATE TABLE IF NOT EXISTS steps" in c for c in calls)
        
        # 2. Test Create Trace
        mock_cursor.reset_mock()
        storage.create_trace(
            trace_id="t1", 
            input_params={"q": "test"}, 
            metadata={"source": "test"}
        )
        
        args, _ = mock_cursor.execute.call_args
        sql = args[0]
        assert "INSERT INTO traces" in sql
        assert "VALUES (%s, %s, %s, %s, %s, %s)" in sql
        
        # 3. Test Update Trace
        mock_cursor.reset_mock()
        storage.update_trace("t1", output_result={"ans": "42"}, status="SUCCESS")
        
        args, _ = mock_cursor.execute.call_args
        sql = args[0]
        assert "UPDATE traces SET" in sql
        assert "output_result = %s" in sql
        assert "status = %s" in sql
        
        # 4. Test Cleanup
        mock_cursor.reset_mock()
        mock_cursor.rowcount = 5
        storage.cleanup_old_traces(days=7)
        
        assert mock_cursor.execute.call_count == 2
        assert "DELETE FROM traces" in str(mock_cursor.execute.mock_calls[1])
        
        # 5. Clean close
        storage.close()
        mock_pool_instance.closeall.assert_called_once()
        print("PG Mock Test Passed")
        
    finally:
        # Restore original modules to avoid Side Effects on other tests
        if original_psycopg2:
            sys.modules["psycopg2"] = original_psycopg2
        else:
            del sys.modules["psycopg2"]
            
        if original_pool:
            sys.modules["psycopg2.pool"] = original_pool
        else:
            del sys.modules["psycopg2.pool"]


