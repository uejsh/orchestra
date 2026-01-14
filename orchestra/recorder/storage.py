# storage.py - SQLite storage for Orchestra Recorder
# ============================================================================
# FILE: orchestra/recorder/storage.py
# High-performance local storage for traces using SQLite
# ============================================================================

import sqlite3
import json
import time
import os
import logging
from typing import Dict, Any, List, Optional

from .base import BaseStorage, TraceRecord, StepRecord, OrchestraEncoder

logger = logging.getLogger(__name__)

class SQLiteStorage(BaseStorage):
    """
    Manager for the local SQLite trace database.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            cwd = os.getcwd()
            orchestra_dir = os.path.join(cwd, ".orchestra")
            os.makedirs(orchestra_dir, exist_ok=True)
            self.db_path = os.path.join(orchestra_dir, "traces.db")
        else:
            self.db_path = db_path
            
        self._init_db()
        
    def _get_conn(self):
        return sqlite3.connect(self.db_path)
    
    def _init_db(self):
        """Initialize the database schema."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            
            # Traces Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS traces (
                id TEXT PRIMARY KEY,
                started_at REAL,
                ended_at REAL,
                input_params TEXT,
                output_result TEXT,
                total_cost REAL,
                status TEXT,
                metadata TEXT
            )
            """)
            
            # Steps Table (Nodes)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS steps (
                id TEXT PRIMARY KEY,
                trace_id TEXT,
                node_name TEXT,
                input_state TEXT,
                output_state TEXT,
                input_diff TEXT,
                output_diff TEXT,
                started_at REAL,
                ended_at REAL,
                cost REAL,
                status TEXT,
                FOREIGN KEY(trace_id) REFERENCES traces(id)
            )
            """)
            
            # Indices for speed
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_traces_started ON traces(started_at DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_steps_trace_id ON steps(trace_id)")
            
            conn.commit()

    def create_trace(
        self, 
        trace_id: str, 
        input_params: Any,
        metadata: Optional[Dict] = None
    ):
        """Create a new trace entry."""
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO traces (id, started_at, input_params, total_cost, status, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    trace_id,
                    time.time(),
                    json.dumps(input_params, cls=OrchestraEncoder),
                    0.0,
                    "RUNNING",
                    json.dumps(metadata or {}, cls=OrchestraEncoder)
                )
            )

    def update_trace(self, trace_id: str, **kwargs):
        """Update an existing trace (completion)."""
        updates = []
        params = []
        
        allowed_fields = ["output_result", "total_cost", "status", "ended_at"]
        
        for k, v in kwargs.items():
            if k in allowed_fields:
                updates.append(f"{k} = ?")
                if isinstance(v, (dict, list)):
                    params.append(json.dumps(v, cls=OrchestraEncoder))
                else:
                    params.append(v)
        
        if "status" in kwargs and kwargs["status"] in ["SUCCESS", "ERROR"] and "ended_at" not in kwargs:
             updates.append("ended_at = ?")
             params.append(time.time())
             
        if not updates:
            return

        params.append(trace_id)
        sql = f"UPDATE traces SET {', '.join(updates)} WHERE id = ?"
        
        with self._get_conn() as conn:
            conn.execute(sql, tuple(params))

    def create_step(
        self,
        step_id: str,
        trace_id: str,
        node_name: str,
        input_state: Any,
        input_diff: Optional[Any] = None
    ):
        """Log the start of a step (node execution)."""
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO steps (id, trace_id, node_name, input_state, input_diff, started_at, cost, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    step_id,
                    trace_id,
                    node_name,
                    json.dumps(input_state, cls=OrchestraEncoder),
                    json.dumps(input_diff, cls=OrchestraEncoder) if input_diff else None,
                    time.time(),
                    0.0,
                    "RUNNING"
                )
            )

    def update_step(self, step_id: str, **kwargs):
        """Log the end of a step."""
        updates = []
        params = []
        
        allowed_fields = ["output_state", "output_diff", "cost", "status", "ended_at"]
        
        for k, v in kwargs.items():
            if k in allowed_fields:
                updates.append(f"{k} = ?")
                if isinstance(v, (dict, list)):
                    params.append(json.dumps(v, cls=OrchestraEncoder))
                else:
                    params.append(v)

        if "status" in kwargs and kwargs["status"] in ["SUCCESS", "ERROR"] and "ended_at" not in kwargs:
             updates.append("ended_at = ?")
             params.append(time.time())
             
        if not updates:
            return

        params.append(step_id)
        sql = f"UPDATE steps SET {', '.join(updates)} WHERE id = ?"
        
        with self._get_conn() as conn:
            conn.execute(sql, tuple(params))

    def list_traces(self, limit: int = 10, offset: int = 0) -> List[TraceRecord]:
        """List most recent traces."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM traces ORDER BY started_at DESC LIMIT ? OFFSET ?", 
                (limit, offset)
            )
            rows = cursor.fetchall()
            
            return [
                TraceRecord(
                    id=r[0],
                    started_at=r[1],
                    ended_at=r[2],
                    input_params=json.loads(r[3]) if r[3] else None,
                    output_result=json.loads(r[4]) if r[4] else None,
                    total_cost=r[5],
                    status=r[6],
                    metadata=json.loads(r[7]) if r[7] else {}
                )
                for r in rows
            ]

    def get_trace_steps(self, trace_id: str) -> List[StepRecord]:
        """Get all steps for a trace."""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM steps WHERE trace_id = ? ORDER BY started_at ASC", 
                (trace_id,)
            )
            rows = cursor.fetchall()
            
            return [
                StepRecord(
                    id=r[0],
                    trace_id=r[1],
                    node_name=r[2],
                    input_state=json.loads(r[3]) if r[3] else None,
                    output_state=json.loads(r[4]) if r[4] else None,
                    input_diff=json.loads(r[5]) if r[5] else None,
                    output_diff=json.loads(r[6]) if r[6] else None,
                    started_at=r[7],
                    ended_at=r[8],
                    cost=r[9],
                    status=r[10]
                )
                for r in rows
            ]

    def cleanup_old_traces(self, days: int):
        """Delete traces older than X days."""
        cutoff = time.time() - (days * 24 * 3600)
        
        with self._get_conn() as conn:
            # Delete steps first (foreign key)
            conn.execute(
                "DELETE FROM steps WHERE trace_id IN (SELECT id FROM traces WHERE started_at < ?)",
                (cutoff,)
            )
            # Delete traces
            cursor = conn.execute("DELETE FROM traces WHERE started_at < ?", (cutoff,))
            logger.info(f"Cleaned up {cursor.rowcount} traces older than {days} days.")
            conn.commit()

class PostgresStorage(BaseStorage):
    """
    Production PostgreSQL storage backend.
    Requires 'psycopg2-binary' or 'psycopg2'.
    """
    def __init__(self, dsn: str, pool_size: int = 10, max_overflow: int = 20):
        self.dsn = dsn
        try:
            import psycopg2
            import psycopg2.pool
            self.psycopg2 = psycopg2
        except ImportError:
            raise ImportError("PostgresStorage requires 'psycopg2'. Install: pip install psycopg2-binary")
            
        # Connection pool for production
        self.pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=pool_size + max_overflow,
            dsn=dsn
        )
        self._init_db()

    def _get_conn(self):
        return self.pool.getconn()
    
    def _return_conn(self, conn):
        self.pool.putconn(conn)

    def _init_db(self):
        """Initialize Postgres schema."""
        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                # Traces Table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS traces (
                    id TEXT PRIMARY KEY,
                    started_at DOUBLE PRECISION,
                    ended_at DOUBLE PRECISION,
                    input_params JSONB,
                    output_result JSONB,
                    total_cost DOUBLE PRECISION,
                    status TEXT,
                    metadata JSONB
                )
                """)
                
                # Steps Table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS steps (
                    id TEXT PRIMARY KEY,
                    trace_id TEXT REFERENCES traces(id) ON DELETE CASCADE,
                    node_name TEXT,
                    input_state JSONB,
                    output_state JSONB,
                    input_diff JSONB,
                    output_diff JSONB,
                    started_at DOUBLE PRECISION,
                    ended_at DOUBLE PRECISION,
                    cost DOUBLE PRECISION,
                    status TEXT
                )
                """)
                
                # Indices
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_traces_started ON traces(started_at DESC)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_steps_trace_id ON steps(trace_id)")
                
                conn.commit()
        finally:
            self._return_conn(conn)

    def create_trace(self, trace_id: str, input_params: Any, metadata: Optional[Dict] = None):
        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO traces (id, started_at, input_params, total_cost, status, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        trace_id,
                        time.time(),
                        json.dumps(input_params, cls=OrchestraEncoder),
                        0.0,
                        "RUNNING",
                        json.dumps(metadata or {}, cls=OrchestraEncoder)
                    )
                )
                conn.commit()
        finally:
            self._return_conn(conn)

    def update_trace(self, trace_id: str, **kwargs):
        conn = self._get_conn()
        try:
            updates = []
            params = []
            
            allowed_fields = ["output_result", "total_cost", "status", "ended_at"]
            
            for k, v in kwargs.items():
                if k in allowed_fields:
                    updates.append(f"{k} = %s")
                    if isinstance(v, (dict, list)):
                        params.append(json.dumps(v, cls=OrchestraEncoder))
                    else:
                        params.append(v)
            
            if "status" in kwargs and kwargs["status"] in ["SUCCESS", "ERROR"] and "ended_at" not in kwargs:
                 updates.append("ended_at = %s")
                 params.append(time.time())
                 
            if not updates:
                return

            params.append(trace_id)
            sql = f"UPDATE traces SET {', '.join(updates)} WHERE id = %s"
            
            with conn.cursor() as cursor:
                cursor.execute(sql, tuple(params))
                conn.commit()
        finally:
            self._return_conn(conn)

    def create_step(self, step_id: str, trace_id: str, node_name: str, input_state: Any, input_diff: Optional[Any] = None):
        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO steps (id, trace_id, node_name, input_state, input_diff, started_at, cost, status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        step_id,
                        trace_id,
                        node_name,
                        json.dumps(input_state, cls=OrchestraEncoder),
                        json.dumps(input_diff, cls=OrchestraEncoder) if input_diff else None,
                        time.time(),
                        0.0,
                        "RUNNING"
                    )
                )
                conn.commit()
        finally:
            self._return_conn(conn)

    def update_step(self, step_id: str, **kwargs):
        conn = self._get_conn()
        try:
            updates = []
            params = []
            
            allowed_fields = ["output_state", "output_diff", "cost", "status", "ended_at"]
            
            for k, v in kwargs.items():
                if k in allowed_fields:
                    updates.append(f"{k} = %s")
                    if isinstance(v, (dict, list)):
                        params.append(json.dumps(v, cls=OrchestraEncoder))
                    else:
                        params.append(v)

            if "status" in kwargs and kwargs["status"] in ["SUCCESS", "ERROR"] and "ended_at" not in kwargs:
                 updates.append("ended_at = %s")
                 params.append(time.time())
                 
            if not updates:
                return

            params.append(step_id)
            sql = f"UPDATE steps SET {', '.join(updates)} WHERE id = %s"
            
            with conn.cursor() as cursor:
                cursor.execute(sql, tuple(params))
                conn.commit()
        finally:
            self._return_conn(conn)

    def list_traces(self, limit: int = 10, offset: int = 0) -> List[TraceRecord]:
        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT * FROM traces ORDER BY started_at DESC LIMIT %s OFFSET %s", 
                    (limit, offset)
                )
                rows = cursor.fetchall()
                
                return [
                    TraceRecord(
                        id=r[0],
                        started_at=r[1],
                        ended_at=r[2],
                        input_params=r[3],  # Already JSONB
                        output_result=r[4],
                        total_cost=r[5],
                        status=r[6],
                        metadata=r[7] or {}
                    )
                    for r in rows
                ]
        finally:
            self._return_conn(conn)

    def get_trace_steps(self, trace_id: str) -> List[StepRecord]:
        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT * FROM steps WHERE trace_id = %s ORDER BY started_at ASC", 
                    (trace_id,)
                )
                rows = cursor.fetchall()
                
                return [
                    StepRecord(
                        id=r[0],
                        trace_id=r[1],
                        node_name=r[2],
                        input_state=r[3],
                        output_state=r[4],
                        input_diff=r[5],
                        output_diff=r[6],
                        started_at=r[7],
                        ended_at=r[8],
                        cost=r[9],
                        status=r[10]
                    )
                    for r in rows
                ]
        finally:
            self._return_conn(conn)

    def cleanup_old_traces(self, days: int):
        """Delete traces older than X days."""
        cutoff = time.time() - (days * 24 * 3600)
        conn = self._get_conn()
        try:
            with conn.cursor() as cursor:
                # Delete steps first (foreign key with CASCADE should handle this, but being explicit)
                cursor.execute(
                    "DELETE FROM steps WHERE trace_id IN (SELECT id FROM traces WHERE started_at < %s)",
                    (cutoff,)
                )
                # Delete traces
                cursor.execute("DELETE FROM traces WHERE started_at < %s", (cutoff,))
                count = cursor.rowcount
                conn.commit()
                logger.info(f"Cleaned up {count} traces older than {days} days.")
        finally:
            self._return_conn(conn)
    
    def close(self):
        """Close all connections in the pool."""
        if self.pool:
            self.pool.closeall()
