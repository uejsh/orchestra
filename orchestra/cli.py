#!/usr/bin/env python3
# cli.py - Orchestra CLI
# ============================================================================
# FILE: orchestra/cli.py
# CLI tools for inspecting recorder traces
# ============================================================================

import argparse
import sys
import json
import datetime
from typing import List, Optional
from .recorder.base import BaseStorage, TraceRecord, StepRecord
from .recorder.storage import SQLiteStorage

def format_timestamp(ts: float) -> str:
    return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

def format_duration(seconds: float) -> str:
    if seconds is None: return "N/A"
    return f"{seconds:.3f}s"

class OrchestraCLI:
    def __init__(self, storage: Optional[BaseStorage] = None):
        self.storage = storage or SQLiteStorage()
    
    def prune_traces(self, days: int):
        print(f"Pruning traces older than {days} days...")
        self.storage.cleanup_old_traces(days)
        print("Cleanup complete.")
        
    def list_traces(self, limit: int = 10):
        try:
            traces = self.storage.list_traces(limit=limit)
        except Exception as e:
            print(f"Error accessing trace database: {e}")
            return

        print(f"\n🎼 Orchestra Traces (Last {limit})\n")
        print(f"{'ID':<38} | {'Status':<10} | {'Started':<20} | {'Latency':<10} | {'Cost'}")
        print("-" * 100)
        
        for t in traces:
            latency = (t.ended_at - t.started_at) if t.ended_at else None
            print(
                f"{t.id:<38} | "
                f"{t.status:<10} | "
                f"{format_timestamp(t.started_at):<20} | "
                f"{format_duration(latency):<10} | "
                f"${t.total_cost:.4f}"
            )
        print("")

    def view_trace(self, trace_id: str):
        steps = self.storage.get_trace_steps(trace_id)
        if not steps:
            print(f"No steps found for trace {trace_id}")
            return
            
        print(f"\n🔍 Trace Inspection: {trace_id}\n")
        
        for i, step in enumerate(steps):
            latency = (step.ended_at - step.started_at) if step.ended_at else None
            
            print(f"[{i+1}] {step.node_name} ({format_duration(latency)})")
            print(f"    Status: {step.status}")
            
            if step.input_diff:
                 print(f"    Input Diff: {json.dumps(step.input_diff, indent=2)}")
            elif step.input_state:
                 # Truncate
                 inp = str(step.input_state)
                 if len(inp) > 100: inp = inp[:100] + "..."
                 print(f"    Input: {inp}")

            print("")

def main():
    parser = argparse.ArgumentParser(description="Orchestra CLI")
    subparsers = parser.add_subparsers(dest="command", help="start|list|view")
    
    # helper for 'trace' command group
    trace_parser = subparsers.add_parser("trace", help="Manage traces")
    trace_subs = trace_parser.add_subparsers(dest="subcommand")
    
    # trace ls
    ls_parser = trace_subs.add_parser("ls", help="List recent traces")
    ls_parser.add_argument("--limit", type=int, default=10)
    
    # trace view
    view_parser = trace_subs.add_parser("view", help="View specific trace")
    view_parser.add_argument("id", type=str, help="Trace ID")
    
    # trace prune
    prune_parser = trace_subs.add_parser("prune", help="Prune old traces")
    prune_parser.add_argument("--days", type=int, default=30, help="Days to keep")
    
    args = parser.parse_args()
    
    cli = OrchestraCLI()
    
    if args.command == "trace":
        if args.subcommand == "ls":
            cli.list_traces(limit=args.limit)
        elif args.subcommand == "view":
            cli.view_trace(trace_id=args.id)
        elif args.subcommand == "prune":
            cli.prune_traces(days=args.days)
        else:
            trace_parser.print_help()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
