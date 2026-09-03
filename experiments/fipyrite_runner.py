#!/usr/bin/env python3
"""
fipyrite_runner.py

A lightweight controller script to run and monitor long-running simulations.
It checks if the simulation process is active, monitors its progress via log files
to detect hangs, rotates/appends chunk logs to a history file, and automatically
restarts the simulation from its last known stage. It stops restarting if the
simulation reaches steady state, is interrupted by the user, or crashes repeatedly.

Designed to be executed periodically (e.g. every 5-10 minutes) via cron.
"""

import sys
import os
import json
import time
import subprocess
import re
import gzip
import psutil

# --- Configuration Constants ---
PYTHON_PATH = "/home/uliw/.local/share/mamba/envs/py314/bin/python"
TIMEOUT_SECONDS = 1800  # 30 minutes without log updates -> considered hung
MAX_FAILURES = 3        # Stop restarting if failed consecutively 3 times without step progress

def log_msg(msg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)

def is_process_running(pid, script_basename):
    """
    Checks if a process with the given PID is running and corresponds to our target script.
    """
    if pid is None:
        return None
    try:
        proc = psutil.Process(pid)
        if proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE:
            cmdline = proc.cmdline()
            if any(script_basename in arg for arg in cmdline):
                return proc
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return None

def kill_process_tree(proc):
    """
    Kills a process and all of its child processes.
    """
    try:
        log_msg(f"Killing process tree for PID {proc.pid}...")
        children = proc.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except Exception:
                pass
        proc.kill()
        # Wait up to 5 seconds for termination
        proc.wait(timeout=5)
        log_msg(f"Successfully killed PID {proc.pid}.")
    except Exception as e:
        log_msg(f"Error killing process PID {proc.pid}: {e}")

def get_last_step_and_status(log_path):
    """
    Parses the log file (or compressed .log.gz) to get the maximum step number and execution status.
    Returns (last_step, status).
    """
    target_path = log_path
    if not os.path.exists(target_path) and os.path.exists(log_path + ".gz"):
        target_path = log_path + ".gz"

    if not os.path.exists(target_path):
        return 0, "NO_LOG"
    
    last_step = 0
    converged = False
    interrupted = False
    completed = False
    
    try:
        open_fn = gzip.open if target_path.endswith(".gz") else open
        with open_fn(target_path, "rt", errors="ignore") as f:
            lines = f.readlines()[-50:]  # Read the last 50 lines to inspect end status
            
        for line in lines:
            if "Steady State Converged" in line:
                converged = True
            elif "Solver interrupted by user" in line:
                interrupted = True
            elif "Final Report:" in line:
                if "Steady State Converged" in line:
                    converged = True
                elif "interrupted" in line:
                    interrupted = True
                else:
                    completed = True  # e.g., max steps or simulation time reached
            
            # Parse step number, e.g. "Step  800"
            match = re.search(r"Step\s+(\d+)", line)
            if match:
                last_step = max(last_step, int(match.group(1)))
    except Exception as e:
        log_msg(f"Error parsing log file {target_path}: {e}")
        
    if converged:
        return last_step, "COMPLETED"
    elif interrupted:
        return last_step, "STOPPED_BY_USER"
    elif completed:
        return last_step, "CHUNK_DONE"
    else:
        return last_step, "RUNNING"

def append_log_to_history(log_path, history_path, chunk_number, exit_reason=""):
    """
    Appends the content of the current chunk's log file (or .log.gz) to a history log file.
    """
    target_path = log_path
    if not os.path.exists(target_path) and os.path.exists(log_path + ".gz"):
        target_path = log_path + ".gz"

    if not os.path.exists(target_path):
        return
    try:
        open_fn = gzip.open if target_path.endswith(".gz") else open
        with open_fn(target_path, "rt", errors="ignore") as f:
            content = f.read()
        
        with open(history_path, "a") as f_hist:
            f_hist.write(f"\n=== CHUNK {chunk_number} START ({exit_reason}) ===\n")
            f_hist.write(content)
            f_hist.write(f"=== END OF CHUNK {chunk_number} ===\n")
        log_msg(f"Appended log to history file: {history_path}")
    except Exception as e:
        log_msg(f"Error appending log to history: {e}")

def spawn_process(script_abs_path, stdout_path):
    """
    Spawns the target simulation process in the background.
    """
    script_dir = os.path.dirname(script_abs_path)
    script_basename = os.path.basename(script_abs_path)
    
    cmd = [PYTHON_PATH, script_basename]
    log_msg(f"Spawning: {' '.join(cmd)} in CWD: {script_dir}")
    
    try:
        # Open the .out file in append mode to capture all startup and traceback details
        with open(stdout_path, "a") as out_file:
            # We run the command with script_basename and set cwd=script_dir so it resolves
            # files relative to the script directory.
            proc = subprocess.Popen(
                cmd,
                cwd=script_dir,
                stdout=out_file,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid  # run in a new process group so we can kill easily
            )
        log_msg(f"Spawned successfully. PID: {proc.pid}")
        return proc.pid
    except Exception as e:
        log_msg(f"Failed to spawn process: {e}")
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: fipyrite_runner.py <target_script_name.py>")
        sys.exit(1)
        
    target_script = sys.argv[1]
    script_abs_path = os.path.abspath(target_script)
    script_dir = os.path.dirname(script_abs_path)
    script_basename = os.path.basename(script_abs_path)
    experiment_name = os.path.splitext(script_basename)[0]
    
    if not os.path.exists(script_abs_path):
        log_msg(f"Error: Target script {script_abs_path} does not exist.")
        sys.exit(1)
        
    # Define file paths
    log_path = os.path.join(script_dir, f"{experiment_name}.log")
    history_path = os.path.join(script_dir, f"{experiment_name}.log.history")
    stdout_path = os.path.join(script_dir, f"{experiment_name}.out")
    state_path = os.path.join(script_dir, f"{experiment_name}.runner.json")
    
    # Load runner state
    state = {
        "script_name": script_basename,
        "last_run_pid": None,
        "last_step_count": 0,
        "consecutive_failures": 0,
        "status": "IDLE",
        "last_start_time": 0.0,
        "chunk_number": 0
    }
    
    if os.path.exists(state_path):
        try:
            with open(state_path, "r") as f:
                state.update(json.load(f))
        except Exception as e:
            log_msg(f"Warning: Failed to load runner state from {state_path}: {e}")
            
    log_msg(f"Current Runner State: status={state['status']}, pid={state['last_run_pid']}, step={state['last_step_count']}, fails={state['consecutive_failures']}")
    
    # If the state was marked completed, user-stopped, or failed, do not do anything
    if state["status"] in ["COMPLETED", "STOPPED_BY_USER", "FAILED"]:
        log_msg(f"Simulation has reached state {state['status']}. No action taken.")
        return
        
    # Check if process is currently running
    proc = is_process_running(state["last_run_pid"], script_basename)
    
    if proc is not None:
        log_msg(f"Process PID {proc.pid} is running.")
        
        # Check for hangs: does the log file exist and has it been modified recently?
        if os.path.exists(log_path):
            log_mtime = os.path.getmtime(log_path)
            time_since_update = time.time() - log_mtime
            log_msg(f"Time since last log update: {time_since_update:.1f}s (Threshold: {TIMEOUT_SECONDS}s)")
            
            if time_since_update > TIMEOUT_SECONDS:
                log_msg(f"HUNG DETECTED: Log file has not been updated for {time_since_update:.1f}s.")
                kill_process_tree(proc)
                append_log_to_history(log_path, history_path, state["chunk_number"], "HUNG_AND_KILLED")
                
                # Treat as failure
                state["consecutive_failures"] += 1
                state["status"] = "FROZEN"
                
                if state["consecutive_failures"] >= MAX_FAILURES:
                    state["status"] = "FAILED"
                    log_msg(f"CRITICAL: Consecutive failure limit ({MAX_FAILURES}) reached. Disabling restarts.")
                else:
                    # Trigger restart
                    pid = spawn_process(script_abs_path, stdout_path)
                    if pid:
                        state["last_run_pid"] = pid
                        state["last_step_count"] = 0
                        state["last_start_time"] = time.time()
                        state["chunk_number"] += 1
                        state["status"] = "RUNNING"
            else:
                # Running normally. Let's update our cached step count in the state file.
                last_step, _ = get_last_step_and_status(log_path)
                if last_step > state["last_step_count"]:
                    state["last_step_count"] = last_step
        else:
            # Process is running but log file hasn't been created yet.
            # Check how long the process has been running.
            proc_age = time.time() - proc.create_time()
            log_msg(f"Process PID {proc.pid} is running but no log file exists yet. Process age: {proc_age:.1f}s")
            # If it's been running for over 10 minutes without creating a log, it might be stuck starting up.
            if proc_age > 600:
                log_msg("HUNG DETECTED: Process running for >10 mins without creating a log file.")
                kill_process_tree(proc)
                state["consecutive_failures"] += 1
                state["status"] = "STARTUP_HUNG"
                
                if state["consecutive_failures"] >= MAX_FAILURES:
                    state["status"] = "FAILED"
                else:
                    pid = spawn_process(script_abs_path, stdout_path)
                    if pid:
                        state["last_run_pid"] = pid
                        state["last_step_count"] = 0
                        state["last_start_time"] = time.time()
                        state["chunk_number"] += 1
                        state["status"] = "RUNNING"
    else:
        # Process is NOT running. Inspect why.
        log_msg("Process is not running. Checking logs for exit reason...")
        last_step, log_status = get_last_step_and_status(log_path)
        log_msg(f"Log status: {log_status}, Last parsed step: {last_step}")
        
        if log_status == "COMPLETED":
            log_msg("Steady state reached! Marking as completed.")
            state["status"] = "COMPLETED"
            append_log_to_history(log_path, history_path, state["chunk_number"], "COMPLETED")
        elif log_status == "STOPPED_BY_USER":
            log_msg("Simulation was interrupted by the user. Stopping restarts.")
            state["status"] = "STOPPED_BY_USER"
            append_log_to_history(log_path, history_path, state["chunk_number"], "STOPPED_BY_USER")
        else:
            # Either normal CHUNK_DONE (reached max_steps) or CRASHED (RUNNING log status but process died)
            if log_status == "CHUNK_DONE":
                log_msg("Previous chunk completed successfully. Resetting failures and preparing next chunk.")
                state["consecutive_failures"] = 0
                exit_reason = "CHUNK_DONE"
            else:
                # It died or crashed prematurely
                log_msg("Process terminated unexpectedly (crashed or killed). Checking progress...")
                # Did it progress at all since we last checked/recorded?
                if last_step > state["last_step_count"]:
                    log_msg(f"Progress was made: step {state['last_step_count']} -> {last_step}. Resetting failures.")
                    state["consecutive_failures"] = 0
                    state["last_step_count"] = last_step
                else:
                    state["consecutive_failures"] += 1
                    log_msg(f"No step progress made since last check. Consecutive failures: {state['consecutive_failures']}/{MAX_FAILURES}")
                exit_reason = f"CRASHED_STEP_{last_step}"
                
            append_log_to_history(log_path, history_path, state["chunk_number"], exit_reason)
            
            if state["consecutive_failures"] >= MAX_FAILURES:
                state["status"] = "FAILED"
                log_msg(f"CRITICAL: Simulation failed consecutively {MAX_FAILURES} times. Disabling restarts.")
            else:
                # Restart!
                pid = spawn_process(script_abs_path, stdout_path)
                if pid:
                    state["last_run_pid"] = pid
                    state["last_step_count"] = 0
                    state["last_start_time"] = time.time()
                    state["chunk_number"] += 1
                    state["status"] = "RUNNING"
                else:
                    state["consecutive_failures"] += 1
                    if state["consecutive_failures"] >= MAX_FAILURES:
                        state["status"] = "FAILED"
                        
    # Save updated state
    try:
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2)
        log_msg(f"Updated state saved to {state_path}")
    except Exception as e:
        log_msg(f"Error saving state JSON to {state_path}: {e}")

if __name__ == "__main__":
    main()
