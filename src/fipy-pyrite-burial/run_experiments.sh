#!/bin/bash

# 1. Define the list of Python scripts you want to run
MODELS=(
    "run_pyrite_model_year.py"
    "run_pyrite_model_year_bt.py"
    "run_pyrite_model_month.py"
    "run_pyrite_model_month_bt.py"
    "run_pyrite_model_week.py"
    "run_pyrite_model_week_bt.py"
)

# 2. Define the setup command (loading your alias)
# We use 'bash -ic' so your py314 alias is recognized
BASE_CMD="py314 && python"

for SCRIPT in "${MODELS[@]}"; do
    # Create a unique session name based on the filename (removing the .py)
    SESSION_NAME="${SCRIPT%.py}"
    
    echo "Launching $SESSION_NAME in a new Konsole tab..."

    # 3. The magic command:
    # --new-tab: Keeps your taskbar clean
    # screen -S: Names the session so you can find it later
    # bash -ic: Ensures your environment is exactly like your manual shell
    konsole --new-tab -e screen --qwindowtitle "$SESSION_NAME" -S "$SESSION_NAME" bash -ic "$BASE_CMD $SCRIPT; exec bash" &
    
    # Small sleep to prevent Konsole from glitching if opening many tabs at once
    sleep 0.5
done
