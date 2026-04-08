#!/bin/bash

# Define your setup command. 
# We use 'interactive' bash (-i) so it loads your .bashrc aliases.
SETUP_CMD="bash -ic 'py314 && python run_pyrite_model.py; exec bash'"
SETUP_CMD_BT="bash -ic 'py314 && python run_pyrite_model_w_bt.py; exec bash'"

# 1. Launch the first Konsole window, start a screen named 'model1', and run the setup
konsole --new-tab -e screen -S model1 bash -c "$SETUP_CMD" &

# 2. Launch the second Konsole window, start a screen named 'model2', and run the setup
konsole --new-tab -e screen -S model2 bash -c "$SETUP_CMD_BT" &
