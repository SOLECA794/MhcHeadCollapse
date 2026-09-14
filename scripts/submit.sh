#!/bin/bash
# oj-pilot: submit a version. usage: wsl bash submit.sh <ver>
cd /mnt/d/Desktop/OP-Learning/Ascend/cann-learing-hub/skills/cannjudge-submit-plaintext
python cannjudge_cli.py submit --problem-id 6a7c23a6a52e0f540a8a1779 --project-dir "/mnt/d/Desktop/OP-Learning/Ascend/xingchen-operators/C组赛题/【C组中等题】MhcHeadCollapse 算子（mHC四路归一）/versions/$1/code"
