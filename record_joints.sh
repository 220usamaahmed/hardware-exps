#!/bin/bash

BASE_FRAME=${1:-base}
EE_FRAME=${2:-tool0}

TF_OUTPUT=$(ros2 run tf2_ros tf2_echo "$BASE_FRAME" "$EE_FRAME" 2>/dev/null | awk '
    /Translation:/ {t=$0}
    /Rotation: in Quaternion/ {r=$0}
    t && r {print t; print r; exit}
')

POSITION_LINE=$(echo "$TF_OUTPUT" | grep -m 1 "Translation:")
ROTATION_LINE=$(echo "$TF_OUTPUT" | grep -m 1 "Rotation: in Quaternion")

POSITION=$(echo "$POSITION_LINE" | sed -E 's/.*\[([^]]+)\].*/\1/')
ORIENTATION=$(echo "$ROTATION_LINE" | sed -E 's/.*\[([^]]+)\].*/\1/')

if [[ -z "$POSITION" || -z "$ORIENTATION" ]]; then
    echo "Failed to read TF from $BASE_FRAME to $EE_FRAME."
    echo "Check that TF is available and the frames are correct."
    exit 1
fi

echo "End-effector position (m):"
echo "[$POSITION]"
echo "End-effector orientation (quat):"
echo "[$ORIENTATION]"
echo "PoseTarget helper:"
echo "PoseTarget(position=[$POSITION], orientation=[$ORIENTATION])"