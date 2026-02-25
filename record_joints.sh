#!/bin/bash

# 1. Get the 6 lines of positions
# 2. Use sed to remove leading '  - ' but keep the '-' if it's part of the number
# 3. xargs cleans up the whitespace into a single line
RAW_POSITIONS=$(ros2 topic echo /joint_states --once | grep "position:" -A 6 | tail -n 6 | sed 's/^[[:space:]]*-[[:space:]]*//' | xargs)

echo "Raw Radians: $RAW_POSITIONS"

# Convert to degrees and format as a Python list
DEGREES_ARRAY=$(echo $RAW_POSITIONS | awk '{
    printf "[";
    for (i=1; i<=NF; i++) {
        # awk handles scientific notation automatically in math
        deg = $i * 180 / 3.141592653589793;
        printf "%.2f%s", deg, (i==NF ? "" : ", ");
    }
    print "]";
}')

echo "UR3e Joint Angles (Degrees):"
echo $DEGREES_ARRAY