#!/bin/bash

# 1. Get the 6 lines following 'position:'
# 2. Use 'tr' to remove brackets, commas, AND the YAML dashes '-'
# 3. Use 'xargs' to normalize everything into a single line of space-separated numbers
RAW_POSITIONS=$(ros2 topic echo /joint_states --once | grep "position:" -A 6 | tail -n 6 | tr -d '[],-' | xargs)

echo "Cleaned Radians: $RAW_POSITIONS"

# Convert to degrees and format as a Python list
DEGREES_ARRAY=$(echo $RAW_POSITIONS | awk '{
    printf "[";
    for (i=1; i<=NF; i++) {
        deg = $i * 180 / 3.141592653589793;
        # Round to 2 decimal places
        printf "%.0f.0%s", deg, (i==NF ? "" : ", ");
    }
    print "]";
}')

echo "UR3e Joint Angles (Degrees):"
echo $DEGREES_ARRAY