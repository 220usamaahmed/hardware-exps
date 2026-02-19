#!/bin/bash

set -euo pipefail

ROS_DISTRO="humble"

if [[ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
	# shellcheck disable=SC1091
	set +u
	source "/opt/ros/${ROS_DISTRO}/setup.bash"
	set -u
fi

sudo apt-get update

MOVEIT_PACKAGES=(
	"ros-${ROS_DISTRO}-moveit"
	"ros-${ROS_DISTRO}-moveit-ros-move-group"
	"ros-${ROS_DISTRO}-moveit-planners-ompl"
	"ros-${ROS_DISTRO}-moveit-ros-visualization"
	"ros-${ROS_DISTRO}-moveit-servo"
	"ros-${ROS_DISTRO}-moveit-setup-assistant"
)

echo "Installing MoveIt 2 packages (apt)..."
sudo apt-get install -y "${MOVEIT_PACKAGES[@]}" || true

USERNAME="${USER:-$(id -un)}"
ROS2_WS="/home/${USERNAME}/ros2"
SRC_PATH="${ROS2_WS}/src"
DRIVER_REPO="https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver.git"
DRIVER_DIR="${SRC_PATH}/Universal_Robots_ROS2_Driver"

mkdir -p "${SRC_PATH}"

if [[ ! -d "${DRIVER_DIR}" ]]; then
	echo "Cloning Universal Robots ROS 2 driver (humble)..."
	git clone -b "${ROS_DISTRO}" "${DRIVER_REPO}" "${DRIVER_DIR}"
else
	echo "Universal Robots ROS 2 driver already present."
fi

# Import upstream dependencies (includes ur_msgs) from the driver .repos file
REPOS_FILE="${DRIVER_DIR}/Universal_Robots_ROS2_Driver.${ROS_DISTRO}.repos"
if [[ -f "${REPOS_FILE}" ]]; then
	echo "Importing driver dependencies from ${REPOS_FILE}..."
	vcs import "${SRC_PATH}" < "${REPOS_FILE}"
fi

pushd "${ROS2_WS}" >/dev/null
rosdep update || true
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
popd >/dev/null

if ! grep -q "${ROS2_WS}/install/setup.bash" "/home/${USERNAME}/.bashrc"; then
	echo "source ${ROS2_WS}/install/setup.bash" >> "/home/${USERNAME}/.bashrc"
fi


