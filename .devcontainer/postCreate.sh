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

# Camera setup start

sudo apt install -y ros-$ROS_DISTRO-geographic-msgs
sudo apt install -y ros-$ROS_DISTRO-cob-srvs
sudo apt install -y ros-$ROS_DISTRO-robot-localization
sudo apt install -y ros-$ROS_DISTRO-point-cloud-transport

sudo apt install zstd

export ZED_CAMERA_WS=$OBJECT_PLACEMENT_ROOT/zed_ws
mkdir -p $ZED_CAMERA_WS/src
cd $ZED_CAMERA_WS

#Install CUDA 12.8

# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
# Download CUDA pin file if not already downloaded
if [ ! -f "cuda-ubuntu2204.pin" ]; then
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
fi
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
# wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb
# Download CUDA 12.8 local installer if not already downloaded
if [ ! -f "cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb" ]; then
  wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb
fi

sudo dpkg -i cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8

# Install CUDA 11.8

# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
# sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
# wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
# sudo dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
# sudo cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
# sudo apt-get update
# sudo apt-get -y install cuda



# wget https://download.stereolabs.com/zedsdk/5.0/cu12/ubuntu22?_gl=1*dlqgwb*_gcl_au*MTk1NjU2MDQyNy4xNzQzNzYyMzA1
# chmod +x ./ZED_SDK_Ubuntu22_cuda11.8_tensorrt10.9_v5.0.0.zstd.run
# sudo ./ZED_SDK_Ubuntu22_cuda11.8_tensorrt10.9_v5.0.0.zstd.run -- silent skip_cuda


# wget "https://download.stereolabs.com/zedsdk/5.0/cu12/ubuntu22?_gl=1*1586pgp*_gcl_au*MTk1NjU2MDQyNy4xNzQzNzYyMzA1" -O ZED_SDK_Ubuntu22_cuda12.8_tensorrt10.9_v5.0.0.zstd.run
if [ ! -f "ZED_SDK_Ubuntu22_cuda12.8_tensorrt10.9_v5.0.0.zstd.run" ]; then
  wget "https://download.stereolabs.com/zedsdk/5.0/cu12/ubuntu22?_gl=1*1586pgp*_gcl_au*MTk1NjU2MDQyNy4xNzQzNzYyMzA1" -O ZED_SDK_Ubuntu22_cuda12.8_tensorrt10.9_v5.0.0.zstd.run
fi

chmod +x ./ZED_SDK_Ubuntu22_cuda12.8_tensorrt10.9_v5.0.0.zstd.run
sudo ./ZED_SDK_Ubuntu22_cuda12.8_tensorrt10.9_v5.0.0.zstd.run -- silent skip_cuda

sudo chown -R $USER:$USER /usr/local/zed
sudo chmod -R a+rx /usr/local/zed
sudo chmod -R 755 /usr/local/zed/resources
sudo chown -R $(whoami):$(whoami) /usr/local/zed/resources

export ZED_SDK_ROOT=/usr/local/zed
export CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}:/usr/local/zed
# export CUDA_HOME=/usr/local/cuda-11.8
export CUDA_HOME=/usr/local/cuda-12.8
# export CUDA_HOME=/usr/local/cuda-export CUDA_HOME=/usr/local/cuda-11.8
export CUDA_HOME=/usr/local/cuda-export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$PATH:$CUDA_HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64

source /home/$(whoami)/.bashrc
cd src/
git clone https://github.com/stereolabs/zed-ros2-wrapper.git
git clone https://github.com/stereolabs/zed-ros2-interfaces.git -b humble
git clone https://github.com/ros-drivers/nmea_msgs.git -b ros2
cd ..
sudo apt update
# Install the required dependencies
# rosdep install --from-paths src --ignore-src -r -y
# Build the wrapper

sudo chown -R $(whoami):$(whoami) /usr/local/lib/python3.10/dist-packages/

colcon build --symlink-install --cmake-args=-DCMAKE_BUILD_TYPE=Release
# Setup the environment variables
echo source $(pwd)/install/local_setup.bash >> ~/.bashrc


mkdir -p ~/.zed
cd ~/.zed
wget "http://www.stereolabs.com/developers/calib/?SN=35477861" -O SN35477861.conf
chmod 644 ~/.zed/SN35477861.conf

# wget "http://www.stereolabs.com/developers/calib/?SN=2063" -O SN2063.conf
wget "http://calib.stereolabs.com/?SN=2063" -O SN2063.conf
chmod 644 ~/.zed/SN2063.conf

cd /usr/local/zed/settings
sudo wget "http://www.stereolabs.com/developers/calib/?SN=35477861" -O SN35477861.conf
sudo wget "http://www.stereolabs.com/developers/calib/?SN=2063" -O SN2063.conf
sudo chmod 644 SN35477861.conf
sudo chmod 644 SN2063.conf


sudo apt install -y libnvidia-encode-570
sudo apt install -y libnvidia-decode-570

echo source /home/$(whoami)/ros2/object_placement/zed_ws/install/local_setup.bash >> ~/.bashrc


# Camera setup end

pushd "${ROS2_WS}" >/dev/null
rosdep update || true
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
popd >/dev/null

if ! grep -q "${ROS2_WS}/install/setup.bash" "/home/${USERNAME}/.bashrc"; then
	echo "source ${ROS2_WS}/install/setup.bash" >> "/home/${USERNAME}/.bashrc"
fi


