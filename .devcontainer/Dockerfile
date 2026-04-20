ARG DOCKER_REPO
ARG ROS_DISTRO
ARG IMAGE_SUFFIX
FROM $DOCKER_REPO:$ROS_DISTRO$IMAGE_SUFFIX
ARG USERNAME
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Install additonal packages - add any that you need
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y apt-utils python3-pip python-is-python3 ssh neovim git python3-rosdep \
    build-essential cmake git python3-colcon-common-extensions python3-flake8 python3-rosdep python3-setuptools python3-vcstool wget
    
#Install MoveIt & ROS Conrol & Ros Controllers  
RUN apt-get update \
    && apt-get install -y \
        ros-${ROS_DISTRO}-ros2-control \
        ros-${ROS_DISTRO}-ros2-controllers \
        ros-${ROS_DISTRO}-rmw-cyclonedds-cpp

ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

ENV SHELL /bin/bash 

RUN usermod -aG dialout ${USERNAME}

# Source ROS environment automatically
RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> /home/$USERNAME/.bashrc

# Workspace helpers (mounted workspace root: /home/$USERNAME/ros2_ws)
RUN echo "ROS2_WS_ROOT=\"/home/$USERNAME/ros2_ws\"" >> /home/$USERNAME/.bashrc && \
    echo "if [ -f \"${ROS2_WS_ROOT}/install/local_setup.bash\" ]; then" >> /home/$USERNAME/.bashrc && \
    echo "  source \"${ROS2_WS_ROOT}/install/local_setup.bash\"" >> /home/$USERNAME/.bashrc && \
    echo "fi" >> /home/$USERNAME/.bashrc && \
    echo "function source_ws() {" >> /home/$USERNAME/.bashrc && \
    echo "  local ws=\"${ROS2_WS_ROOT}/$1\"" >> /home/$USERNAME/.bashrc && \
    echo "  if [ -f \"${ws}/install/local_setup.bash\" ]; then" >> /home/$USERNAME/.bashrc && \
    echo "    source \"${ws}/install/local_setup.bash\"" >> /home/$USERNAME/.bashrc && \
    echo "  else" >> /home/$USERNAME/.bashrc && \
    echo "    echo \"Workspace not found or not built: ${ws}\"" >> /home/$USERNAME/.bashrc && \
    echo "  fi" >> /home/$USERNAME/.bashrc && \
    echo "}" >> /home/$USERNAME/.bashrc

RUN echo "export ROS_DOMAIN_ID=10 # Sets the ROS domain id (between 1 and 101). Nodes can communicate only if they run in the same domain." >> /home/$USERNAME/.bashrc

# Set the default user
USER $USERNAME
CMD ["/bin/bash"]
