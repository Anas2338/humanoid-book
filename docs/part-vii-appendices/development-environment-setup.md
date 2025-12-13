---
sidebar_position: 1
---

# Appendix A: Development Environment Setup

## Overview

This appendix provides comprehensive guidance for setting up the development environment required for the Physical AI & Humanoid Robotics curriculum. The environment includes all necessary software tools, libraries, frameworks, and configurations needed to implement, test, and validate humanoid robot systems. Proper environment setup is crucial for successful completion of the course projects and ensures compatibility with the examples and code provided throughout the curriculum.

The development environment encompasses multiple domains: robotics frameworks (ROS 2), AI and machine learning libraries, simulation environments (Gazebo, Isaac Sim), computer vision tools, and real-time control systems. This appendix guides users through the installation, configuration, and validation of each component, ensuring a robust and reliable development platform for humanoid robotics projects.

The setup process involves installing core dependencies, configuring development tools, setting up simulation environments, and validating the complete environment. The instructions accommodate different operating systems and hardware configurations while maintaining consistency in the development experience across platforms.

## Learning Outcomes

By the end of this appendix, you should be able to:

- Install and configure the complete development environment for humanoid robotics
- Set up ROS 2 and associated robotics frameworks
- Configure simulation environments (Gazebo, Isaac Sim)
- Install and configure AI/ML libraries and tools
- Validate the development environment setup
- Troubleshoot common environment setup issues
- Maintain and update the development environment
- Optimize the environment for specific hardware configurations

## Key Concepts

### Core Development Components

Essential components of the development environment:

- **ROS 2 (Humble Hawksbill)**: Robot Operating System for robotics framework
- **Gazebo Simulation**: Physics simulation environment for robotics
- **Isaac Sim**: NVIDIA's simulation platform for AI-powered robots
- **Python 3.8+**: Primary programming language for robotics applications
- **CUDA Toolkit**: GPU computing platform for AI acceleration
- **OpenCV**: Computer vision library for perception systems
- **PyTorch/TensorFlow**: Machine learning frameworks for AI components

### Development Tools and IDEs

Recommended development tools and integrated development environments:

- **VS Code**: Primary IDE with ROS 2 extensions
- **PyCharm**: Alternative IDE for Python development
- **Git**: Version control system for code management
- **Docker**: Containerization for consistent environments
- **CMake**: Build system for C++ components
- **Colcon**: ROS 2 build tool for workspace management

### Hardware Requirements

Minimum and recommended hardware specifications:

- **CPU**: Multi-core processor (Intel i7 or AMD Ryzen 7 recommended)
- **RAM**: 16GB minimum, 32GB recommended for simulation
- **GPU**: NVIDIA GPU with CUDA support (RTX series recommended)
- **Storage**: 500GB SSD minimum for development environment
- **Network**: Stable internet connection for package installation
- **Peripherals**: Camera, sensors, and other hardware for testing

### Environment Validation

Procedures for validating the complete development environment:

- **ROS 2 Installation Verification**: Testing ROS 2 core functionality
- **Simulation Environment Testing**: Validating Gazebo and Isaac Sim
- **AI Framework Validation**: Confirming PyTorch/TensorFlow installation
- **Hardware Interface Testing**: Verifying sensor and actuator interfaces
- **Performance Benchmarking**: Assessing system performance capabilities
- **Sample Project Execution**: Running complete sample robotics projects

## Setup Instructions

### Step 1: System Requirements Check

Before beginning the installation, verify that your system meets the minimum requirements:

```bash
# Check operating system
uname -a

# Check available memory
free -h

# Check disk space
df -h

# Check CPU information
lscpu
```

For Ubuntu systems, ensure you're running Ubuntu 22.04 LTS or later, which is the officially supported platform for ROS 2 Humble Hawksbill.

### Step 2: Install Core System Dependencies

Install essential system packages and dependencies:

```bash
# Update package lists
sudo apt update && sudo apt upgrade -y

# Install basic development tools
sudo apt install -y build-essential cmake git python3-dev python3-pip

# Install essential libraries
sudo apt install -y libeigen3-dev libopencv-dev libboost-all-dev

# Install graphics drivers (for NVIDIA GPUs)
sudo apt install -y nvidia-driver-535 nvidia-utils-535

# Install additional utilities
sudo apt install -y curl wget vim htop tmux screen
```

### Step 3: Install Python Environment

Set up Python and virtual environment management:

```bash
# Install Python packages
pip3 install --upgrade pip setuptools wheel

# Install virtual environment tools
pip3 install virtualenv virtualenvwrapper

# Install development Python packages
pip3 install numpy scipy matplotlib pandas jupyter notebook
```

### Step 4: Install ROS 2 Humble Hawksbill

Install ROS 2 core components and essential packages:

```bash
# Add ROS 2 GPG key
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

# Add ROS 2 repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Update package lists
sudo apt update

# Install ROS 2 base packages
sudo apt install -y ros-humble-ros-base
sudo apt install -y ros-humble-desktop
sudo apt install -y ros-humble-perception
sudo apt install -y ros-humble-navigation2
sudo apt install -y ros-humble-nav2-bringup

# Install additional ROS packages
sudo apt install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
```

### Step 5: Install Gazebo Simulation

Install Gazebo simulation environment:

```bash
# Add Gazebo repository
sudo curl -sSL http://get.gazebosim.org | sh

# Install Gazebo Garden
sudo apt install gz-garden

# Install ROS 2 Gazebo bridge
sudo apt install ros-humble-gazebo-ros-pkgs
```

### Step 6: Install NVIDIA Isaac Sim (Optional)

For users with NVIDIA GPUs, install Isaac Sim components:

```bash
# Install CUDA Toolkit (version compatible with your GPU)
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run
sudo sh cuda_12.2.0_535.54.03_linux.run

# Install Isaac Sim (download from NVIDIA developer portal)
# Follow NVIDIA's installation instructions for Isaac Sim
```

### Step 7: Install AI/ML Frameworks

Install machine learning frameworks for AI components:

```bash
# Install PyTorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install TensorFlow
pip3 install tensorflow[and-cuda]

# Install additional AI libraries
pip3 install transformers datasets accelerate diffusers
```

### Step 8: Install Computer Vision Libraries

Install computer vision and image processing libraries:

```bash
# Install OpenCV
pip3 install opencv-python opencv-contrib-python

# Install additional vision libraries
pip3 install pillow scikit-image imageio

# Install ROS 2 vision packages
sudo apt install ros-humble-vision-opencv ros-humble-cv-bridge ros-humble-image-transport
```

### Step 9: Configure Development Environment

Set up environment variables and configurations:

```bash
# Add ROS 2 to bashrc
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# Add Gazebo to bashrc
echo "source /usr/share/gz/setup.bash" >> ~/.bashrc

# Set up workspace directory
mkdir -p ~/humanoid_ws/src
cd ~/humanoid_ws

# Source ROS 2 environment
source /opt/ros/humble/setup.bash

# Install colcon build tools
pip3 install -U colcon-common-extensions
```

### Step 10: Create and Build Workspace

Create a ROS 2 workspace for humanoid robotics development:

```bash
# Navigate to workspace
cd ~/humanoid_ws

# Build the workspace
colcon build --symlink-install

# Source the workspace
source install/setup.bash

# Add workspace to bashrc
echo "source ~/humanoid_ws/install/setup.bash" >> ~/.bashrc
```

## Environment Validation

### Test ROS 2 Installation

Verify that ROS 2 is properly installed:

```bash
# Check ROS 2 version
ros2 --version

# Test ROS 2 communication
# Terminal 1:
ros2 topic pub /chatter std_msgs/String "data: Hello World"

# Terminal 2 (after sourcing ROS 2):
ros2 topic echo /chatter std_msgs/String
```

### Test Gazebo Installation

Verify that Gazebo is properly installed:

```bash
# Launch Gazebo
gz sim

# Test Gazebo with ROS 2 bridge
ros2 launch gazebo_ros empty_world.launch.py
```

### Test Python Environment

Verify that Python packages are correctly installed:

```python
# Test Python environment
python3 -c "
import numpy as np
import cv2
import torch
import tensorflow as tf
import rclpy

print('NumPy version:', np.__version__)
print('OpenCV version:', cv2.__version__)
print('PyTorch version:', torch.__version__)
print('TensorFlow version:', tf.__version__)
print('ROS 2 Python client available')

# Test GPU availability (if NVIDIA GPU present)
if torch.cuda.is_available():
    print('CUDA GPU available:', torch.cuda.get_device_name(0))
else:
    print('CUDA GPU not available')
"
```

### Test Sample Project

Create and run a simple test project to validate the complete environment:

```bash
# Create a test package
cd ~/humanoid_ws/src
ros2 pkg create --build-type ament_python test_robot_control --dependencies rclpy geometry_msgs std_msgs sensor_msgs

# Build the workspace
cd ~/humanoid_ws
colcon build --packages-select test_robot_control

# Source the workspace
source install/setup.bash

# Run the test node
ros2 run test_robot_control test_robot_control
```

## Troubleshooting Common Issues

### ROS 2 Installation Issues

**Problem**: ROS 2 packages not found during installation
**Solution**:
```bash
sudo apt update
sudo apt upgrade
# Check repository configuration
cat /etc/apt/sources.list.d/ros2.list
```

**Problem**: Permission denied errors during ROS 2 build
**Solution**: Ensure proper workspace permissions
```bash
sudo chown -R $USER:$USER ~/humanoid_ws
```

### GPU/CUDA Issues

**Problem**: CUDA not detected by PyTorch
**Solution**: Verify CUDA installation and reinstall PyTorch with CUDA support
```bash
nvidia-smi  # Check GPU status
nvcc --version  # Check CUDA version
# Reinstall PyTorch with correct CUDA version
```

### Simulation Environment Issues

**Problem**: Gazebo fails to launch with graphics errors
**Solution**: Check graphics drivers and environment variables
```bash
# Check graphics environment
echo $DISPLAY
glxinfo | grep "OpenGL renderer"
```

### Python Package Conflicts

**Problem**: Package version conflicts or import errors
**Solution**: Use virtual environments to isolate dependencies
```bash
# Create virtual environment
python3 -m venv ~/humanoid_env
source ~/humanoid_env/bin/activate
pip install --upgrade pip
# Install packages in virtual environment
```

## Performance Optimization

### GPU Acceleration Setup

To optimize performance with GPU acceleration:

```bash
# Verify GPU setup
nvidia-smi

# Install GPU-optimized packages
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU acceleration
python3 -c "import torch; print('GPU available:', torch.cuda.is_available())"
```

### Memory Management

Configure system for optimal memory usage:

```bash
# Check current memory usage
free -h

# Configure swap space if needed (for systems with limited RAM)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Development Environment Optimization

Optimize the development environment for better performance:

```bash
# Use SSD for workspace storage
# Configure build tools for parallel compilation
# Set up efficient development workflows
```

## Maintenance and Updates

### Regular Maintenance Tasks

Perform regular maintenance to keep the environment healthy:

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Update Python packages
pip3 list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1 | xargs -n1 pip3 install -U

# Clean up unused packages
sudo apt autoremove && sudo apt autoclean
```

### Environment Updates

Update the development environment components:

```bash
# Update ROS 2 packages
sudo apt update
sudo apt upgrade ros-humble-*

# Update simulation environments
sudo apt upgrade gz-*

# Update AI/ML frameworks
pip3 install --upgrade torch torchvision torchaudio tensorflow
```

## Summary

This appendix provided comprehensive instructions for setting up the development environment required for the Physical AI & Humanoid Robotics curriculum. The environment includes ROS 2, simulation platforms, AI/ML frameworks, and all necessary tools for implementing and testing humanoid robot systems. Proper environment setup is essential for successful completion of the course projects and ensures compatibility with the provided examples and code. Regular maintenance and updates are important for keeping the development environment secure and efficient.