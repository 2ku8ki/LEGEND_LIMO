# Jetson Nano (Ubuntu 18.04, arm64)에서 쓰는 ROS Melodic 기본 이미지
FROM ros:melodic-ros-base-bionic

# 기본 환경
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Seoul \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    ROS_PYTHON_VERSION=3

# 최소한으로 필요한 패키지들만 설치
# - python3 + numpy + yaml + opencv
# - 빌드를 위한 build-essential, cmake, git
# - 너 코드에서 쓰는 메시지/TF 쪽 ROS 패키지
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      locales tzdata \
      python3-pip python3-dev \
      python3-numpy python3-yaml python3-opencv \
      build-essential cmake git \
      libboost-all-dev \
      ros-melodic-geometry-msgs \
      ros-melodic-sensor-msgs \
      ros-melodic-nav-msgs \
      ros-melodic-tf \
      ros-melodic-image-transport \
      && rm -rf /var/lib/apt/lists/* && \
    locale-gen en_US.UTF-8 && \
    update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

# ROS Python3 유틸은 pip로만 설치 (apt 버전은 ROS랑 의존성 꼬여서 제거해버림)
# matplotlib은 LKAS 코드 import 대비용 (안 쓰면 나중에 직접 지워도 됨)
RUN pip3 install --no-cache-dir \
      rospkg catkin_pkg catkin_tools matplotlib

# ==========================
#  Python3용 cv_bridge 빌드
# ==========================
RUN mkdir -p /opt/cvbridge_ws/src
WORKDIR /opt/cvbridge_ws

# catkin workspace 설정 (catkin_tools는 위에서 pip로 설치됨)
RUN catkin init && \
    catkin config \
      -DPYTHON_EXECUTABLE=/usr/bin/python3 \
      -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m \
      -DPYTHON_LIBRARY=/usr/lib/aarch64-linux-gnu/libpython3.6m.so \
      --install

# vision_opencv (cv_bridge) 소스만 가져와서 빌드
WORKDIR /opt/cvbridge_ws/src
RUN git clone -b melodic https://github.com/ros-perception/vision_opencv.git

WORKDIR /opt/cvbridge_ws
RUN catkin build cv_bridge

# ==========================
#  유저 워크스페이스
# ==========================
RUN mkdir -p /root/LEGEND_WS/drive_ws/src
WORKDIR /root/LEGEND_WS/drive_ws

# 쉘 들어갔을 때 ROS + cv_bridge + 유저 WS 자동 로드
RUN echo "source /opt/ros/melodic/setup.bash" >> /root/.bashrc && \
    echo "source /opt/cvbridge_ws/install/setup.bash --extend" >> /root/.bashrc && \
    echo "if [ -f /root/LEGEND_WS/drive_ws/devel/setup.bash ]; then source /root/LEGEND_WS/drive_ws/devel/setup.bash; fi" >> /root/.bashrc && \
    echo "export ROS_PYTHON_VERSION=3" >> /root/.bashrc

# 엔트리포인트 추가
COPY entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]


