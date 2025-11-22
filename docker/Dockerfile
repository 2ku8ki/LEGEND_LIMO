# 1) ROS Melodic 최소(base) 이미지 (arm64, Ubuntu 18.04)
FROM ros:melodic-ros-base-bionic

# 2) 기본 환경 설정
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Seoul \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    ROS_PYTHON_VERSION=3

# 3) 꼭 필요한 리눅스 / Python3 / OpenCV / 빌드 도구만 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3-pip python3-dev \
      python3-numpy python3-yaml python3-opencv \
      build-essential cmake git \
      libboost-all-dev \
      ros-melodic-tf \
      && rm -rf /var/lib/apt/lists/*

# 4) ROS Python3 유틸 (rospkg, catkin_pkg, catkin_tools) - pip로만
RUN pip3 install --no-cache-dir \
      rospkg catkin_pkg catkin_tools

# 5) Python3용 cv_bridge 빌드 (LKAS에서 필요)
RUN mkdir -p /opt/cvbridge_ws/src
WORKDIR /opt/cvbridge_ws/src

# cv_bridge 만 있는 vision_opencv 가져오기
RUN git clone -b melodic https://github.com/ros-perception/vision_opencv.git

WORKDIR /opt/cvbridge_ws

# catkin workspace 설정 (Python3 경로만 지정)
RUN catkin init && \
    catkin config \
      -DPYTHON_EXECUTABLE=/usr/bin/python3 \
      -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m \
      -DPYTHON_LIBRARY=/usr/lib/aarch64-linux-gnu/libpython3.6m.so \
      --install && \
    catkin build cv_bridge

# 6) 유저 ROS 워크스페이스 (호스트 마운트 위치와 맞춰서)
RUN mkdir -p /root/LEGEND_WS/drive_ws/src
WORKDIR /root/LEGEND_WS/drive_ws

# 7) 엔트리포인트 스크립트 복사
COPY entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]



