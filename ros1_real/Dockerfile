FROM ros:noetic-ros-base-focal

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    python3-pip \
    python3-yaml \
    ros-noetic-mavros \
    ros-noetic-mavros-extras \
    ros-noetic-rviz \
    ros-noetic-tf \
    ros-noetic-tf2-ros \
    ros-noetic-vrpn \
    && rm -rf /var/lib/apt/lists/*

RUN /opt/ros/noetic/lib/mavros/install_geographiclib_datasets.sh

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python3 -m pip install --no-cache-dir numpy==1.24.4 scipy==1.10.1 \
    && python3 -m pip install --no-cache-dir \
       --index-url https://download.pytorch.org/whl/cpu torch==2.4.1+cpu

WORKDIR /root/catkin_ws
COPY src ./src
RUN . /opt/ros/noetic/setup.sh && catkin_make -DCATKIN_ENABLE_TESTING=OFF

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["roslaunch", "srlc_real", "real_px4.launch"]
