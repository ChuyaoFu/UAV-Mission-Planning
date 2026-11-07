# Multiarchitecture ROS1 Melodic/Bionic, original PL-VINS OpenCV/Ceres API era.
FROM ros:melodic-ros-base-bionic@sha256:b4366ca0536e2ad3654ed66fb4475bc43b7a0c15bd8a5be91fd5ac9dfaf935d0
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
COPY container/ros.asc /etc/apt/trusted.gpg.d/ros.asc
RUN rm -f /etc/apt/sources.list.d/ros1-latest.list /etc/apt/sources.list.d/ros-latest.list && \
    printf '%s\n' 'deb [signed-by=/etc/apt/trusted.gpg.d/ros.asc] http://packages.ros.org/ros/ubuntu bionic main' > /etc/apt/sources.list.d/ros1-candidate.list && \
    apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git ca-certificates libeigen3-dev libceres-dev \
      libopencv-dev libboost-filesystem-dev libboost-program-options-dev \
      libboost-system-dev ros-melodic-cv-bridge ros-melodic-image-transport \
      ros-melodic-tf ros-melodic-nav-msgs ros-melodic-visualization-msgs \
      ros-melodic-rosbag python-psutil python-yaml && \
    dpkg-query -W > /opt/build-packages.txt && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/plvins_ws
COPY upstream.tar.gz /tmp/upstream.tar.gz
RUN mkdir -p src/PL-VINS && tar -xzf /tmp/upstream.tar.gz -C src/PL-VINS && rm /tmp/upstream.tar.gz
COPY patches /tmp/plvins-patches
RUN cd src/PL-VINS && git apply /tmp/plvins-patches/0001-portable-source-build-and-exact-exposure.patch
# Build only the sensor frontends and estimator, never simulation/GT publishers.
ARG BUILD_JOBS=2
RUN source /opt/ros/melodic/setup.bash && \
    catkin_make -j${BUILD_JOBS} -l${BUILD_JOBS} \
      -DCATKIN_WHITELIST_PACKAGES='camera_model;feature_tracker;plvins_estimator' \
      -DCMAKE_BUILD_TYPE=Release && \
    ldd devel/lib/feature_tracker/LineFeature_tracker && \
    test -x devel/lib/plvins_estimator/plvins_estimator
COPY standalone /opt/candidate-standalone
COPY tests/native_smoke.py /opt/native_smoke.py
RUN mkdir -p /opt/lsd-build && cd /opt/lsd-build && \
    cmake /opt/candidate-standalone -DPLVINS_SOURCE=/opt/plvins_ws/src/PL-VINS && \
    cmake --build . -- -j2 && \
    python2 /opt/native_smoke.py /opt/lsd-build/plvins_lsd_probe
COPY container/replay.launch /opt/plvins_ws/replay.launch
COPY container/run_replay.sh /opt/run_replay.sh
COPY container/entrypoint.sh /candidate-entrypoint.sh
COPY tools /opt/candidate-tools
ENTRYPOINT ["/candidate-entrypoint.sh"]
CMD ["bash"]
