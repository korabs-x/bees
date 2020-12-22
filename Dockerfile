FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04 as builder
LABEL maintainer="simon@siboehm.com"

ARG NUMCORES=12

SHELL ["/bin/bash", "-c"]

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
RUN apt-get update && apt-get install -y \
  cmake \
  g++ \
  wget \
  unzip \
  git \
  pkg-config

WORKDIR /home/yolo
RUN wget -qO opencv.zip https://github.com/opencv/opencv/archive/master.zip &&\
  unzip -q opencv.zip &&\
  rm opencv.zip

WORKDIR /home/yolo/build
RUN cmake -D OPENCV_GENERATE_PKGCONFIG=YES ../opencv-master &&\
  make -j$NUMCORES && make install

WORKDIR /home/yolo
# clone darknet and jump to fixed commit (for reproducibility)
RUN git clone "https://github.com/AlexeyAB/darknet" darknet &&\
  cd darknet &&\
  git checkout c5b8bc7f2472c32636a9c7122d78229219b84482

WORKDIR /home/yolo/darknet
RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf && ldconfig &&\
  sed -i 's:OPENCV=0:OPENCV=1:' Makefile &&\
  sed -i 's:GPU=0:GPU=1:' Makefile &&\
  sed -i 's:CUDNN=0:CUDNN=1:' Makefile &&\
  sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile &&\
  make

FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04

SHELL ["/bin/bash", "-c"]

WORKDIR /home/yolo
# pick any weights you like, but make sure the yolov4-custom.cfg
# has the correct width and height saved (I always set them equal)
# small: 224; medium: 416; large: 608
# by default width and height are set to 608 in the yolov4-custom.cfg
# and to 224 in yolov4-tiny-custom.cfg
COPY yolov4-custom.cfg yolov4-tiny-custom.cfg yolov4-custom_6000.weights yolov4-tiny-custom_28000.weights ./

COPY --from=builder /home/yolo/darknet/darknet ./
COPY --from=builder /home/yolo/darknet/data/labels ./data/labels
COPY --from=builder /usr/local/lib /usr/local/lib

RUN printf "bee\n" > obj.names &&\
  printf 'classes = 2\nnames = obj.names\n' > obj.data

RUN apt-get update && apt-get install -y python3 python3-pip

CMD ["/bin/bash", "-l"]
