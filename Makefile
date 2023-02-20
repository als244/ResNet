CC = gcc
CFLAGS = -g3 -std=c99 -pedantic -Wall

all: ResNet ResNetOpt

ResNet: resnet.cu
	nvcc -g -G resnet.cu -lcurand -o ResNet

ResNetOpt: resnet.cu
	nvcc -O2 resnet.cu -lcurand -o ResNetOpt

BuildShards: build_training_shards.c
	${CC} ${CFLAGS} -o $@ $^