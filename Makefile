CC = gcc
CFLAGS = -g3 -std=c99 -pedantic -Wall

all: ResNet BuildShards

ResNet: resnet.cu
	nvcc -g -G resnet.cu -lcurand -o ResNet

BuildShards: build_training_shards.c
	${CC} ${CFLAGS} -o $@ $^