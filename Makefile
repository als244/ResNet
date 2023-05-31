CC = gcc
CFLAGS = -g3 -std=c99 -pedantic -Wall

all: ResNet ResNetOpt ResNetClean ResNetCleanOpt ResNetCuDNN ResNetCuDNNOpt

ResNet: resnet.cu
	nvcc -g -G -arch=sm_80 resnet.cu -lcurand -o ResNet

ResNetOpt: resnet.cu
	nvcc -O3 -arch=sm_80 resnet.cu -lcurand -o ResNetOpt

ResNetClean: resnet_clean.cu
	nvcc -g -G -arch=sm_80 resnet_clean.cu -lcurand -o ResNetClean

ResNetCleanOpt: resnet_clean.cu
	nvcc -O3 -arch=sm_80 resnet_clean.cu -lcurand -o ResNetCleanOpt

ResNetCuDNN: resnet_cudnn.cu
	nvcc -g -G -arch=sm_80 resnet_cudnn.cu -lcurand -lcudnn -o ResNetCuDNN

ResNetCuDNNOpt: resnet_cudnn.cu
	nvcc -O3 -arch=sm_80 resnet_cudnn.cu -lcurand -lcudnn -o ResNetCuDNNOpt

BuildShards: build_training_shards.c
	${CC} ${CFLAGS} -o $@ $^

