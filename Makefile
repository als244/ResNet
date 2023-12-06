CC = gcc
CFLAGS = -g3 -std=c99 -pedantic -Wall

all: ResNetFast ResNetFastOpt ResNetCuDNNLowMem ResNetCuDNNLowMemOpt ResNet ResNetOpt ResNetCuDNN ResNetCuDNNOpt 

ResNet: resnet.cu
	nvcc -g -G -arch=sm_80 resnet.cu -lcurand -o ResNet

ResNetOpt: resnet.cu
	nvcc -O3 -arch=sm_80 resnet.cu -lcurand -o ResNetOpt

ResNetCuDNN: resnet_cudnn.cu
	nvcc -g -G -arch=sm_80 resnet_cudnn.cu -lcurand -lcudnn -o ResNetCuDNN

ResNetCuDNNOpt: resnet_cudnn.cu
	nvcc -O3 -arch=sm_80 resnet_cudnn.cu -lcurand -lcudnn -o ResNetCuDNNOpt

ResNetCuDNNLowMem: resnet_cudnn_lowmem.cu
	nvcc -g -G -arch=sm_80 resnet_cudnn_lowmem.cu -lcurand -lcudnn -o ResNetCuDNNLowMem

ResNetCuDNNLowMemOpt: resnet_cudnn_lowmem.cu
	nvcc -O3 -arch=sm_80 resnet_cudnn_lowmem.cu -lcurand -lcudnn -o ResNetCuDNNLowMemOpt

ResNetFast: resnet_cudnn_fast.cu
	nvcc -g -G -arch=sm_80 resnet_cudnn_fast.cu -lcurand -lcudnn -lcublas -o ResNetFast

ResNetFastOpt: resnet_cudnn_fast.cu
	nvcc -O3 -arch=sm_80 resnet_cudnn_fast.cu -lcurand -lcudnn -lcublas -o ResNetFastOpt

BuildShards: build_training_shards.c
	${CC} ${CFLAGS} -o $@ $^

