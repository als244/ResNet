all: ResNet

ResNet: resnet.cu
	nvcc -g -G resnet.cu -lcurand -o ResNet
