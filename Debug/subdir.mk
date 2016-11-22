################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../call_cudafphi.cu \
../compute_F.cu \
../compute_h2.cu \
../compute_pval.cu \
../cudafphi.cu \
../cudafphi_device_functions.cu 

CC_SRCS += \
../cuda_fphi_main.cc 

CU_DEPS += \
./call_cudafphi.d \
./compute_F.d \
./compute_h2.d \
./compute_pval.d \
./cudafphi.d \
./cudafphi_device_functions.d 

OBJS += \
./call_cudafphi.o \
./compute_F.o \
./compute_h2.o \
./compute_pval.o \
./cuda_fphi_main.o \
./cudafphi.o \
./cudafphi_device_functions.o 

CC_DEPS += \
./cuda_fphi_main.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -G -g -O3 --use_fast_math -std=c++11 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -G -g -O3 --use_fast_math -std=c++11 --compile --relocatable-device-code=false -gencode arch=compute_35,code=compute_35 -gencode arch=compute_37,code=compute_37 -gencode arch=compute_50,code=compute_50 -gencode arch=compute_52,code=compute_52 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

%.o: ../%.cc
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -G -g -O3 --use_fast_math -std=c++11 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -G -g -O3 --use_fast_math -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


