################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/cuddha.cu 

CU_DEPS += \
./src/cuddha.d 

OBJS += \
./src/cuddha.o 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-5.5/bin/nvcc -g -O0 -w -gencode arch=compute_30,code=sm_30 -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-5.5/bin/nvcc --compile -O0 -g -gencode arch=compute_30,code=compute_30 -gencode arch=compute_30,code=sm_30 -w  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


