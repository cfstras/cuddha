################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../src/bmp.c 

CU_SRCS += \
../src/cuddha.cu 

CU_DEPS += \
./src/cuddha.d 

OBJS += \
./src/bmp.o \
./src/cuddha.o 

C_DEPS += \
./src/bmp.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-5.5/bin/nvcc -O3 -m64 -w -Xcompiler -Wall -gencode arch=compute_30,code=sm_30 -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-5.5/bin/nvcc -O3 -m64 -w -Xcompiler -Wall --compile  -x c -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-5.5/bin/nvcc -O3 -m64 -w -Xcompiler -Wall -gencode arch=compute_30,code=sm_30 -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-5.5/bin/nvcc --compile -O3 -Xcompiler -Wall -gencode arch=compute_30,code=compute_30 -gencode arch=compute_30,code=sm_30 -m64 -w  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


