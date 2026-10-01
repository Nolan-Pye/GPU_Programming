NVCC := nvcc
NVCC_FLAGS := -O2 -std=c++17 -arch=native
TARGET := memory_assignment
SRCS := main.cu histogram_kernels.cu text_utils.cpp
HDRS := common.h histogram_kernels.cuh text_utils.h

all: $(TARGET)

$(TARGET): $(SRCS) $(HDRS)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SRCS)

clean:
	rm -f $(TARGET) $(TARGET).exe *.exp *.lib

.PHONY: all clean
