NVCC := nvcc
NVCC_FLAGS := -O2 -std=c++17 -arch=native
TARGET := streams_assignment
SRCS := main.cu blur_kernels.cu image_io.cpp
HDRS := common.h blur_kernels.cuh image_io.h

all: $(TARGET)

$(TARGET): $(SRCS) $(HDRS)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SRCS)

clean:
	rm -rf $(TARGET) $(TARGET).exe *.exp *.lib output

.PHONY: all clean
