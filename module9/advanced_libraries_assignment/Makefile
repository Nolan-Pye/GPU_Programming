# Extra nvcc flags (e.g. -I/-L for a CUDA toolkit outside the default
# paths) can be passed in NVCC_EXTRA.
NVCC := nvcc
NVCC_FLAGS := -O2 -std=c++17 -arch=native $(NVCC_EXTRA)
LIBS := -lcurand
TARGET := savings_sim
SRCS := main.cu simulator.cu history_data.cpp cpu_check.cpp
HDRS := common.h simulator.cuh history_data.h cpu_check.h

all: $(TARGET)

$(TARGET): $(SRCS) $(HDRS)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SRCS) $(LIBS)

clean:
	rm -rf $(TARGET) $(TARGET).exe *.exp *.lib

.PHONY: all clean
