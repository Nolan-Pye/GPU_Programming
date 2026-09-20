NVCC := nvcc
NVCC_FLAGS := -O2 -std=c++17 -arch=native
TARGET := assignment.exe
SRC := assignment.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET) results.csv
