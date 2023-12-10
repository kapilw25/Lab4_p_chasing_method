# Makefile for compiling CUDA program

# Specify the compiler
NVCC=nvcc

# Specify the flags for the compiler
NVCC_FLAGS=-O2

# Specify the target executable name
TARGET=cache_line_profiler

# Source files
SOURCES=cache_line_profiler_3.cu

# Build target
all: $(TARGET)

$(TARGET): $(SOURCES)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SOURCES)

# Command to run the program
run:
	./$(TARGET)

# Command to generate output file
file:
	./$(TARGET) > output.txt

# Clean up the build
clean:
	rm -f $(TARGET)
