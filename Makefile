CXX = g++

# Compiler flags
CXXFLAGS = -std=c++17 -Wall -Wextra

# Include directories (use -I to add new directories)
INCLUDES = -I./include \
	-I/opt/homebrew/Cellar/boost/1.83.0/include \
	-I/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3

# Source directory and source files
SRC_DIR = ./src
SOURCES = $(wildcard $(SRC_DIR)/*.cpp)

# Object files (corresponding to source files)
OBJ_DIR = ./
OBJECTS = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SOURCES))

# Libraries (use -L for library paths and -l for specific libraries)
LIBS = -L./src/lib # -lyour_library

# Output binary
TARGET = main

# Default target
all: $(TARGET)

# Linking the target with object files
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LIBS)

# Compiling source files into object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean up
clean:
	rm -f $(OBJ_DIR)/*.o $(TARGET)

# Phony targets
.PHONY: all clean
