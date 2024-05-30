cmake_minimum_required(VERSION 3.23)
project(TPS_CPP)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)

# Find OpenCV package in the system
find_package(OpenCV REQUIRED)

# Include the OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Add executable target with source files
add_executable(TPS_CPP main.cpp)

# Link OpenCV libraries to your project
target_link_libraries(TPS_CPP ${OpenCV_LIBS})
