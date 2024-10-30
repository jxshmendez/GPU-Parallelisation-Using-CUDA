cmake_minimum_required(VERSION 3.18)
project(cudacw LANGUAGES CUDA CXX)

# Set the CUDA and C++ standards
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Add the CUDA executable
add_executable(cudacw main.cu)

# Enable separable compilation for CUDA
set_target_properties(cudacw PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Set the architecture for RTX 2080 (Turing - Compute Capability 7.5)
set(CMAKE_CUDA_ARCHITECTURES 75)
