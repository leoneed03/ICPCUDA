cmake_minimum_required(VERSION 3.10.0)

project(ICP)

set(FOR_EIGEN_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/third-party/Eigen")
include_directories(${FOR_EIGEN_INCLUDE_DIRS})

if(NOT Pangolin_LOCAL_INSTALL_DIR)
    set(Pangolin_LOCAL_INSTALL_DIR installed_Pangolin)
endif()

if(NOT Pangolin_BUILD_NAME)
    set(Pangolin_BUILD_NAME build)
endif()

set(Pangolin_TO_BUILD_DIR ${PROJECT_SOURCE_DIR}/third-party/Pangolin)

include(ExternalProject)

ExternalProject_Add(pangolin
    SOURCE_DIR "${Pangolin_TO_BUILD_DIR}"
    BINARY_DIR "${Pangolin_TO_BUILD_DIR}/${Pangolin_BUILD_NAME}"
    CMAKE_ARGS
    -DBUILD_TESTS=OFF
    -DBUILD_EXAMPLES=OFF
    -DBUILD_TOOLS=OFF
    -DCMAKE_INSTALL_PREFIX=${PROJECT_SOURCE_DIR}/${Pangolin_BUILD_NAME}/${Pangolin_LOCAL_INSTALL_DIR}
    BUILD_COMMAND $(MAKE) -j$(npoc)
    )

set(Pangolin_INSTALL_DIR "${PROJECT_SOURCE_DIR}/${Pangolin_BUILD_NAME}/${Pangolin_LOCAL_INSTALL_DIR}" CACHE PATH "Pangolin install directory")
set(Sophus_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/third-party/Sophus" CACHE PATH "Sophus Include directory")
set(Pangolin_INCLUDE_DIRS "${Pangolin_INSTALL_DIR}/include")
set(Pangolin_LIBRARIES ${Pangolin_INSTALL_DIR}/lib/libpangolin.so)

find_package(CUDA REQUIRED)

file(GLOB srcs ICPOdometry.cpp ICPOdometry.h)
file(GLOB cuda Cuda/*.cu Cuda/*.cpp Cuda/*.cuh Cuda/*.h)
file(GLOB containers Cuda/containers/*.cpp Cuda/containers/*.h Cuda/containers/*.cu Cuda/containers/*.cuh)

set(CUDA_ARCH_BIN "" CACHE STRING "Specify 'real' GPU arch to build binaries for, BIN(PTX) format is supported. Example: 1.3 2.1(1.3) or 13 21(13)")
set(CUDA_ARCH_PTX "" CACHE STRING "Specify 'virtual' PTX arch to build PTX intermediate code for. Example: 1.0 1.2 or 10 12")

include("CudaDetect.cmake")
detect_installed_gpus(CUDA_NVCC_ARCHS)

foreach(NVCC_ARCH IN LISTS CUDA_NVCC_ARCHS)
    list(APPEND CUDA_ARCH_BIN "${NVCC_ARCH} ")
endforeach(NVCC_ARCH)

include("CudaComputeTargetFlags.cmake")
APPEND_TARGET_ARCH_FLAGS()

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler;-fPIC;--expt-relaxed-constexpr")

CUDA_COMPILE(cuda_objs ${cuda})

set(CMAKE_CXX_FLAGS "-O3 -msse2 -msse3 -std=c++11")

add_library(ICP
    SHARED
    ${srcs}
    ${cuda}
    ${cuda_objs}
    ${containers}
    )

add_dependencies(ICP pangolin)

target_link_libraries(ICP
    ${Pangolin_LIBRARIES}
    ${CUDA_LIBRARIES}
    )

target_include_directories(ICP
    PUBLIC
    ${Pangolin_INCLUDE_DIRS}
    ${Sophus_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    )

unset(Pangolin_LOCAL_INSTALL_DIR CACHE)
unset(Pangolin_BUILD_NAME CACHE)
