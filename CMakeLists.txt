cmake_minimum_required(VERSION 3.20)
project(cpp_feedforward_net CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" HAS_MARCH_NATIVE)
if(HAS_MARCH_NATIVE)
    add_compile_options(-march=native)
endif()

# header-only library: everything lives under the repo root, Eigen is vendored in Eigen/
add_library(ffnn INTERFACE)
target_include_directories(ffnn INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

find_package(OpenMP QUIET)

function(add_ffnn_executable name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE ffnn)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${name} PRIVATE OpenMP::OpenMP_CXX)
    endif()
endfunction()

add_ffnn_executable(demo main.cpp)
add_ffnn_executable(mnist_train MNIST/MNIST.cpp)
add_ffnn_executable(mnist_transform MNIST/Transform_Data_To_Binary.cpp)
add_ffnn_executable(spiral_demo spiral/spiral.cpp)
add_ffnn_executable(ttt_interface tictactoe/interface.cpp)
# tictactoe/train.cpp is an incomplete stub (get_training_data() is called with no agents) and
# is intentionally not built here

# executables that read relative paths like "MNIST/bin/test.dat" must be run from the repo root

enable_testing()
foreach(test_name test_gradients test_serialization test_batcher)
    add_executable(${test_name} tests/${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE ffnn)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
