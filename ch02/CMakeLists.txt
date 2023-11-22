cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
set(CMAKE_VERBOSE_MAKEFILE ON)

project(cudaAddMatrix LANGUAGES CXX CUDA) 


add_executable(${PROJECT_NAME} "add.cu")
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)