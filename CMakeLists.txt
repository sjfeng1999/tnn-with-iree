cmake_minimum_required(VERSION 3.16)
project(tnn_iree_runtime LANGUAGES CXX C)


set_property(GLOBAL PROPERTY USE_FOLDERS ON)

list(APPEND CMAKE_MODULE_PATH 
  "${PROJECT_SOURCE_DIR}/cmake"
  "${PROJECT_BINARY_DIR}/lib/cmake/mlir"
)
include(mlir_tblgen)
include(iree_compile)

set(TNN_ROOT_DIR "/home/sjfeng1999/Develop/TNN")
set(TNN_INCLUDE_DIR 
  "${TNN_ROOT_DIR}/include/"
  "${TNN_ROOT_DIR}/source/"
)
message(STATUS " >>>> ${TNN_INCLUDE_DIR}")
set(TNN_SHARED_LIB_PATH "${TNN_ROOT_DIR}/build/libTNN.so")

add_subdirectory(third_party/iree EXCLUDE_FROM_ALL)

add_subdirectory(Dialect)
add_subdirectory(runtime)





