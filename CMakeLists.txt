cmake_minimum_required(VERSION 3.16)
project(tnn_iree_runtime LANGUAGES CXX C)


set_property(GLOBAL PROPERTY USE_FOLDERS ON)

list(APPEND CMAKE_MODULE_PATH 
  "${PROJECT_SOURCE_DIR}/cmake"
  "${PROJECT_BINARY_DIR}/lib/cmake/mlir"
)
include(mlir_tblgen)
include(iree_compile)

add_subdirectory(third_party/iree EXCLUDE_FROM_ALL)

add_subdirectory(dialect)
add_subdirectory(runtime)





