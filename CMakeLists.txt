cmake_minimum_required(VERSION 3.16)
project(tnn_iree_runtime ASM C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)


set(LLVM_MLIR_ROOT_DIR "/home/sjfeng1999/Develop/llvm-project")
set(IREE_SOURCE_DIR "/media/sjfeng1999/Datasets/Dev/iree")
set(IREE_BINARY_DIR "/media/sjfeng1999/Datasets/Dev/iree-build")

message(STATUS " > llvm-project : ${LLVM_MLIR_ROOT_DIR}")
message(STATUS " > iree         : ${IREE_SOURCE_DIR}")
message(STATUS " > iree-build   : ${IREE_BINARY_DIR}")


set(MLIR_INCLUDE_DIR ${LLVM_MLIR_ROOT_DIR}/mlir/include)
set(MLIR_TABLEGEN_EXE ${LLVM_MLIR_ROOT_DIR}/build/bin/mlir-tblgen)


list(APPEND CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_LIST_DIR}/cmake
  ${LLVM_MLIR_ROOT_DIR}/cmake/Modules
  ${LLVM_MLIR_ROOT_DIR}/llvm/cmake/modules
  ${LLVM_MLIR_ROOT_DIR}/mlir/cmake/modules
)

include(AddLLVM)
include(AddMLIR)
include(TableGen)

include(iree_macros)


add_subdirectory(Dialect)

