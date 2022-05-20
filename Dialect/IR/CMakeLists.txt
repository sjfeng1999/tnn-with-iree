
add_library(tnn_dialect STATIC "")
target_sources(tnn_dialect
  PRIVATE
    "TnnDialect.h"
    "TnnDialect.cpp"
    "TnnOps.h.inc"
    "TnnOps.cc.inc"
    "ConversionPatterns.h"
    "ConversionPatterns.cpp"
)
target_include_directories(tnn_dialect
  PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${IREE_SOURCE_DIR}
    ${IREE_SOURCE_DIR}/third_party/llvm-project/llvm/include
    ${IREE_SOURCE_DIR}/third_party/llvm-project/mlir/include
    ${IREE_BINARY_DIR}
    ${IREE_BINARY_DIR}/third_party/llvm-project/llvm/include
    ${IREE_BINARY_DIR}/third_party/llvm-project/llvm/tools/mlir/include
)
target_link_libraries(tnn_dialect
  tnn_imports
  LLVMSupport
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRTransforms
  iree_compiler_Dialect_HAL_Conversion_Conversion
  iree_compiler_Dialect_HAL_IR_IR
  iree_compiler_Dialect_Util_IR_IR
  iree_compiler_Dialect_VM_Conversion_Conversion
)
target_compile_options(tnn_dialect PRIVATE "-fno-rtti")


mlir_tblgen_library(
  TARGET 
    TnnOpsGenTarget
  TD_FILES 
    TnnOps.td 
  OUTS 
    -gen-op-decls TnnOps.h.inc
    -gen-op-defs TnnOps.cc.inc
)


set(_ARGS)
list(APPEND _ARGS "--output_header=Tnn.imports.h")
list(APPEND _ARGS "--output_impl=Tnn.imports.c")
list(APPEND _ARGS "--identifier=Tnn_imports")
list(APPEND _ARGS "--flatten")
list(APPEND _ARGS "${CMAKE_CURRENT_SOURCE_DIR}/tnn.imports.mlir")
add_custom_command(
  OUTPUT
    "Tnn.imports.h"
    "Tnn.imports.c"
  COMMAND generate_embed_data ${_ARGS}
  DEPENDS generate_embed_data tnn.imports.mlir
)

add_library(tnn_imports STATIC "")
target_sources(tnn_imports
  PRIVATE
    Tnn.imports.c
    Tnn.imports.h
)


