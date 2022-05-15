
set(MLIR_TBLGEN_INCLUDE_DIRS ${PROJECT_SOURCE_DIR})

list(APPEND MLIR_TBLGEN_INCLUDE_DIRS
  ${LLVM_MLIR_ROOT_DIR}/mlir/include
)
list(TRANSFORM MLIR_TBLGEN_INCLUDE_DIRS PREPEND "-I")


set(LLVM_TARGET_DEFINITIONS "TnnDialect.td")
tablegen("MLIR" TnnDialect.h.inc  "-gen-dialect-decls" ${MLIR_TBLGEN_INCLUDE_DIRS})
tablegen("MLIR" TnnDialect.cc.inc "-gen-dialect-defs" ${MLIR_TBLGEN_INCLUDE_DIRS})
add_custom_target(TnnDialectInsGen DEPENDS TnnDialect.h.inc TnnDialect.cc.inc)


set(LLVM_TARGET_DEFINITIONS "TnnOps.td")
tablegen("MLIR" TnnOps.h.inc  "-gen-op-decls" ${MLIR_TBLGEN_INCLUDE_DIRS})
tablegen("MLIR" TnnOps.cc.inc "-gen-op-defs" ${MLIR_TBLGEN_INCLUDE_DIRS})
add_custom_target(TnnOpsInsGen DEPENDS TnnOps.h.inc TnnOps.cc.inc)
add_dependencies(TnnOpsInsGen TnnDialectInsGen)

message(" > mlir tablegen finished")
