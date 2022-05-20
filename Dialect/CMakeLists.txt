
add_subdirectory(IR)

add_executable(tnn_translate "")
target_sources(tnn_translate 
  PRIVATE 
    tnn_translate.cpp 
    init_dialect.h
)

target_link_libraries(tnn_translate
  PUBLIC
    tnn_dialect
    iree_tools_init_compiler_modules
    iree_tools_init_iree_passes_and_dialects
    iree_tools_init_mlir_passes_and_dialects
    iree_tools_init_targets
    iree_tools_init_translations
    iree_tools_init_xla_dialects
    LLVMSupport
    MLIRIR
    MLIRPass
    MLIRSCFTransforms
    MLIRSupport
    MLIRTranslateLib
    iree_compiler_Dialect_VM_Target_Bytecode_Bytecode
    iree_compiler_Dialect_VM_Target_init_targets
    iree_compiler_Translation_IREEVM
)