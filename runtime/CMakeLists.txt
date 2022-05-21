
add_subdirectory(modules)

iree_compile_vmfb(
  TARGET
    tnn_module_vmfb
  TRANSLATE_TOOL
    tnn_translate
  FLAGS 
    "-iree-input-type=mhlo"
    "-iree-mlir-to-vm-bytecode-module"
    "-iree-hal-target-backends=vmvx"
    "${PROJECT_SOURCE_DIR}/model/tnn_iree_mnist.mlir"
    "-o" "${PROJECT_SOURCE_DIR}/model/tnn_iree_mnist.vmfb"
  DEPS
    tnn_translate
)


add_executable(tnn_modules_runtime "")
target_sources(tnn_modules_runtime
  PRIVATE
    tnn_iree_runtime.cpp
)
add_dependencies(tnn_modules_runtime tnn_module_vmfb)
set_target_properties(tnn_modules_runtime PROPERTIES OUTPUT_NAME tnn_modules_runtime)

target_include_directories(tnn_modules_runtime
  PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${IREE_SOURCE_DIR}
)
target_link_libraries(tnn_modules_runtime
  tnn_module
  iree_base_base
  iree_base_logging
  iree_hal_hal
  iree_hal_vmvx_registration_registration
  iree_modules_hal_hal
  iree_testing_gtest
  iree_testing_gtest_main
  iree_vm_vm
  iree_vm_bytecode_module
  iree_vm_cc
  iree_runtime_runtime
  iree_tools_utils_image_util
)
