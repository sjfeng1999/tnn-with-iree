

set(_TRANSLATE_TOOL_EXECUTABLE $<TARGET_FILE:tnn_translate>)

set(_ARGS)
list(APPEND _ARGS "-iree-input-type=mhlo")
list(APPEND _ARGS "-iree-mlir-to-vm-bytecode-module")
list(APPEND _ARGS "-iree-hal-target-backends=vmvx")
list(APPEND _ARGS "${PROJECT_SOURCE_DIR}/model/tnn_iree_mnist.mlir")
list(APPEND _ARGS "-o")
list(APPEND _ARGS "${PROJECT_SOURCE_DIR}/model/tnn_iree_mnist.vmfb")
add_custom_target(
  tnn_module_vmfb
  COMMAND ${_TRANSLATE_TOOL_EXECUTABLE} ${_ARGS}
  BYPRODUCTS "tnn_iree_mnist.vmfb"
  DEPENDS tnn_translate
)



add_library(tnn_module STATIC "")
target_sources(tnn_module
  PRIVATE
    "tnn_module.h"
    "tnn_module.cpp"
)
target_include_directories(tnn_module
  PUBLIC
    ${IREE_SOURCE_DIR}
)
target_link_libraries(tnn_module
  iree_base_base
  iree_base_cc
  iree_hal_hal
  iree_modules_hal_hal
  iree_vm_vm
  iree_vm_cc
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
