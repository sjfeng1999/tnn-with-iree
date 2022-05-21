
add_library(tnn_module STATIC "")
target_sources(tnn_module
  PRIVATE
    "tnn_module.h"
    "tnn_module.cpp"
)
target_include_directories(tnn_module
  PUBLIC
    ${IREE_SOURCE_DIR}
    ${TNN_INCLUDE_DIR}
)
target_link_libraries(tnn_module
  ${TNN_SHARED_LIB_PATH}
  iree_base_base
  iree_base_cc
  iree_hal_hal
  iree_modules_hal_hal
  iree_vm_vm
  iree_vm_cc
)
