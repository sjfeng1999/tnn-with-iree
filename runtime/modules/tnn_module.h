//
//
//
//

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/vm/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

class iree_tnn_blob_t;

iree_status_t iree_tnn_module_register_types(void);

iree_status_t iree_tnn_native_module_create(iree_allocator_t host_allocator, iree_hal_allocator_t* device_allocator,
                                            iree_vm_module_t** out_module);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

IREE_VM_DECLARE_TYPE_ADAPTERS(iree_tnn_blob, iree_tnn_blob_t);
