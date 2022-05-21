//
//
//
//

#include <stdio.h>

#include <type_traits>
#include <vector>

#include "iree/base/internal/file_io.h"
#include "iree/hal/vmvx/registration/driver_module.h"
#include "iree/modules/hal/module.h"
#include "iree/runtime/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/tools/utils/image_util.h"
#include "iree/vm/bytecode_module.h"
#include "iree/vm/ref_cc.h"

#include "runtime/modules/tnn_module.h"


int main(int argc, char* argv[]) {

    iree_vm_instance_t* instance_        = nullptr;
    iree_vm_context_t* context_          = nullptr;
    iree_vm_module_t* bytecode_module_   = nullptr;
    iree_vm_module_t* native_module_     = nullptr;
    iree_vm_module_t* hal_module_        = nullptr;
    iree_hal_allocator_t* hal_allocator_ = nullptr;

    iree_hal_driver_t* hal_driver = nullptr;
    iree_hal_device_t* hal_device = nullptr;

    IREE_CHECK_OK(iree_hal_vmvx_driver_module_register(iree_hal_driver_registry_default()));
    IREE_CHECK_OK(iree_vm_instance_create(iree_allocator_system(), &instance_));

    IREE_CHECK_OK(iree_hal_module_register_types());
    IREE_CHECK_OK(iree_hal_driver_registry_try_create_by_name(
        iree_hal_driver_registry_default(), iree_make_cstring_view("vmvx"), iree_allocator_system(), &hal_driver));
    IREE_CHECK_OK(iree_hal_driver_create_default_device(hal_driver, iree_allocator_system(), &hal_device));
    IREE_CHECK_OK(iree_hal_module_create(hal_device, iree_allocator_system(), &hal_module_));

    hal_allocator_ = iree_hal_device_allocator(hal_device);

    IREE_CHECK_OK(iree_tnn_module_register_types());
    IREE_CHECK_OK(iree_tnn_native_module_create(iree_allocator_system(), hal_allocator_, &native_module_));

    iree_file_contents_t* flatbuffer_contents = NULL;

    const char* module_file = {"../model/tnn_iree_mnist.vmfb"};
    printf("IREE module loaded by vmfb file\n");

    IREE_CHECK_OK(iree_file_read_contents(module_file, iree_allocator_system(), &flatbuffer_contents));
    IREE_CHECK_OK(iree_vm_bytecode_module_create(flatbuffer_contents->const_buffer,
                                                 iree_file_contents_deallocator(flatbuffer_contents),
                                                 iree_allocator_system(), &bytecode_module_));

    std::vector<iree_vm_module_t*> modules = {hal_module_, native_module_, bytecode_module_};
    IREE_CHECK_OK(iree_vm_context_create_with_modules(instance_, IREE_VM_CONTEXT_FLAG_NONE, modules.data(),
                                                      modules.size(), iree_allocator_system(), &context_));

    iree_string_view_t image_path = iree_make_cstring_view("../model/mnist_test.png");

    iree_hal_buffer_view_t* buffer_view      = NULL;
    iree_hal_dim_t buffer_shape[]            = {1, 28, 28, 1};
    iree_hal_element_type_t hal_element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_32;

    float input_range[2] = {0.0f, 1.0f};

    IREE_CHECK_OK(iree_tools_utils_buffer_view_from_image_rescaled(
        image_path, buffer_shape, IREE_ARRAYSIZE(buffer_shape), hal_element_type, iree_hal_device_allocator(hal_device),
        input_range, IREE_ARRAYSIZE(input_range), &buffer_view));

    iree::vm::ref<iree_vm_list_t> inputs;
    IREE_CHECK_OK(iree_vm_list_create(/*element_type=*/nullptr, 1, iree_allocator_system(), &inputs));

    iree_vm_ref_t input_buffer_ref = iree_hal_buffer_view_move_ref(buffer_view);
    IREE_CHECK_OK(iree_vm_list_push_ref_move(inputs.get(), &input_buffer_ref));

    iree::vm::ref<iree_vm_list_t> outputs;
    IREE_CHECK_OK(iree_vm_list_create(/*element_type=*/nullptr, 1, iree_allocator_system(), &outputs));

    iree_vm_function_t function;
    IREE_CHECK_OK(iree_vm_context_resolve_function(context_, iree_make_cstring_view("module.predict"), &function));
    IREE_CHECK_OK(iree_vm_invoke(context_, function, IREE_VM_INVOCATION_FLAG_NONE,
                                 /*policy=*/nullptr, inputs.get(), outputs.get(), iree_allocator_system()));


    iree_hal_device_release(hal_device);
    iree_hal_driver_release(hal_driver);
    // iree_vm_module_release(hal_module_);
    // iree_vm_module_release(native_module_);
    // iree_vm_module_release(bytecode_module_);
    iree_vm_context_release(context_);
    iree_vm_instance_release(instance_);
    return 0;
}