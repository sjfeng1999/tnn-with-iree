//
//
//

#include "tnn_module.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

#include "iree/base/api.h"
#include "iree/base/status_cc.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/module.h"
#include "iree/vm/native_module_cc.h"
#include "iree/vm/ref_cc.h"
#include "stdio.h"

// #include "TNN/someHeaderfile"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct iree_tnn_blob_t {
public:
    iree_vm_ref_object_t ref_object;
    iree_allocator_t allocator;
    iree_string_view_t value;
};

static iree_vm_ref_type_descriptor_t iree_tnn_blob_descriptor = {0};

IREE_VM_DEFINE_TYPE_ADAPTERS(iree_tnn_blob, iree_tnn_blob_t);

void iree_tnn_blob_destroy(void* ptr) {
    iree_tnn_blob_t* message = (iree_tnn_blob_t*)ptr;
    iree_allocator_free(message->allocator, ptr);
}

iree_status_t iree_tnn_module_register_types() {
    if (iree_tnn_blob_descriptor.type) {
        return iree_ok_status();
    }
    iree_tnn_blob_descriptor.type_name        = iree_make_cstring_view("tnn.blob");
    iree_tnn_blob_descriptor.destroy          = iree_tnn_blob_destroy;
    iree_tnn_blob_descriptor.offsetof_counter = offsetof(iree_tnn_blob_t, ref_object.counter);

    return iree_vm_ref_register_type(&iree_tnn_blob_descriptor);
}

namespace iree {
namespace tnn {

class TnnModuleState final {
private:
    iree_allocator_t host_allocator_ = iree_allocator_system();
    vm::ref<iree_hal_allocator_t> device_allocator_;
    // some tnn global var

public:
    explicit TnnModuleState(iree_allocator_t host_allocator, vm::ref<iree_hal_allocator_t> device_allocator)
        : host_allocator_(host_allocator), device_allocator_(std::move(device_allocator)) {}

    ~TnnModuleState() = default;

    Status Initialize() {
        printf("  -> tnn module state init");
        return iree_ok_status();
    }

    StatusOr<vm::ref<iree_tnn_blob_t>> CastTensorToBlob(vm::ref<iree_hal_buffer_view_t> tensor) {
        vm::ref<iree_tnn_blob_t> out_blob;

        printf("  -> use tnn cast tensor to blob op");

        return std::move(out_blob);
    }

    StatusOr<vm::ref<iree_hal_buffer_view_t>> CastBlobToTensor(vm::ref<iree_tnn_blob_t> blob) {
        vm::ref<iree_hal_buffer_view_t> out_buffer;

        printf("  -> use tnn cast blob to tensor op");

        return std::move(out_buffer);
    }

    StatusOr<vm::ref<iree_tnn_blob_t>> Relu(vm::ref<iree_tnn_blob_t> blob) {
        vm::ref<iree_tnn_blob_t> out_blob;

        printf("  -> use tnn relu op");
        out_blob = std::move(blob);

        return std::move(out_blob);
    }
};

static const vm::NativeFunction<TnnModuleState> kTnnModuleFunctions[] = {
    vm::MakeNativeFunction("cast_blob_to_buffer", &TnnModuleState::CastBlobToTensor),
    vm::MakeNativeFunction("cast_buffer_to_blob", &TnnModuleState::CastTensorToBlob),
    vm::MakeNativeFunction("relu", &TnnModuleState::Relu),
};

class TnnModule final : public vm::NativeModule<TnnModuleState> {
private:
    vm::ref<iree_hal_allocator_t> device_allocator_;

public:
    using vm::NativeModule<TnnModuleState>::NativeModule;

    Status Initialize(iree_hal_allocator_t* device_allocator) {
        device_allocator_ = vm::retain_ref(device_allocator);
        return OkStatus();
    }

    StatusOr<std::unique_ptr<TnnModuleState>> CreateState(iree_allocator_t host_allocator) override {
        auto state = std::make_unique<TnnModuleState>(host_allocator, vm::retain_ref(device_allocator_));
        IREE_RETURN_IF_ERROR(state->Initialize());
        return state;
    }
};


extern "C" iree_status_t iree_tnn_native_module_create(iree_allocator_t host_allocator,
                                                       iree_hal_allocator_t* device_allocator,
                                                       iree_vm_module_t** out_module) {
    IREE_ASSERT_ARGUMENT(out_module);
    *out_module = NULL;
    auto module = std::make_unique<TnnModule>(
        "tnn", host_allocator, iree::span<const vm::NativeFunction<TnnModuleState>>(kTnnModuleFunctions));

    IREE_RETURN_IF_ERROR(module->Initialize(device_allocator));
    *out_module = module.release()->interface();
    return iree_ok_status();
}

}  // namespace tnn
}  // namespace IREE