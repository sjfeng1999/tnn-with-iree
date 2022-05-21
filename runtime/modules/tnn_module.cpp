//
//
//

#include <stdio.h>
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

#include "tnn/core/blob.h"
#include "tnn/core/blob_impl.h"
#include "tnn/utils/dims_vector_utils.h"

#include "tnn_module.h"
//#include "runtime/modules/tnn_utils.h" 


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct iree_tnn_blob_t {
public:
    iree_vm_ref_object_t ref_object;
    iree_allocator_t allocator;
    TNN_NS::BlobImpl* blob_impl_ptr;
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
public:
    using BlobImpl = ::TNN_NS::BlobImpl;
    using BlobDesc = ::TNN_NS::BlobDesc;
    using BlobHandle = ::TNN_NS::BlobHandle;

private:
    iree_allocator_t host_allocator_ = iree_allocator_system();
    vm::ref<iree_hal_allocator_t> device_allocator_;

public:
    explicit TnnModuleState(iree_allocator_t host_allocator, vm::ref<iree_hal_allocator_t> device_allocator)
        : host_allocator_(host_allocator), device_allocator_(std::move(device_allocator)) {}

    ~TnnModuleState() = default;

    Status Initialize() {
        printf("  -> tnn module state init\n");
        return iree_ok_status();
    }

    StatusOr<vm::ref<iree_tnn_blob_t>> CastBufferToBlob(vm::ref<iree_hal_buffer_view_t> buffer_view) {
        vm::ref<iree_tnn_blob_t> out_blob;

        printf("  -> 1 use tnn cast tensor to blob op\n");

        iree_tnn_blob_t* blob_ptr;
        void* handle_base_ptr = nullptr;
        iree_host_size_t rank = iree_hal_buffer_view_shape_rank(buffer_view.get());
        iree_device_size_t bytes_offset = sizeof(float) * iree_hal_buffer_view_element_count(buffer_view.get());

        iree_allocator_malloc(host_allocator_, sizeof(iree_tnn_blob_t), (void**)&blob_ptr);
        iree_allocator_malloc(iree_hal_allocator_host_allocator(device_allocator_.get()), bytes_offset, (void**)&handle_base_ptr);

        iree_hal_buffer_map_read(iree_hal_buffer_view_buffer(buffer_view.get()), 0, handle_base_ptr, bytes_offset);

        std::vector<int> dims;
        for (int i = 0; i < rank; ++i){
            dims.push_back(iree_hal_buffer_view_shape_dim(buffer_view.get(), i));
        }

        BlobDesc desc = {.dims = std::move(dims)};
        BlobHandle handle = {.base = handle_base_ptr, 
                             .bytes_offset = bytes_offset};

        blob_ptr->ref_object.counter = IREE_ATOMIC_VAR_INIT(1);
        blob_ptr->allocator = host_allocator_;
        blob_ptr->blob_impl_ptr = new BlobImpl(desc, handle);

        out_blob = blob_ptr;
        printf("  -> 2 use tnn cast tensor to blob op\n");
        return std::move(out_blob);
    }

    StatusOr<vm::ref<iree_hal_buffer_view_t>> CastBlobToBuffer(vm::ref<iree_tnn_blob_t> blob) {
        vm::ref<iree_hal_buffer_view_t> out_buffer_view;
        printf("  -> 1 use tnn cast blob to tensor op\n");

        BlobDesc desc = blob->blob_impl_ptr->GetBlobDesc();
        BlobHandle handle = blob->blob_impl_ptr->GetHandle();

        iree_hal_dim_t* shape = desc.dims.data();
        iree_host_size_t shape_rank = desc.dims.size();
        iree_device_size_t allocation_size = sizeof(float) * ::TNN_NS::DimsVectorUtils::Count(desc.dims);
        iree_hal_element_type_t element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_32;
        iree_hal_encoding_type_t encoding_type = IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR;

        iree_hal_buffer_t* buffer = nullptr;
        iree_hal_buffer_params_t* buffer_param;
        iree_allocator_malloc(iree_hal_allocator_host_allocator(device_allocator_.get()), sizeof(iree_hal_buffer_params_t), (void**)&buffer_param);
        iree_hal_buffer_params_canonicalize(buffer_param);
        buffer_param->type = IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
        // inplace relu
        iree_hal_allocator_allocate_buffer(device_allocator_.get(), *buffer_param, allocation_size,
                                           iree_make_const_byte_span(handle.base, handle.bytes_offset), &buffer);
        iree_hal_buffer_view_create(buffer, shape, shape_rank, element_type, encoding_type, 
                                    iree_hal_allocator_host_allocator(device_allocator_.get()), &out_buffer_view);

        printf("  -> 2 use tnn cast blob to tensor op\n");
        return std::move(out_buffer_view);
    }

    StatusOr<vm::ref<iree_tnn_blob_t>> Relu(vm::ref<iree_tnn_blob_t> blob) {
        vm::ref<iree_tnn_blob_t> out_blob;

        printf("  -> 1 use tnn relu op\n");
        out_blob = std::move(blob);
        return std::move(out_blob);
    }
};

static const vm::NativeFunction<TnnModuleState> kTnnModuleFunctions[] = {
    vm::MakeNativeFunction("cast_blob_to_buffer", &TnnModuleState::CastBlobToBuffer),
    vm::MakeNativeFunction("cast_buffer_to_blob", &TnnModuleState::CastBufferToBlob),
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