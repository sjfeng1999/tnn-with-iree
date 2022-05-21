//
// import tnn-module func
//
vm.module @tnn {

vm.import @cast_buffer_to_blob(%buffer: !vm.ref<!hal.buffer_view>) -> !vm.ref<!tnn.blob>

vm.import @cast_blob_to_buffer(%blob: !vm.ref<!tnn.blob>) -> !vm.ref<!hal.buffer_view>

vm.import @relu(%input : !vm.ref<!tnn.blob>) -> !vm.ref<!tnn.blob> attributes {nosideeffects}

}  // vm.module
