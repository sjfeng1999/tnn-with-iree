//
// import tnn-module func
//
vm.module @tnn {

vm.import @cast_tensor_to_blob(
  %tensor: !vm.ref<!hal.buffer_view>
) -> !vm.ref<!tnn.blob>
attributes {nosideeffects}

vm.import @cast_blob_to_tensor(
  %tensor: !vm.ref<!tnn.blob>
) -> !vm.ref<!hal.buffer_view>
attributes {nosideeffects}

vm.import @relu(
  %message : !vm.ref<!tnn.blob>
) -> !vm.ref<!tnn.blob>
attributes {nosideeffects}

}  // vm.module
