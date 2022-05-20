//
// import tnn-module func
//
vm.module @tnn {

vm.import @cast_tensor_to_blob(
  %tensor: !vm.ref<!hal.buffer_view>
) -> !vm.ref<!tnn.TnnBlob>
attributes {nosideeffects}

vm.import @cast_blob_to_tensor(
  %tensor: !vm.ref<!tnn.TnnBlob>
) -> !vm.ref<!hal.buffer_view>
attributes {nosideeffects}

vm.import @relu(
  %message : !vm.ref<!tnn.TnnBlob>
) -> !vm.ref<!tnn.TnnBlob>
attributes {nosideeffects}

}  // vm.module
