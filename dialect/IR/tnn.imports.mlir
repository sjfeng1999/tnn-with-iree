//
// Tnn IR map to CXX-Function
//
vm.module @tnn {

// Formats the tensor using the IREE buffer printer to have a shape/type and
// the contents as a string.
vm.import @buffer_to_message(
  %buffer_view : !vm.ref<!hal.buffer_view>
) -> !vm.ref<!tnn.message>
attributes {nosideeffects}

// Parses the message containing a IREE buffer parser-formatted tensor.
vm.import @message_to_buffer(
  %message : !vm.ref<!tnn.message>
) -> !vm.ref<!hal.buffer_view>
attributes {nosideeffects}

// Prints the %message provided %count times.
// Maps to the IREE::Custom::PrintOp.
vm.import @print(
  %message : !vm.ref<!tnn.message>,
  %count : i32
)

// Returns the message with its characters reversed.
// Maps to the IREE::Custom::ReverseOp.
vm.import @reverse(
  %message : !vm.ref<!tnn.message>
) -> !vm.ref<!tnn.message>
attributes {nosideeffects}

// Returns a per-context unique message.
// Maps to the IREE::Custom::GetUniqueMessageOp.
vm.import @get_unique_message() -> !vm.ref<!tnn.message>
attributes {nosideeffects}


vm.import @relu(
  %message : !vm.ref<!tnn.message>
) -> !vm.ref<!tnn.TnnBlob>
attributes {nosideeffects}

}  // vm.module