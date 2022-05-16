

#ifndef IREE_SAMPLES_Tnn_MODULES_DIALECT_Tnn_OPS
#define IREE_SAMPLES_Tnn_MODULES_DIALECT_Tnn_OPS

include "iree/compiler/Dialect/HAL/IR/HALBase.td"
include "iree/compiler/Dialect/Util/IR/UtilBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Tnn_Dialect : Dialect {
  let name = "tnn";
  let cppNamespace = "::mlir::iree_compiler::IREE::tnn";

  let summary = [{
    A custom dialect demonstrating custom ops and implementations.
  }];
}

def Tnn_Message : DialectType<
    Tnn_Dialect,
    CPred<"$_self.isa<IREE::tnn::MessageType>()">,
    "message"> {
  let description = [{
    A string message that can be printed. These types will be reference counted
    at runtime and must derive from iree_vm_ref_t but otherwise can be whatever
    the dialect wants.
  }];
}

def Tnn_Blob : DialectType<
    Tnn_Dialect,
    CPred<"$_self.isa<IREE::tnn::TnnBlob>()">,
    "TnnBlob"> {

    let summary = [{
        TNN Data Struct
    }];
}


def Tnn_TensorToMessageOp : Op<Tnn_Dialect, "tensor_to_message"> {
  let summary = [{formats tensor contents as a message}];
  let description = [{
    Formats the tensor using the IREE buffer printer to have a shape/type and
    the contents as a string.

    This demonstrates handling tensor->buffer conversion through the HAL layer.
    This op (tensor_to_message) is only used prior to translation and instances
    of it will be converted to the lower-level HAL-based buffer_to_message op.
  }];

  let arguments = (ins
    AnyTensor:$operand
  );
  let results = (outs
    Tnn_Message:$result
  );
}

def Tnn_BufferToMessageOp : Op<Tnn_Dialect, "buffer_to_message"> {
  let summary = [{formats buffer contents as a message}];
  let description = [{
    Formats the tensor using the IREE buffer printer to have a shape/type and
    the contents as a string.
  }];

  let arguments = (ins
    HAL_BufferView:$buffer_view
  );
  let results = (outs
    Tnn_Message:$result
  );
}

def Tnn_MessageToTensorOp : Op<Tnn_Dialect, "message_to_tensor"> {
  let summary = [{parses message contents as a tensor}];
  let description = [{
    Parses the message containing a IREE buffer parser-formatted tensor.
  }];

  let arguments = (ins
    Tnn_Message:$message
  );
  let results = (outs
    AnyTensor:$result
  );
}

def Tnn_MessageToBufferOp : Op<Tnn_Dialect, "message_to_buffer"> {
  let summary = [{parses message contents as a buffer}];
  let description = [{
    Parses the message containing a IREE buffer parser-formatted tensor.
  }];

  let arguments = (ins
    Tnn_Message:$message
  );
  let results = (outs
    HAL_BufferView:$result
  );
}

def Tnn_PrintOp : Op<Tnn_Dialect, "print"> {
  let summary = [{prints a message zero or more times}];
  let description = [{
    Prints the input message zero or more times with a newline inbetween.

    This demonstrates passing arguments from the application (the original
    message).
  }];

  let arguments = (ins
    Tnn_Message:$message,
    I32:$count
  );
}

def Tnn_ReverseOp : Op<Tnn_Dialect, "reverse", [NoSideEffect]> {
  let summary = [{reverses the characters in the given the message}];
  let description = [{
    Reverses the message characters and returns the new value.

    This demonstrates allocating new values from within the op implementation
    (the reversed result). Note that this op (as well as the vm.import) are
    marked with NoSideEffect/nosideeffect to enable additional compiler
    optimization.
  }];

  let arguments = (ins
    Tnn_Message:$operand
  );
  let results = (outs
    Tnn_Message:$result
  );
}

def Tnn_GetUniqueMessageOp : Op<Tnn_Dialect, "get_unique_message", [NoSideEffect]> {
  let summary = [{returns a per-context unique message}];
  let description = [{
    Returns a unique message allocated per context using the module.

    This demonstrates context-specific global storage. Mutation methods could
    exist that update the global storage as desired.
  }];

  let results = (outs
    Tnn_Message:$result
  );
}

def Tnn_ReluOp : Op<Tnn_Dialect, "relu"> {
    let summary = "Tensor Relu Activation";

    let arguments = (ins 
        Tnn_Message: $input
    );
    let results = (outs
        Tnn_Blob: $output
    );
}


#endif  // IREE_SAMPLES_Tnn_MODULES_DIALECT_Tnn_OPS