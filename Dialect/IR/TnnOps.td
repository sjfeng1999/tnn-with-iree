

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


def Tnn_Blob : DialectType<
    Tnn_Dialect,
    CPred<"$_self.isa<IREE::tnn::TnnBlob>()">,
    "blob"> {

    let summary = [{
        TNN Data Struct
    }];
}



def Tnn_CastTensorToBlob : Op<Tnn_Dialect, "cast_tensor_to_blob"> {
  let summary = [{ cast build-in tensor to TNN Blob}];

  let arguments = (ins 
    AnyTensor: $tensor
  );
  let results = (outs 
    Tnn_Blob: $blob 
  );
}


def Tnn_CastBlobToTensor : Op<Tnn_Dialect, "cast_blob_to_tensor"> {
  let summary = [{ cast TNN Blob to build-in tensor}];

  let arguments = (ins 
    Tnn_Blob: $blob 
  );
  let results = (outs 
    AnyTensor: $tensor
  );
}


def Tnn_ReluOp : Op<Tnn_Dialect, "relu", [NoSideEffect]> {
  let summary = [{ Tensor Relu Activation }];

  let arguments = (ins 
    Tnn_Blob: $input
  );
  let results = (outs
    Tnn_Blob: $output
  );
} 


#endif  // IREE_SAMPLES_Tnn_MODULES_DIALECT_Tnn_OPS