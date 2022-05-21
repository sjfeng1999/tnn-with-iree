

#ifndef DIALECT_IR_TNNOPS
#define DIALECT_IR_TNNOPS

include "iree/compiler/Dialect/HAL/IR/HALBase.td"
include "iree/compiler/Dialect/Util/IR/UtilBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//////////////////////////////////////////////////////////////////////////////////////////////////

def TNN_Dialect : Dialect {
  let name = "tnn";
  let cppNamespace = "mlir::iree_compiler::IREE::tnn";

  let summary = [{ A dialect }];
}

def TNN_Blob : DialectType<TNN_Dialect, CPred<"$_self.isa<IREE::tnn::Blob>()">, "blob"> {
    let summary = [{ TNN Data Struct }];
}

class TNN_Op<string mnemonic, list<Trait> traits = []> :
    Op<TNN_Dialect, mnemonic, traits>;

//////////////////////////////////////////////////////////////////////////////////////////////////

def TNN_CastTensorToBlob : TNN_Op<"cast_tensor_to_blob"> {
  let summary = [{ cast build-in tensor to TNN Blob}];

  let arguments = (ins AnyTensor: $tensor);
  let results = (outs TNN_Blob: $blob );
}

def TNN_CastBlobToTensor : TNN_Op<"cast_blob_to_tensor"> {
  let summary = [{ cast TNN Blob to build-in tensor}];

  let arguments = (ins TNN_Blob: $blob);
  let results = (outs AnyTensor: $tensor);
}

//////////////////////////////////////////////////////////////////////////////////////////////////

def TNN_CastBufferToBlob : TNN_Op<"cast_buffer_to_blob"> {
  let summary = [{ cast hal-buffer to TNN Blob}];

  let arguments = (ins HAL_BufferView: $tbuffer);
  let results = (outs TNN_Blob: $blob );
}

def TNN_CastBlobToBuffer : TNN_Op<"cast_blob_to_buffer"> {
  let summary = [{ cast TNN Blob to build-in hal-buffer}];

  let arguments = (ins TNN_Blob: $blob);
  let results = (outs HAL_BufferView: $buffer);
}

//////////////////////////////////////////////////////////////////////////////////////////////////

def TNN_ReluOp : TNN_Op<"relu", [NoSideEffect]> {
  let summary = [{ Tensor Relu Activation }];

  let arguments = (ins TNN_Blob: $input);
  let results = (outs TNN_Blob: $output);
} 


#endif  // DIALECT_IR_TNNOPS