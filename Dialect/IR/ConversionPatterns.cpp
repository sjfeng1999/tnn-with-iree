//
//
//

#include "iree/compiler/Dialect/HAL/Conversion/ConversionTarget.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"

#include "Dialect/IR/ConversionPatterns.h"
#include "Dialect/IR/TnnDialect.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace tnn {


void populateTnnToHALPatterns(MLIRContext *context, RewritePatternSet &patterns,
                              TypeConverter &typeConverter) {

    patterns.insert<HALOpConversion<CastTensorToBlob, CastBufferToBlob>>(
        context, typeConverter);
    patterns.insert<HALOpConversion<CastBlobToTensor, CastBlobToBuffer>>(
        context, typeConverter);
}


void populateTnnToVMPatterns(MLIRContext *context, SymbolTable &importSymbols, RewritePatternSet &patterns,
                             TypeConverter &typeConverter) {

    patterns.insert<VMImportOpConversion<CastBlobToBuffer>>(
        context, importSymbols, typeConverter, "tnn.cast_blob_to_buffer");
    patterns.insert<VMImportOpConversion<CastBufferToBlob>>(
        context, importSymbols, typeConverter, "tnn.cast_buffer_to_blob");
    patterns.insert<VMImportOpConversion<ReluOp>>(
        context, importSymbols, typeConverter, "tnn.relu");
}

}  // namespace tnn
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
