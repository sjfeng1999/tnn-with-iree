//
//

#include "iree/compiler/Dialect/HAL/Conversion/ConversionTarget.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"

#include "dialect/IR/conversion_patterns.h"
#include "dialect/IR/TnnDialect.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace tnn {

void populateCustomToHALPatterns(MLIRContext *context,
                                 RewritePatternSet &patterns,
                                 TypeConverter &typeConverter) {
  // We can use the HAL conversion handler for this tensor->buffer conversion
  // as we just want the simple form. If we wanted to perform additional
  // verification or have a specific use case (such as a place where only the
  // buffer is required and the shape is not) we could add our own.
  patterns.insert<HALOpConversion<TensorToMessageOp, BufferToMessageOp>>(
      context, typeConverter);
  patterns.insert<HALOpConversion<MessageToTensorOp, MessageToBufferOp>>(
      context, typeConverter);
}

void populateCustomToVMPatterns(MLIRContext *context,
                                SymbolTable &importSymbols,
                                RewritePatternSet &patterns,
                                TypeConverter &typeConverter) {
  // We can use the VM conversion handler for all of these as they are simple
  // 1:1 mappings. More complex mappings can provide their own conversions
  // (such as the HAL dialect does).
  patterns.insert<VMImportOpConversion<BufferToMessageOp>>(
      context, importSymbols, typeConverter, "tnn.buffer_to_message");
  patterns.insert<VMImportOpConversion<MessageToBufferOp>>(
      context, importSymbols, typeConverter, "tnn.message_to_buffer");
  patterns.insert<VMImportOpConversion<PrintOp>>(
      context, importSymbols, typeConverter, "tnn.print");
  patterns.insert<VMImportOpConversion<ReverseOp>>(
      context, importSymbols, typeConverter, "tnn.reverse");
  patterns.insert<VMImportOpConversion<GetUniqueMessageOp>>(
      context, importSymbols, typeConverter, "tnn.get_unique_message");
  patterns.insert<VMImportOpConversion<ReluOp>>(
      context, importSymbols, typeConverter, "tnn.relu");
}

}
}
}  // namespace iree
}  // namespace TNN_NS
