//
//
//

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace tnn {

void populateTnnToVMPatterns(MLIRContext *context, SymbolTable &importSymbols, RewritePatternSet &patterns,
                                TypeConverter &typeConverter);

}  // namespace tnn
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

