//
//
//

#pragma once 
    
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Dialect/IR/TnnDialect.h" 

namespace mlir {
namespace iree_compiler {

inline void registerTnnDialect(DialectRegistry &registry) {
    registry.insert<IREE::tnn::TnnDialect>();
}

}  // namespace iree_compiler
}  // namespace mlir


