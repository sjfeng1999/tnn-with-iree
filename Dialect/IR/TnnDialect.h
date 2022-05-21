//
//
//

#pragma once

#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace tnn {

class TnnDialect : public Dialect {
public:
    explicit TnnDialect(MLIRContext *context);
    static StringRef getDialectNamespace() {
        return "tnn";
    }

    Type parseType(DialectAsmParser &parser) const override;
    void printType(Type type, DialectAsmPrinter &p) const override;
};

class Blob : public Type::TypeBase<Blob, Type, TypeStorage> {
public:
    using Base::Base;
};

}  // namespace tnn
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#define GET_OP_CLASSES
#include "Dialect/IR/TnnOps.h.inc"
