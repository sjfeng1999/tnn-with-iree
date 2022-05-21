//
//
//

#include "iree/compiler/Dialect/HAL/Conversion/ConversionDialectInterface.h"
#include "iree/compiler/Dialect/VM/Conversion/ConversionDialectInterface.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Dialect/IR/TnnDialect.h"
#include "Dialect/IR/ConversionPatterns.h"
#include "Tnn.imports.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace tnn {

namespace {

class TnnToHALConversionInterface : public HALConversionDialectInterface {
public:
    using HALConversionDialectInterface::HALConversionDialectInterface;

    void setupConversionTarget(ConversionTarget &target, RewritePatternSet &patterns,
                               TypeConverter &typeConverter) const override {
        populateTnnToHALPatterns(getDialect()->getContext(), patterns, typeConverter);
    }
};


class TnnToVMConversionInterface : public VMConversionDialectInterface {
public:
    using VMConversionDialectInterface::VMConversionDialectInterface;

    OwningOpRef<mlir::ModuleOp> parseVMImportModule() const override {
        return mlir::parseSourceString<mlir::ModuleOp>(
            StringRef(Tnn_imports_create()->data, Tnn_imports_create()->size),
            getDialect()->getContext());
    }

    void populateVMConversionPatterns(SymbolTable &importSymbols, RewritePatternSet &patterns,
                                        TypeConverter &typeConverter) const override {
        populateTnnToVMPatterns(getDialect()->getContext(), importSymbols, patterns, typeConverter);
    }
};

}  // namespace

TnnDialect::TnnDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<TnnDialect>()) {

    addInterfaces<TnnToHALConversionInterface, TnnToVMConversionInterface>();

    addTypes<Blob>();

#define GET_OP_LIST
    addOperations<
#include "Dialect/IR/TnnOps.cc.inc"
    >();
}

Type TnnDialect::parseType(DialectAsmParser &parser) const {
    StringRef typeName;
    if (failed(parser.parseKeyword(&typeName)))
        return {};
        
    auto type = llvm::StringSwitch<Type>(typeName)
                .Case("blob", Blob::get(getContext()))
                .Default(nullptr);
    if (!type) {
        parser.emitError(parser.getCurrentLocation()) << "unknown type: " << typeName;
    }
    return type;
}

void TnnDialect::printType(Type type, DialectAsmPrinter &p) const {
    if (type.isa<Blob>()) {
        p << "blob";
    } else {
        assert(false && "unknown type");
    }
}

}  // namespace tnn
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#define GET_OP_CLASSES
#include "Dialect/IR/TnnOps.cc.inc"
