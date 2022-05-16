// 
//

#include "iree/compiler/Dialect/HAL/Conversion/ConversionDialectInterface.h"
#include "iree/compiler/Dialect/VM/Conversion/ConversionDialectInterface.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/DialectConversion.h"

#include "tnn.imports.h"
#include "dialect/IR/TnnDialect.h"
#include "dialect/IR/conversion_patterns.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace tnn {
  
namespace {

// Exposes conversion patterns that transition tensors to buffers during the
// Flow->HAL dialect lowering. This is only required if the dialect has ops that
// use tensor types.
class CustomToHALConversionInterface : public HALConversionDialectInterface {
 public:
  using HALConversionDialectInterface::HALConversionDialectInterface;

  void setupConversionTarget(ConversionTarget &target,
                             RewritePatternSet &patterns,
                             TypeConverter &typeConverter) const override {
    populateCustomToHALPatterns(getDialect()->getContext(), patterns,
                                typeConverter);
  }
};

// Exposes the import module and conversion patterns used to convert custom
// ops to their vm.import counterparts.
class CustomToVMConversionInterface : public VMConversionDialectInterface {
 public:
  using VMConversionDialectInterface::VMConversionDialectInterface;

  OwningOpRef<mlir::ModuleOp> parseVMImportModule() const override {
    return mlir::parseSourceString<mlir::ModuleOp>(
        StringRef(tnn_imports_create()->data, tnn_imports_create()->size),
        getDialect()->getContext());
  }

  void populateVMConversionPatterns(
      SymbolTable &importSymbols, RewritePatternSet &patterns,
      TypeConverter &typeConverter) const override {
    populateCustomToVMPatterns(getDialect()->getContext(), importSymbols,
                               patterns, typeConverter);
  }
};

}  // namespace

TnnDialect::TnnDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context, TypeID::get<TnnDialect>()) {
  addInterfaces<CustomToHALConversionInterface,
                CustomToVMConversionInterface>();

  addTypes<MessageType>();
  addTypes<TnnBlob>();

#define GET_OP_LIST
  addOperations<
#include "TnnOps.cc.inc"
      >();
}

Type TnnDialect::parseType(DialectAsmParser &parser) const {
  StringRef typeName;
  if (failed(parser.parseKeyword(&typeName))) return {};
  auto type = llvm::StringSwitch<Type>(typeName)
                  .Case("message", MessageType::get(getContext()))
                  .Case("TnnBlob", TnnBlob::get(getContext()))
                  .Default(nullptr);
  if (!type) {
    parser.emitError(parser.getCurrentLocation())
        << "unknown type: " << typeName;
  }
  return type;
}

void TnnDialect::printType(Type type, DialectAsmPrinter &p) const {
  if (type.isa<MessageType>()) {
    p << "message";
  } else if (type.isa<TnnBlob>()) {
    p << "TnnBlob";
  } else {
    assert(false && "unknown type");
  }
}

}  // namespace iree
}  // namespace TNN_NS
}
}

#define GET_OP_CLASSES
#include "TnnOps.cc.inc"
