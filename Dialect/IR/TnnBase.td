
#ifndef TNN_BASE
#define TNN_BASE

include "Dialect/IR/TnnDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class Tnn_Op<string mnemonic, list<Trait> traits = []> :
    Op<Tnn_Dialect, mnemonic, traits>;

class Tnn_PureOp<string mnemonic, list<Trait> traits = []> :
    Tnn_Op<mnemonic, !listconcat(traits, [NoSideEffect])>;


def Tnn_Blob : DialectType<
    Tnn_Dialect,
    CPred<"$_self.isa<TNN_NS::iree::BlobType>()>">,
    "TnnBlob"> {

    let summary = [{
        TNN Data Struct
    }];
}

#endif
