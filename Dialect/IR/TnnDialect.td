
#ifndef TNN_DIALECT
#define TNN_DIALECT

include "mlir/IR/OpBase.td"

def Tnn_Dialect : Dialect {
    let name = "tnn";

    let summary = [{
        TNN Inference Framework Dialect
    }];

    let description = [{
        TNN Inference Framework Dialect.
        Just a longer description.
        Multi-lines.
    }];

    let cppNamespace = "TNN_NS::iree";
    
}

#endif // TNN_DIALECT