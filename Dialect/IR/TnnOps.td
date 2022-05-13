
#ifndef TNN_OPS
#define TNN_OPS

include "Dialect/IR/TnnBase.td"

def Tnn_ReluOp : Tnn_PureOp<"Relu", []> {
    let summary = "Tensor Relu Activation";

    let arguments = (ins 
        Tnn_Blob: $input
    );

    let results = (outs
        Tnn_Blob: $output
    );
}

#endif 
