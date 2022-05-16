// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
  static StringRef getDialectNamespace() { return "tnn"; }

  Type parseType(DialectAsmParser &parser) const override;
  void printType(Type type, DialectAsmPrinter &p) const override;
};

class MessageType : public Type::TypeBase<MessageType, Type, TypeStorage> {
 public:
  using Base::Base;
};

class TnnBlob : public Type::TypeBase<TnnBlob, Type, TypeStorage> {
 public:
  using Base::Base;
};

}
}
}  // namespace iree
}  // namespace TNN_NS

#define GET_OP_CLASSES
#include "TnnOps.h.inc"

