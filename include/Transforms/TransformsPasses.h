//===- TransformsPasses.h - Dialect-independent passes -----------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares dialect-independent transformation passes.
//
//===----------------------------------------------------------------------===//

#ifndef TRANSFORMS_TRANSFORMSPASSES_H
#define TRANSFORMS_TRANSFORMSPASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir::transforms {

#define GEN_PASS_DECL
#include "Transforms/TransformsPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "Transforms/TransformsPasses.h.inc"

/// Registers all dialect-independent transformation passes.
inline void registerPasses() { registerTransformsPasses(); }

} // namespace mlir::transforms

#endif // TRANSFORMS_TRANSFORMSPASSES_H
