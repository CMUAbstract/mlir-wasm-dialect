//===- WAMIAttrs.h - WAMI dialect attributes -------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef WAMI_WAMIATTRS_H
#define WAMI_WAMIATTRS_H

#include "WAMI/WAMIDialect.h"
#include "mlir/IR/Attributes.h"

#define GET_ATTRDEF_CLASSES
#include "WAMI/WAMIAttrs.h.inc"

#endif // WAMI_WAMIATTRS_H
