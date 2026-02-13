//===- WAMIAttrs.td - WAMI dialect attributes --------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef WAMI_ATTRS
#define WAMI_ATTRS

include "WAMI/WAMIDialect.td"
include "mlir/IR/AttrTypeBase.td"

class WAMI_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<WAMI_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def WAMI_OnLabelHandlerAttr : WAMI_Attr<"OnLabelHandler", "on_label"> {
  let summary = "Resume handler that dispatches to an enclosing structured label";
  let parameters = (ins
    "::mlir::FlatSymbolRefAttr":$tag,
    "int64_t":$level
  );
  let assemblyFormat = "`<` `tag` `=` $tag `,` `level` `=` $level `>`";
}

def WAMI_OnSwitchHandlerAttr : WAMI_Attr<"OnSwitchHandler", "on_switch"> {
  let summary = "Resume handler that installs switch handling for a tag";
  let parameters = (ins
    "::mlir::FlatSymbolRefAttr":$tag
  );
  let assemblyFormat = "`<` `tag` `=` $tag `>`";
}

#endif // WAMI_ATTRS
