//===- WasmStackAttrs.td - WasmStack dialect attributes ------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef WASMSTACK_ATTRS
#define WASMSTACK_ATTRS

include "mlir/IR/AttrTypeBase.td"
include "wasmstack/WasmStackDialect.td"

class WasmStack_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<WasmStack_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def WasmStack_OnLabelHandlerAttr
    : WasmStack_Attr<"OnLabelHandler", "on_label"> {
  let summary = "Resume handler that dispatches to a structured label";
  let parameters = (ins
    "::mlir::FlatSymbolRefAttr":$tag,
    "::mlir::FlatSymbolRefAttr":$label
  );
  let assemblyFormat = "`<` `tag` `=` $tag `,` `label` `=` $label `>`";
}

def WasmStack_OnSwitchHandlerAttr
    : WasmStack_Attr<"OnSwitchHandler", "on_switch"> {
  let summary = "Resume handler that installs switch handling for a tag";
  let parameters = (ins
    "::mlir::FlatSymbolRefAttr":$tag
  );
  let assemblyFormat = "`<` `tag` `=` $tag `>`";
}

#endif // WASMSTACK_ATTRS
