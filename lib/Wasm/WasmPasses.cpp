//===- WasmPasses.cpp - Wasm passes -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Wasm/WasmPasses.h"

#include <algorithm>
#include <vector>

namespace mlir::wasm {
#define GEN_PASS_DEF_ARITHTOWASMPASS
#define GEN_PASS_DEF_VARIABLEANALYSIS
#include "Wasm/WasmPasses.h.inc"

using func::FuncOp;
using mlir::Value;
class VariableAnalysis {
public:
  VariableAnalysis(Operation *op) {
    numArguments = 0;
    numVariables = 0;
    if (auto func = dyn_cast<FuncOp>(op)) {
      numArguments = func.getNumArguments();
      // TODO: initialize reg2Loc with arguments

      func.walk([&](Operation *op) {
        if (auto constantOp = dyn_cast<mlir::arith::ConstantOp>(op)) {
          mlir::Value result = constantOp.getResult();
          reg2Loc.push_back(result);
          numVariables++;
        }
        if (auto addOp = dyn_cast<mlir::arith::AddIOp>(op)) {
          mlir::Value result = addOp.getResult();
          reg2Loc.push_back(result);
          numVariables++;
        }
        // TODO: handle other operations that define new variables
      });
    }
  }
  int getNumVariables() { return numVariables; }
  int getLocal(const mlir::Value &reg) {
    auto search = std::find(reg2Loc.begin(), reg2Loc.end(), reg);
    if (search != reg2Loc.end()) {
      return search - reg2Loc.begin();
    }
    return -1;
  }

private:
  int numArguments;
  int numVariables;
  // TODO: We need to keep track of the wasm type of each local, i.e. whether
  // it is an i32, i64, f32, or f64.
  std::vector<mlir::Value> reg2Loc;
};

class ArithToWasmPass : public impl::ArithToWasmPassBase<ArithToWasmPass> {
public:
  using impl::ArithToWasmPassBase<ArithToWasmPass>::ArithToWasmPassBase;

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, wasm::WasmDialect>();
  }

  void runOnOperation() final {

    VariableAnalysis &analysis = getAnalysis<VariableAnalysis>();

    FuncOp func = getOperation();
    MLIRContext *context = func.getContext();

    PatternRewriter rewriter(context);

    // TODO: It would be simpler to use func.walk([&](Operation *op)
    for (Region &region : func->getRegions()) {
      for (Block &block : region) {
        for (auto it = block.rbegin(), e = block.rend(); it != e;) {
          Operation *op = &(*it);
          if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
            // replace this with wasm.constant follwed by wasm.set_local
            mlir::Value result = constOp.getResult();
            int local = analysis.getLocal(result);

            mlir::Attribute attr = constOp->getAttr("value");

            // put wasm instructions here
            rewriter.setInsertionPoint(op);
            // add wasm.constant
            rewriter.create<wasm::I32ConstantOp>(
                op->getLoc(), mlir::cast<mlir::IntegerAttr>(attr));
            rewriter.create<wasm::LocalSetOp>(op->getLoc(), local);
            rewriter.clearInsertionPoint();
            ++it;
            rewriter.eraseOp(op);
          }

          else if (auto addOp = dyn_cast<arith::AddIOp>(op)) {
            mlir::Value result = addOp.getResult();
            int local = analysis.getLocal(result);

            auto lhs = addOp.getLhs();
            auto rhs = addOp.getRhs();

            int lhsLocal = analysis.getLocal(lhs);
            int rhsLocal = analysis.getLocal(rhs);

            rewriter.setInsertionPoint(op);
            rewriter.create<wasm::LocalGetOp>(op->getLoc(), lhsLocal);
            rewriter.create<wasm::LocalGetOp>(op->getLoc(), rhsLocal);
            rewriter.create<wasm::IAddOp>(op->getLoc());
            rewriter.create<wasm::LocalSetOp>(op->getLoc(), local);
            rewriter.clearInsertionPoint();
            ++it;
            rewriter.eraseOp(op);
          } else {
            ++it;
          }
        }
      }
    }

    Operation *firstOp = &(*func->getRegion(0).getBlocks().begin()->begin());
    rewriter.setInsertionPoint(firstOp);
    rewriter.create<wasm::LocalOp>(func.getLoc(), "i32");
  }
};
} // namespace mlir::wasm
