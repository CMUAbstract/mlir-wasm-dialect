
#include "DCont/DContTypes.h"
#include "SsaWasm/SsaWasmOps.h"
#include "SsaWasm/SsaWasmTypes.h"

namespace mlir::ssawasm {
class SsaWasmTypeConverter : public TypeConverter {
public:
  SsaWasmTypeConverter(MLIRContext *ctx) {
    addConversion([ctx](IntegerType type) -> Type {
      auto width = type.getWidth();
      if (width == 1) {
        return WasmIntegerType::get(ctx, 32);
      }
      return WasmIntegerType::get(ctx, width);
    });
    addConversion([ctx](FloatType type) -> Type {
      return WasmFloatType::get(ctx, type.getWidth());
    });
    addConversion([ctx](MemRefType type) -> Type {
      return WasmMemRefType::get(ctx, type);
    });
    addConversion([ctx](IndexType type) -> Type {
      return WasmIntegerType::get(ctx, 32);
    });
    addConversion([ctx](dcont::ContType type) -> Type {
      return WasmContinuationType::get(ctx, type.getId());
    });
    addConversion([ctx](dcont::StorageType type) -> Type {
      return WasmContinuationType::get(ctx, type.getId());
    });

    addSourceMaterialization([](OpBuilder &builder, Type type,
                                ValueRange inputs, Location loc) -> Value {
      return builder.create<UnrealizedConversionCastOp>(loc, type, inputs[0])
          .getResult(0);
    });

    addTargetMaterialization([](OpBuilder &builder, Type type,
                                ValueRange inputs, Location loc) -> Value {
      return builder.create<UnrealizedConversionCastOp>(loc, type, inputs[0])
          .getResult(0);
    });
  }
};
} // namespace mlir::ssawasm