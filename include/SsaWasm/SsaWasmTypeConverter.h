
#include "DCont/DContTypes.h"
#include "SsaWasm/SsaWasmOps.h"
#include "SsaWasm/SsaWasmTypes.h"

namespace mlir::ssawasm {
class SsaWasmTypeConverter : public TypeConverter {
public:
  SsaWasmTypeConverter(MLIRContext *ctx) {
    addConversion([ctx](IntegerType type) -> Type {
      auto width = type.getWidth();
      if (width != 32 && width != 64) {
        return IntegerType::get(ctx, 32);
      }
      return type;
    });
    addConversion([](FloatType type) -> Type {
      auto width = type.getWidth();
      assert((width == 32 || width == 64) && "Unsupported float type");
      return type;
    });
    addConversion([](MemRefType type) -> Type { return type; });
    addConversion(
        [ctx](IndexType type) -> Type { return IntegerType::get(ctx, 32); });
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