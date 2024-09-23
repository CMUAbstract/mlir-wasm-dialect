#include "Wasm/utility.h"

namespace mlir::wasm {

int64_t memRefSize(MemRefType memRefType, int64_t alignment) {
  // Step 1: Get the shape (dimensions)
  auto shape = memRefType.getShape();

  // Step 2: Compute the total number of elements
  int64_t totalElements = 1;
  for (int64_t dimSize : shape) {
    totalElements *= dimSize;
  }

  // Step 3: Determine the size of each element
  int64_t elementSize = memRefType.getElementType().getIntOrFloatBitWidth() / 8;

  // Step 4: Calculate the total memory size
  int64_t totalMemorySize = totalElements * elementSize;

  // Step 5: Adjust for alignment
  // Align to the nearest multiple of 'alignment'
  int64_t alignedMemorySize =
      ((totalMemorySize + alignment - 1) / alignment) * alignment;

  return alignedMemorySize;
}
} // namespace mlir::wasm
