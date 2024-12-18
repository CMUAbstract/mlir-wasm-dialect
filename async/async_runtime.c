#include <stdbool.h>
#include <stdint.h>

// Visibility macro (adjust if needed for your environment)
#ifdef _WIN32
#ifndef MLIR_ASYNC_RUNTIME_EXPORT
#ifdef mlir_async_runtime_EXPORTS
#define MLIR_ASYNC_RUNTIME_EXPORT __declspec(dllexport)
#else
#define MLIR_ASYNC_RUNTIME_EXPORT __declspec(dllimport)
#endif
#endif
#else
#define MLIR_ASYNC_RUNTIME_EXPORT __attribute__((visibility("default")))
#endif

//===----------------------------------------------------------------------===//
// Type definitions as per original API
//===----------------------------------------------------------------------===//

typedef struct AsyncToken AsyncToken;
typedef struct AsyncGroup AsyncGroup;
typedef struct AsyncValue AsyncValue;

typedef unsigned char *ValueStorage;
typedef void *CoroHandle;
typedef void (*CoroResume)(void *);
typedef void *RefCountedObjPtr;

//===----------------------------------------------------------------------===//
// Internal data structures
//===----------------------------------------------------------------------===//

// Base structure for all ref-counted objects
typedef struct {
  int64_t refCount;
  bool ready;
  bool error;
} BaseRefCounted;

struct AsyncToken {
  BaseRefCounted base;
};

struct AsyncValue {
  BaseRefCounted base;
  unsigned char storage[64]; // fixed storage size; adjust as needed
  int64_t size;
};

struct AsyncGroup {
  BaseRefCounted base;
  // Fixed-size arrays for group members
  // Adjust capacity as needed
  BaseRefCounted *members[8];
  int64_t count;
};

//===----------------------------------------------------------------------===//
// Global "pools" for tokens, values, and groups
// Adjust sizes as needed
//===----------------------------------------------------------------------===//

#define MAX_TOKENS 16
#define MAX_VALUES 16
#define MAX_GROUPS 16

static AsyncToken g_tokens[MAX_TOKENS];
static bool g_tokenUsed[MAX_TOKENS];

static AsyncValue g_values[MAX_VALUES];
static bool g_valueUsed[MAX_VALUES];

static AsyncGroup g_groups[MAX_GROUPS];
static bool g_groupUsed[MAX_GROUPS];

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

static void initBase(BaseRefCounted *base) {
  base->refCount = 1;
  base->ready = false;
  base->error = false;
}

static AsyncToken *allocateToken() {
  for (int i = 0; i < MAX_TOKENS; i++) {
    if (!g_tokenUsed[i]) {
      g_tokenUsed[i] = true;
      initBase(&g_tokens[i].base);
      return &g_tokens[i];
    }
  }
  return 0; // no free token
}

static AsyncValue *allocateValue(int64_t size) {
  for (int i = 0; i < MAX_VALUES; i++) {
    if (!g_valueUsed[i]) {
      g_valueUsed[i] = true;
      initBase(&g_values[i].base);
      // Truncate if size > 64
      if (size > 64)
        size = 64;
      g_values[i].size = size;
      return &g_values[i];
    }
  }
  return 0; // no free value
}

static AsyncGroup *allocateGroup(int64_t capacity) {
  // capacity ignored in this minimal implementation, max 8 supported
  for (int i = 0; i < MAX_GROUPS; i++) {
    if (!g_groupUsed[i]) {
      g_groupUsed[i] = true;
      initBase(&g_groups[i].base);
      g_groups[i].count = 0;
      return &g_groups[i];
    }
  }
  return 0; // no free group
}

static void addRef(BaseRefCounted *obj, int64_t count) {
  if (!obj)
    return;
  obj->refCount += count;
}

static void dropRef(BaseRefCounted *obj, int64_t count) {
  if (!obj)
    return;
  obj->refCount -= count;
  if (obj->refCount <= 0) {
    // "Free" the object by marking its slot as unused
    // Identify which pool it belongs to
    for (int i = 0; i < MAX_TOKENS; i++) {
      if ((AsyncToken *)obj == &g_tokens[i]) {
        g_tokenUsed[i] = false;
        return;
      }
    }
    for (int i = 0; i < MAX_VALUES; i++) {
      if ((AsyncValue *)obj == &g_values[i]) {
        g_valueUsed[i] = false;
        return;
      }
    }
    for (int i = 0; i < MAX_GROUPS; i++) {
      if ((AsyncGroup *)obj == &g_groups[i]) {
        g_groupUsed[i] = false;
        return;
      }
    }
  }
}

static bool checkGroupError(AsyncGroup *grp) {
  for (int64_t i = 0; i < grp->count; i++) {
    if (grp->members[i]->error)
      return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// API Implementation
//===----------------------------------------------------------------------===//

MLIR_ASYNC_RUNTIME_EXPORT void mlirAsyncRuntimeAddRef(RefCountedObjPtr ptr,
                                                      int64_t count) {
  addRef((BaseRefCounted *)ptr, count);
}

MLIR_ASYNC_RUNTIME_EXPORT void mlirAsyncRuntimeDropRef(RefCountedObjPtr ptr,
                                                       int64_t count) {
  dropRef((BaseRefCounted *)ptr, count);
}

MLIR_ASYNC_RUNTIME_EXPORT AsyncToken *mlirAsyncRuntimeCreateToken() {
  return allocateToken();
}

MLIR_ASYNC_RUNTIME_EXPORT AsyncValue *
mlirAsyncRuntimeCreateValue(int64_t size) {
  return allocateValue(size);
}

MLIR_ASYNC_RUNTIME_EXPORT AsyncGroup *
mlirAsyncRuntimeCreateGroup(int64_t size) {
  return allocateGroup(size);
}

MLIR_ASYNC_RUNTIME_EXPORT int64_t
mlirAsyncRuntimeAddTokenToGroup(AsyncToken *token, AsyncGroup *group) {
  if (!token || !group)
    return 0;
  if (group->count >= 8)
    return group->count;
  group->members[group->count++] = (BaseRefCounted *)token;
  return group->count;
}

MLIR_ASYNC_RUNTIME_EXPORT void mlirAsyncRuntimeEmplaceToken(AsyncToken *token) {
  if (!token)
    return;
  token->base.ready = true;
}

MLIR_ASYNC_RUNTIME_EXPORT void mlirAsyncRuntimeEmplaceValue(AsyncValue *value) {
  if (!value)
    return;
  value->base.ready = true;
}

MLIR_ASYNC_RUNTIME_EXPORT void
mlirAsyncRuntimeSetTokenError(AsyncToken *token) {
  if (!token)
    return;
  token->base.ready = true;
  token->base.error = true;
}

MLIR_ASYNC_RUNTIME_EXPORT void
mlirAsyncRuntimeSetValueError(AsyncValue *value) {
  if (!value)
    return;
  value->base.ready = true;
  value->base.error = true;
}

MLIR_ASYNC_RUNTIME_EXPORT bool mlirAsyncRuntimeIsTokenError(AsyncToken *token) {
  if (!token)
    return false;
  return token->base.error;
}

MLIR_ASYNC_RUNTIME_EXPORT bool mlirAsyncRuntimeIsValueError(AsyncValue *value) {
  if (!value)
    return false;
  return value->base.error;
}

MLIR_ASYNC_RUNTIME_EXPORT bool mlirAsyncRuntimeIsGroupError(AsyncGroup *group) {
  if (!group)
    return false;
  return checkGroupError(group);
}

// In this minimal environment, we do not block or check conditions.
// We simply rely on the caller ensuring readiness.
// Without assertions or imports, we cannot safely handle errors.

MLIR_ASYNC_RUNTIME_EXPORT void mlirAsyncRuntimeAwaitToken(AsyncToken *token) {
  // No operation. Assume ready.
}

MLIR_ASYNC_RUNTIME_EXPORT void mlirAsyncRuntimeAwaitValue(AsyncValue *value) {
  // No operation. Assume ready.
}

MLIR_ASYNC_RUNTIME_EXPORT void
mlirAsyncRuntimeAwaitAllInGroup(AsyncGroup *group) {
  // No operation. Assume ready.
}

MLIR_ASYNC_RUNTIME_EXPORT ValueStorage
mlirAsyncRuntimeGetValueStorage(AsyncValue *value) {
  if (!value)
    return 0;
  return value->storage;
}

MLIR_ASYNC_RUNTIME_EXPORT void mlirAsyncRuntimeExecute(CoroHandle h,
                                                       CoroResume resume) {
  if (resume)
    resume(h);
}

MLIR_ASYNC_RUNTIME_EXPORT void
mlirAsyncRuntimeAwaitTokenAndExecute(AsyncToken *token, CoroHandle h,
                                     CoroResume resume) {
  if (resume)
    resume(h);
}

MLIR_ASYNC_RUNTIME_EXPORT void
mlirAsyncRuntimeAwaitValueAndExecute(AsyncValue *value, CoroHandle h,
                                     CoroResume resume) {
  if (resume)
    resume(h);
}

MLIR_ASYNC_RUNTIME_EXPORT void
mlirAsyncRuntimeAwaitAllInGroupAndExecute(AsyncGroup *group, CoroHandle h,
                                          CoroResume resume) {
  if (resume)
    resume(h);
}

MLIR_ASYNC_RUNTIME_EXPORT int64_t mlirAsyncRuntimGetNumWorkerThreads() {
  return 1;
}

MLIR_ASYNC_RUNTIME_EXPORT void mlirAsyncRuntimePrintCurrentThreadId() {
  // No operation. Avoid printing to reduce imports.
}
