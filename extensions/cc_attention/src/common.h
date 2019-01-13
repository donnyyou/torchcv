#ifndef __COMMON__
#define __COMMON__
#include <cuda_runtime_api.h>

/*
 * General settings
 */
const int WARP_SIZE = 32;
const int MAX_BLOCK_SIZE = 512;

/*
 * Utility functions
 */
template <typename T>
__device__ __forceinline__ T WARP_SHFL_XOR(T value, int laneMask, int width = warpSize,
                                           unsigned int mask = 0xffffffff) {
#if CUDART_VERSION >= 9000
  return __shfl_xor_sync(mask, value, laneMask, width);
#else
  return __shfl_xor(value, laneMask, width);
#endif
}

__device__ __forceinline__ int getMSB(int val) { return 31 - __clz(val); }

static int getNumThreads(int nElem) {
  int threadSizes[5] = {32, 64, 128, 256, MAX_BLOCK_SIZE};
  for (int i = 0; i != 5; ++i) {
    if (nElem <= threadSizes[i]) {
      return threadSizes[i];
    }
  }
  return MAX_BLOCK_SIZE;
}


#endif