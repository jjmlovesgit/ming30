#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Stream BF16 expert banks onto the device and report residency. Buffers are kept as the
// raw BF16 bytes they are stored as - no float32 copy is materialised on the device.
//   out[0] total bytes uploaded (MiB)
//   out[1] peak VRAM in use during the upload (MiB)
//   out[2] free VRAM after the upload (MiB)
int ming_gpu_residency_begin(void);
int ming_gpu_residency_add(const void * bytes, size_t size);
int ming_gpu_residency_end(double * out);
void ming_gpu_residency_release(void);

#ifdef __cplusplus
}
#endif
