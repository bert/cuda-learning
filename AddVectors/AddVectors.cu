/*!
 * \file HelloCUDA/kernel.cu
 *
 * \brief Testing CUDA hardware and code.
 */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

/*!
 * \brief Add vectors (\c A and \c B) to \c C.
 *
 * \note Device code.
 */
__global__ void
VecAdd
(
  float* A,
  float* B,
  float* C
)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

/*!
 * \brief Main program.
 *
 * \note Host code.
 */
int
main ()
{
  /* Kernel invocation with N threads. */
  VecAdd<<<1, N>>>(A, B, C);
  return 0;
}
