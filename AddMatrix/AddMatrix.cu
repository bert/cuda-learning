/*!
 * \file AddMatrix/AddMatrix.cu
 *
 * \brief Testing CUDA hardware and code.
 */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

/*!
 * \brief Add matrices (\c A, \c B) to \c C).
 *
 * \note Device code.
 */
__global__ void
MatAdd
(
  float A[N][N],
  float B[N][N],
  float C[N][N]
)
{
  int i = threadIdx.x;
  int j = threadIdx.y;
  C[i][j] = A[i][j] + B[i][j];
}

/*!
 * \brief Main program.
 *
 * \note Host code.
 */
int
main ()
{
  int numBlocks = 1;

  /* Kernel invocation with one block of N * N * 1 threads. */
  dim3 threadsPerBlock (N, N);

  MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
  return 0;
}
