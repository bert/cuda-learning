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
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < N && j < N)
  {
    C[i][j] = A[i][j] + B[i][j];
  }
}
/*!
 * \brief Main program.
 *
 * \note Host code.
 */

int
main ()
{
  float A;
  float B;
  float C;
  dim3 threadsPerBlock (16, 16);
  dim3 numBlocks (N / threadsPerBlock.x, N / threadsPerBlock.y);

  /* Kernel invocation. */
  MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
}
