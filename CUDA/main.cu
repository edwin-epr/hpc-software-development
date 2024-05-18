#include <stdio.h>
#include <stdint.h>
#include <cublas_v2.h>

#include "tools/toolkit_clang.h"
#include "tools/toolkit_cuda.h"

int main(int argc, char const *argv[])
{
    cublasHandle_t handle;

    double *h_vector_x, *h_vector_y;
    double *d_vector_x, *d_vector_y;

    const int vector_size = 10000;
    const double alpha = 1.0;

    h_vector_x = allocateMemoryVector(vector_size);
    h_vector_y = allocateMemoryVector(vector_size);

    for (int i = 0; i < vector_size; i++)
    {
        h_vector_x[i] = 1.0;
        h_vector_y[i] = 2.0;
    }

    d_vector_x = allocateMemoryVectorDevice(vector_size);
    d_vector_y = allocateMemoryVectorDevice(vector_size);

    transfer_vector_host_to_device(h_vector_x, d_vector_x, vector_size);
    transfer_vector_host_to_device(h_vector_y, d_vector_y, vector_size);

    cublasCreate(&handle);

    cublasDaxpy(handle, vector_size, &alpha, d_vector_x, 1, d_vector_y, 1);


    transfer_vector_device_to_host(d_vector_y, h_vector_y, vector_size);

    printVector(h_vector_y, vector_size);

    cudaFree(d_vector_y);
    cudaFree(d_vector_x);
    cublasDestroy(handle);
    free(h_vector_y);
    free(h_vector_x);

    return 0;
}
