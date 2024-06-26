#ifndef TOOLKIT_CUDA_H
#define TOOLKIT_CUDA_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Error handler: device memory allocation check
void cudaMemoryCheckError(const cudaError_t error)
{
    if (error != cudaSuccess)
    {
        printf("Error: %s:%d, ", __FILE__, __LINE__);
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

// Error handler: cublas status check error
void cublasStatusCheckError(const cublasStatus_t status)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        printf("Error: %s:%d, ", __FILE__, __LINE__);
        printf("code:%d, reason: %s\n", status, cublasGetStatusString(status));
        exit(EXIT_FAILURE);
    }
}

// Allocate memory to an vector of double data type in the device
double *allocateMemoryVectorDevice(const int vector_size)
{
    size_t size_in_bytes; 
    cudaError_t error;
    double *device_pointer;

    size_in_bytes = vector_size * sizeof(double);
    error = cudaMalloc((void **)&device_pointer, size_in_bytes);
    cudaMemoryCheckError(error);
    return device_pointer;
}

// Allocate memory to an matrix of double data type in the device
double *allocateMemoryMatrixDevice(const int rows, const int columns)
{
    size_t size_in_bytes; 
    cudaError_t error;
    double *device_pointer;

    size_in_bytes = rows * columns * sizeof(double);
    error = cudaMalloc((void **)&device_pointer, size_in_bytes);
    cudaMemoryCheckError(error);
    return device_pointer;
}

// Data transfer of a vector, from host to device
void transfer_vector_host_to_device(double *host_array, double *device_array, const int array_size)
{
    size_t size_in_bytes; 
    cudaError_t error;

    size_in_bytes = array_size * sizeof(double);
    error = cudaMemcpy(device_array, host_array, size_in_bytes, cudaMemcpyHostToDevice);
    cudaMemoryCheckError(error);
}

// Data transfer of a vector, from device to hots
void transfer_vector_device_to_host(double *device_array, double *host_array, const int array_size)
{
    size_t size_in_bytes; 
    cudaError_t error;

    size_in_bytes = array_size * sizeof(double);
    error = cudaMemcpy(host_array, device_array, size_in_bytes, cudaMemcpyDeviceToHost);
    cudaMemoryCheckError(error);
}

// Data transfer of a matrix, from host to device
void transfer_matrix_host_to_device(double *host_array, double *device_array, const int rows, const int columns)
{
    size_t size_in_bytes; 
    cudaError_t error;

    size_in_bytes = rows * columns * sizeof(double);
    error = cudaMemcpy(device_array, host_array, size_in_bytes, cudaMemcpyHostToDevice);
    cudaMemoryCheckError(error);
}

// Data transfer of a matrix, from device to hots
void transfer_matrix_device_to_host(double *device_array, double *host_array, const int rows, const int columns)
{
    size_t size_in_bytes; 
    cudaError_t error;

    size_in_bytes = rows * columns * sizeof(double);
    error = cudaMemcpy(host_array, device_array, size_in_bytes, cudaMemcpyDeviceToHost);
    cudaMemoryCheckError(error);
}

#endif