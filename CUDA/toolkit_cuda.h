#ifndef TOOLKIT_CUDA_H
#define TOOLKIT_CUDA_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Error handler
void cudaCheckError(const cudaError_t error);

// Allocate memory to an vector of double data type in the device
double *allocateMemoryVectorDevice(const int vector_size);

// Allocate memory to an matrix of double data type in the device
double *allocateMemoryMatrixDevice(const int rows, const int columns);

// Data transfer of a vector, from host to device
void transfer_vector_host_to_device(double *host_array, double *device_array, const int array_size);

// Data transfer of a vector, from device to hots
void transfer_vector_device_to_host(double *device_array, double *host_array, const int array_size);

// Data transfer of a matrix, from host to device
void transfer_matrix_host_to_device(double *host_array, double *device_array, const int rows, const int columns);

// Data transfer of a matrix, from device to hots
void transfer_matrix_device_to_host(double *device_array, double *host_array, const int rows, const int columns);

#endif