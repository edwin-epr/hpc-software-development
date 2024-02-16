#ifndef TOOLKIT_CLANG_H
#define TOOLKIT_CLANG_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Error handler
void errorHandler(char error_string[]);

// Allocate memory to an vector of double data type
double *allocateMemoryVector(int vector_size);

// Allocate memory to an matrix of double data type
double *allocateMemoryMatrix(int rows, int columns);

// Print to console a Vector
void printVector(double *vector, int vector_size);

// Print to console a Matrix
void printMatrix(double *matrix, int rows, int columns);

#endif