#include "toolkit_clang.h"

// Error handler
void errorHandler(char error_string[])
{
    fprintf(stderr, "C language runtime error...\n");
    fprintf(stderr, "%s\n", error_string);
    fprintf(stderr, "...now exiting to system...\n");
	exit(EXIT_FAILURE);
}

// Allocate memory to an vector of double data type
double *allocateMemoryVector(int vector_size)
{
    double *vector; 
    size_t size_in_bytes;

    size_in_bytes = vector_size * sizeof(double);
    vector = (double *)malloc(size_in_bytes);

    // Check that the memory allocation was successful.
    if (!vector)
    {
        errorHandler("Allocation failure in allocateMemoryVector");
    }
    return vector;
}

// Allocate memory to an matrix of double data type
double *allocateMemoryMatrix(int rows, int columns)
{
    double *matrix;
    size_t size_in_bytes;

    size_in_bytes = rows * columns * sizeof(double); 
    matrix = (double *)malloc(size_in_bytes);

    // Check that the memory allocation was successful.
    if (!matrix)
    {
        errorHandler("Allocation failure in allocateMemoryMatrix");
    }
    return matrix;
}

// Print to console a Vector
void printVector(double *vector, int vector_size)
{
    for (int i = 0; i < vector_size ; ++i)
    {
        printf("%f ", vector[i]);
        printf("\n");
    }
    printf("\n");
}

// Print to console a Matrix
void printMatrix(double *matrix, int rows, int columns)
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < columns; ++j) 
        {
            printf("%f ", matrix[rows * i + j]);

        }
        printf("\n");
    }
    printf("\n");
}
