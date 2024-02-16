
/* Build Linear Equation System of test */

void matrixTest(double *matrix, int dimension_matrix)
{
	int nn_element = dimension_matrix * dimension_matrix - 1;
    for (int i = 0; i < dimension_matrix; i++)
    {
        int diagonalIndex = dimension_matrix * i + i;
        for (int j = 0; j < dimension_matrix; j++)
        {
            int index = dimension_matrix * i + j;
            if (index == diagonalIndex)
            {
                matrix[index] = 3.0;
                if (index == 0) 
                {
                    matrix[index+1] = -1.0;
                    j = index + 1;
                }
                else if (index == nn_element) 
                {
                    matrix[index-1] = -1.0;
                }
                else 
                {
                    matrix[index-1] = -1.0;
                    matrix[index+1] = -1.0;
                    j = index + 1;
                }
            }
            else
            {
                matrix[index] = 0.0;
            }
        }
    }
    for (int i = 0; i < dimension_matrix; i++)
    {
        int index = (dimension_matrix - 1) * (i + 1);
        if (matrix[index] == 0.0) 
        {
            matrix[index] = 0.5;
        }
    }
}

void vectorTest(double *vector, int vector_size)
{
	int halfVector = (int)floor(vector_size/2);

    vector[0] = 2.5;
	vector[vector_size - 1] = 2.5;
	vector[halfVector - 1] = 1.0;
	vector[halfVector] = 1.0;
    
    for (int i = 1; i <= halfVector-2; i++)
    {
        vector[i] = 1.5;
	}
    
    for (int i = halfVector+1; i <= vector_size-2; i++)
    {
        vector[i] = 1.5;
	}
}   