#ifndef SYMNMF_H
#define SYMNMF_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    size_t rows;
    size_t cols;
    double *data;
} Matrix;

Matrix *matrix_new(size_t rows, size_t cols);
void matrix_free(Matrix *matrix);
Matrix *symnmf_affinity(const Matrix *points);
Matrix *symnmf_degree(const Matrix *affinity);
Matrix *symnmf_normalize(const Matrix *affinity);
Matrix *symnmf_factorize(const Matrix *normalized, const Matrix *initial_h,
                        size_t max_iterations, double tolerance);
void matrix_print(const Matrix *matrix);

#endif
