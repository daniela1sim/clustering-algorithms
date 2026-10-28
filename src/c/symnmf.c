#define _POSIX_C_SOURCE 200809L
#include "symnmf.h"
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef SYMNMF_NO_MAIN
#include <sys/types.h>
#endif

#define AT(m, r, c) ((m)->data[(r) * (m)->cols + (c)])

Matrix *matrix_new(size_t rows, size_t cols) {
    Matrix *m;
    if (!rows || !cols || cols > SIZE_MAX / rows || rows * cols > SIZE_MAX / sizeof(double))
        return NULL;
    m = (Matrix *)calloc(1, sizeof(*m));
    if (!m) return NULL;
    m->data = (double *)calloc(rows * cols, sizeof(double));
    if (!m->data) { free(m); return NULL; }
    m->rows = rows;
    m->cols = cols;
    return m;
}

void matrix_free(Matrix *m) {
    if (m) { free(m->data); free(m); }
}

Matrix *symnmf_affinity(const Matrix *points) {
    Matrix *a;
    size_t i, j, p;
    if (!points || !points->rows || !points->cols) return NULL;
    a = matrix_new(points->rows, points->rows);
    if (!a) return NULL;
    for (i = 0; i < points->rows; ++i) {
        for (j = i + 1; j < points->rows; ++j) {
            double sqdist = 0.0;
            for (p = 0; p < points->cols; ++p) {
                double delta = AT(points, i, p) - AT(points, j, p);
                sqdist += delta * delta;
            }
            AT(a, i, j) = AT(a, j, i) = exp(-sqdist / 2.0);
        }
    }
    return a;
}

Matrix *symnmf_degree(const Matrix *affinity) {
    Matrix *d;
    size_t i, j;
    if (!affinity || affinity->rows != affinity->cols) return NULL;
    d = matrix_new(affinity->rows, affinity->cols);
    if (!d) return NULL;
    for (i = 0; i < affinity->rows; ++i)
        for (j = 0; j < affinity->cols; ++j)
            AT(d, i, i) += AT(affinity, i, j);
    return d;
}

Matrix *symnmf_normalize(const Matrix *affinity) {
    Matrix *w;
    double *invsqrt;
    size_t i, j, n;
    if (!affinity || affinity->rows != affinity->cols) return NULL;
    n = affinity->rows;
    w = matrix_new(n, n);
    invsqrt = (double *)calloc(n, sizeof(double));
    if (!w || !invsqrt) { matrix_free(w); free(invsqrt); return NULL; }
    for (i = 0; i < n; ++i) {
        double sum = 0.0;
        for (j = 0; j < n; ++j) sum += AT(affinity, i, j);
        if (sum > 0.0) invsqrt[i] = 1.0 / sqrt(sum);
    }
    for (i = 0; i < n; ++i)
        for (j = 0; j < n; ++j)
            AT(w, i, j) = AT(affinity, i, j) * invsqrt[i] * invsqrt[j];
    free(invsqrt);
    return w;
}

Matrix *symnmf_factorize(const Matrix *w, const Matrix *initial_h,
                        size_t max_iterations, double tolerance) {
    Matrix *h, *next, *wh, *hth;
    size_t i, j, k, t, n, rank, iteration;
    if (!w || !initial_h || w->rows != w->cols ||
        w->rows != initial_h->rows || !initial_h->cols ||
        !max_iterations || tolerance <= 0.0) return NULL;
    n = w->rows;
    rank = initial_h->cols;
    h = matrix_new(n, rank);
    next = matrix_new(n, rank);
    wh = matrix_new(n, rank);
    hth = matrix_new(rank, rank);
    if (!h || !next || !wh || !hth) goto error;
    for (i = 0; i < n * rank; ++i) {
        if (!isfinite(initial_h->data[i]) || initial_h->data[i] < 0.0) goto error;
        h->data[i] = initial_h->data[i];
    }
    for (iteration = 0; iteration < max_iterations; ++iteration) {
        double delta = 0.0;
        memset(wh->data, 0, n * rank * sizeof(double));
        memset(hth->data, 0, rank * rank * sizeof(double));
        /* WH and H^T H; H(H^T H) == HH^T H. */
        for (i = 0; i < n; ++i) {
            for (j = 0; j < n; ++j) {
                double weight = AT(w, i, j);
                for (k = 0; k < rank; ++k) AT(wh, i, k) += weight * AT(h, j, k);
            }
            for (j = 0; j < rank; ++j)
                for (k = 0; k < rank; ++k)
                    AT(hth, j, k) += AT(h, i, j) * AT(h, i, k);
        }
        for (i = 0; i < n; ++i)
            for (j = 0; j < rank; ++j) {
                double denom = 0.0, multiplier, value;
                for (t = 0; t < rank; ++t) denom += AT(h, i, t) * AT(hth, t, j);
                multiplier = denom > 0.0 ? 0.5 + 0.5 * AT(wh, i, j) / denom : 0.5;
                value = AT(h, i, j) * multiplier;
                if (!isfinite(value)) goto error;
                AT(next, i, j) = value;
                delta += (value - AT(h, i, j)) * (value - AT(h, i, j));
            }
        { Matrix *tmp = h; h = next; next = tmp; }
        if (delta < tolerance) break;
    }
    matrix_free(next);
    matrix_free(wh);
    matrix_free(hth);
    return h;
error:
    matrix_free(h);
    matrix_free(next);
    matrix_free(wh);
    matrix_free(hth);
    return NULL;
}

void matrix_print(const Matrix *m) {
    size_t i, j;
    if (!m) return;
    for (i = 0; i < m->rows; ++i) {
        for (j = 0; j < m->cols; ++j)
            printf(j + 1 == m->cols ? "%.4f" : "%.4f,", AT(m, i, j));
        putchar('\n');
    }
}

#ifndef SYMNMF_NO_MAIN
/* Parse numeric CSV rows (with .txt extension), preserving rectangularity. */
static Matrix *read_points(const char *path) {
    FILE *fp = fopen(path, "r");
    char *line = NULL;
    size_t capacity = 0, count = 0, rows = 0, cols = 0, allocated = 0;
    double *values = NULL;
    Matrix *m = NULL;
    ssize_t size;
    if (!fp) return NULL;
    while ((size = getline(&line, &capacity, fp)) >= 0) {
        char *p = line;
        size_t width = 0;
        if (size == 0 || *p == '\n' || *p == '\r') continue;
        while (*p) {
            char *end;
            double x;
            errno = 0;
            x = strtod(p, &end);
            if (end == p || errno == ERANGE || !isfinite(x)) goto cleanup;
            if (count == allocated) {
                size_t newsize = allocated ? allocated * 2 : 64;
                double *tmp;
                if (newsize <= allocated || newsize > SIZE_MAX / sizeof(double)) goto cleanup;
                tmp = (double *)realloc(values, newsize * sizeof(double));
                if (!tmp) goto cleanup;
                values = tmp; allocated = newsize;
            }
            values[count++] = x; ++width; p = end;
            while (*p == ' ' || *p == '\t') ++p;
            if (*p == ',') { ++p; continue; }
            if (*p == '\r' || *p == '\n' || *p == '\0') break;
            goto cleanup;
        }
        if (!width || (cols && width != cols)) goto cleanup;
        cols = width; ++rows;
    }
    if (!rows || !cols || count != rows * cols) goto cleanup;
    m = matrix_new(rows, cols);
    if (m) memcpy(m->data, values, count * sizeof(double));
cleanup:
    free(values); free(line); fclose(fp);
    return m;
}

int main(int argc, char **argv) {
    Matrix *points = NULL, *a = NULL, *result = NULL;
    int status = EXIT_FAILURE;
    if (argc != 3 || (strcmp(argv[1], "sym") && strcmp(argv[1], "ddg") && strcmp(argv[1], "norm"))) {
        fprintf(stderr, "Usage: symnmf {sym|ddg|norm} data.txt\n");
        return EXIT_FAILURE;
    }
    points = read_points(argv[2]);
    if (!points) goto cleanup;
    a = symnmf_affinity(points);
    if (!a) goto cleanup;
    if (!strcmp(argv[1], "sym")) { result = a; a = NULL; }
    else if (!strcmp(argv[1], "ddg")) result = symnmf_degree(a);
    else result = symnmf_normalize(a);
    if (!result) goto cleanup;
    matrix_print(result);
    status = EXIT_SUCCESS;
cleanup:
    if (status) fprintf(stderr, "Invalid input or computation failed.\n");
    matrix_free(result); matrix_free(a); matrix_free(points);
    return status;
}
#endif
