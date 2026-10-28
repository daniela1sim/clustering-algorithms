#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* C implementation of Lloyd iterations, initialized by Python K-means++.
 * No process-wide state; multiple calls are independent.
 * API: fit(max_iterations, epsilon, points, initial_row_indices) -> centroids.
 */
static PyObject *fit(PyObject *self, PyObject *args) {
    (void)self;
    int max_iterations;
    double epsilon;
    PyObject *points_obj, *indices_obj;
    PyObject *points_seq = NULL, *indices_seq = NULL;
    PyObject *result = NULL, *row_obj = NULL;
    double *points = NULL, *centers = NULL, *sums = NULL;
    Py_ssize_t *counts = NULL;
    Py_ssize_t n, d = 0, k, i, j, c, best;
    int iteration;

    if (!PyArg_ParseTuple(args, "idOO", &max_iterations, &epsilon, &points_obj, &indices_obj)) return NULL;
    if (max_iterations < 1 || !(epsilon > 0) || !isfinite(epsilon)) {
        PyErr_SetString(PyExc_ValueError, "max_iterations and epsilon must be positive and finite");
        return NULL;
    }
    points_seq = PySequence_Fast(points_obj, "points must be a sequence of rows");
    if (!points_seq) goto fail;
    indices_seq = PySequence_Fast(indices_obj, "initial indices must be a sequence");
    if (!indices_seq) goto fail;
    n = PySequence_Fast_GET_SIZE(points_seq);
    k = PySequence_Fast_GET_SIZE(indices_seq);
    if (n < 2 || k < 1 || k >= n) {
        PyErr_SetString(PyExc_ValueError, "requires 1 <= k < number of rows");
        goto fail;
    }
    {
        PyObject *first = PySequence_Fast(PySequence_Fast_GET_ITEM(points_seq, 0), "point must be a sequence");
        if (!first) goto fail;
        d = PySequence_Fast_GET_SIZE(first);
        Py_DECREF(first);
    }
    if (d < 1 || n > PY_SSIZE_T_MAX / d || k > PY_SSIZE_T_MAX / d) {
        PyErr_SetString(PyExc_ValueError, "invalid input dimensions");
        goto fail;
    }
    points = PyMem_Malloc((size_t)n * (size_t)d * sizeof(double));
    centers = PyMem_Malloc((size_t)k * (size_t)d * sizeof(double));
    sums = PyMem_Malloc((size_t)k * (size_t)d * sizeof(double));
    counts = PyMem_Malloc((size_t)k * sizeof(Py_ssize_t));
    if (!points || !centers || !sums || !counts) { PyErr_NoMemory(); goto fail; }
    for (i = 0; i < n; ++i) {
        PyObject *seq = PySequence_Fast(PySequence_Fast_GET_ITEM(points_seq, i), "point must be a sequence");
        if (!seq) goto fail;
        if (PySequence_Fast_GET_SIZE(seq) != d) {
            Py_DECREF(seq);
            PyErr_SetString(PyExc_ValueError, "points must be rectangular");
            goto fail;
        }
        for (j = 0; j < d; ++j) {
            double val = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(seq, j));
            if (PyErr_Occurred()) { Py_DECREF(seq); goto fail; }
            if (!isfinite(val)) {
                Py_DECREF(seq);
                PyErr_SetString(PyExc_ValueError, "point coordinates must be finite");
                goto fail;
            }
            points[i*d+j] = val;
        }
        Py_DECREF(seq);
    }
    for (c = 0; c < k; ++c) {
        Py_ssize_t idx = PyLong_AsSsize_t(PySequence_Fast_GET_ITEM(indices_seq, c));
        if (idx == -1 && PyErr_Occurred()) goto fail;
        if (idx < 0 || idx >= n) {
            PyErr_SetString(PyExc_ValueError, "initial index out of bounds");
            goto fail;
        }
        for (i = 0; i < c; ++i) {
            Py_ssize_t prior = PyLong_AsSsize_t(PySequence_Fast_GET_ITEM(indices_seq, i));
            if (prior == idx) {
                PyErr_SetString(PyExc_ValueError, "initial indices must be distinct");
                goto fail;
            }
        }
        memcpy(centers+c*d, points+idx*d, (size_t)d * sizeof(double));
    }
    for (iteration = 0; iteration < max_iterations; ++iteration) {
        double max_shift_squared = 0.0;
        memset(sums, 0, (size_t)k * (size_t)d * sizeof(double));
        memset(counts, 0, (size_t)k * sizeof(Py_ssize_t));
        for (i = 0; i < n; ++i) {
            double best_dist = HUGE_VAL;
            best = 0;
            for (c = 0; c < k; ++c) {
                double dist = 0.0;
                for (j = 0; j < d; ++j) {
                    double delta = points[i*d+j] - centers[c*d+j];
                    dist += delta*delta;
                }
                if (dist < best_dist) { best_dist = dist; best = c; }
            }
            ++counts[best];
            for (j = 0; j < d; ++j) sums[best*d+j] += points[i*d+j];
        }
        for (c = 0; c < k; ++c) {
            double shift_squared = 0.0;
            if (!counts[c]) continue; /* preserve empty-cluster center */
            for (j = 0; j < d; ++j) {
                double updated = sums[c*d+j] / (double)counts[c];
                double delta = updated - centers[c*d+j];
                shift_squared += delta*delta;
                centers[c*d+j] = updated;
            }
            if (shift_squared > max_shift_squared) max_shift_squared = shift_squared;
        }
        if (sqrt(max_shift_squared) < epsilon) break;
    }
    result = PyList_New(k);
    if (!result) goto fail;
    for (c = 0; c < k; ++c) {
        row_obj = PyList_New(d);
        if (!row_obj) goto fail;
        for (j = 0; j < d; ++j) {
            PyObject *value = PyFloat_FromDouble(centers[c*d+j]);
            if (!value) goto fail;
            PyList_SET_ITEM(row_obj, j, value);
        }
        PyList_SET_ITEM(result, c, row_obj);
        row_obj = NULL;
    }
    Py_DECREF(points_seq); Py_DECREF(indices_seq);
    PyMem_Free(points); PyMem_Free(centers); PyMem_Free(sums); PyMem_Free(counts);
    return result;
fail:
    Py_XDECREF(row_obj); Py_XDECREF(result);
    Py_XDECREF(points_seq); Py_XDECREF(indices_seq);
    PyMem_Free(points); PyMem_Free(centers); PyMem_Free(sums); PyMem_Free(counts);
    return NULL;
}

static PyMethodDef methods[] = {
    {"fit", fit, METH_VARARGS, "Run K-means given initial indices: fit(iterations, eps, points, indices)."},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef module = { PyModuleDef_HEAD_INIT, "kmeans_module", NULL, -1, methods };
PyMODINIT_FUNC PyInit_kmeans_module(void) { return PyModule_Create(&module); }
