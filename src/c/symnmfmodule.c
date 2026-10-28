#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include "symnmf.h"

static Matrix *from_python(PyObject *object) {
    PyObject *rows = NULL, *row = NULL;
    Matrix *matrix = NULL;
    Py_ssize_t i, j, n, cols = -1;
    if (PyUnicode_Check(object) || PyBytes_Check(object)) {
        PyErr_SetString(PyExc_TypeError, "Expected a 2D sequence of numeric values");
        return NULL;
    }
    rows = PySequence_Fast(object, "Expected a 2D sequence");
    if (!rows) return NULL;
    n = PySequence_Fast_GET_SIZE(rows);
    if (n < 1) { PyErr_SetString(PyExc_ValueError, "Matrix must not be empty"); goto fail; }
    for (i = 0; i < n; ++i) {
        PyObject *obj = PySequence_Fast_GET_ITEM(rows, i);
        Py_ssize_t size;
        if (PyUnicode_Check(obj) || PyBytes_Check(obj)) {
            PyErr_SetString(PyExc_TypeError, "Matrix rows must be numeric sequences"); goto fail;
        }
        row = PySequence_Fast(obj, "Matrix rows must be sequences");
        if (!row) goto fail;
        size = PySequence_Fast_GET_SIZE(row);
        if (size < 1 || (cols != -1 && size != cols)) {
            PyErr_SetString(PyExc_ValueError, "Matrix must be nonempty and rectangular"); goto fail;
        }
        if (cols < 0) {
            cols = size;
            matrix = matrix_new((size_t)n, (size_t)cols);
            if (!matrix) { PyErr_NoMemory(); goto fail; }
        }
        for (j = 0; j < cols; ++j) {
            double x = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(row, j));
            if (PyErr_Occurred()) goto fail;
            if (!isfinite(x)) { PyErr_SetString(PyExc_ValueError, "Values must be finite"); goto fail; }
            matrix->data[(size_t)i * (size_t)cols + (size_t)j] = x;
        }
        Py_CLEAR(row);
    }
    Py_DECREF(rows);
    return matrix;
fail:
    Py_XDECREF(row); Py_DECREF(rows); matrix_free(matrix); return NULL;
}

static PyObject *to_python(const Matrix *m) {
    PyObject *rows, *row, *v;
    size_t i, j;
    if (!m) { PyErr_SetString(PyExc_RuntimeError, "Matrix operation failed"); return NULL; }
    rows = PyList_New((Py_ssize_t)m->rows);
    if (!rows) return NULL;
    for (i = 0; i < m->rows; ++i) {
        row = PyList_New((Py_ssize_t)m->cols);
        if (!row) { Py_DECREF(rows); return NULL; }
        PyList_SET_ITEM(rows, (Py_ssize_t)i, row);
        for (j = 0; j < m->cols; ++j) {
            v = PyFloat_FromDouble(m->data[i * m->cols + j]);
            if (!v) { Py_DECREF(rows); return NULL; }
            PyList_SET_ITEM(row, (Py_ssize_t)j, v);
        }
    }
    return rows;
}

static PyObject *matrix_operation(PyObject *args, int goal) {
    PyObject *obj, *result_obj;
    Matrix *points = NULL, *affinity = NULL, *result = NULL;
    if (!PyArg_ParseTuple(args, "O", &obj)) return NULL;
    points = from_python(obj);
    if (!points) return NULL;
    affinity = symnmf_affinity(points);
    if (affinity) {
        if (goal == 0) { result = affinity; affinity = NULL; }
        else if (goal == 1) result = symnmf_degree(affinity);
        else result = symnmf_normalize(affinity);
    }
    matrix_free(points);
    matrix_free(affinity);
    if (!result) { PyErr_NoMemory(); return NULL; }
    result_obj = to_python(result);
    matrix_free(result);
    return result_obj;
}

static PyObject *py_sym(PyObject *self, PyObject *args) { (void)self; return matrix_operation(args, 0); }
static PyObject *py_ddg(PyObject *self, PyObject *args) { (void)self; return matrix_operation(args, 1); }
static PyObject *py_norm(PyObject *self, PyObject *args) { (void)self; return matrix_operation(args, 2); }
static PyObject *py_symnmf(PyObject *self, PyObject *args) {
    PyObject *h_obj, *w_obj, *result_obj;
    Matrix *h = NULL, *w = NULL, *result = NULL;
    (void)self;
    if (!PyArg_ParseTuple(args, "OO", &h_obj, &w_obj)) return NULL;
    h = from_python(h_obj);
    if (!h) return NULL;
    w = from_python(w_obj);
    if (!w) { matrix_free(h); return NULL; }
    if (w->rows != w->cols || w->rows != h->rows) {
        matrix_free(h); matrix_free(w);
        PyErr_SetString(PyExc_ValueError, "W must be square and match H row count");
        return NULL;
    }
    result = symnmf_factorize(w, h, 300, 0.0001);
    matrix_free(h); matrix_free(w);
    if (!result) { PyErr_SetString(PyExc_ValueError, "SymNMF failed; check nonnegative input and numerical stability"); return NULL; }
    result_obj = to_python(result);
    matrix_free(result);
    return result_obj;
}

static PyMethodDef methods[] = {
    {"sym", py_sym, METH_VARARGS, "Return Gaussian affinity matrix for points."},
    {"ddg", py_ddg, METH_VARARGS, "Return diagonal degree matrix for points."},
    {"norm", py_norm, METH_VARARGS, "Return normalized similarity matrix for points."},
    {"symnmf", py_symnmf, METH_VARARGS, "Factorize W using a supplied initial H."},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "symnmfmodule", NULL, -1, methods};
PyMODINIT_FUNC PyInit_symnmfmodule(void) { return PyModule_Create(&module); }
