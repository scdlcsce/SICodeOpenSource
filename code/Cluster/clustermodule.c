#include "Python.h"
#include <stdio.h>
#include <string.h>
#include <float.h>
#include "cluster.h"


/* ========================================================================= */
/* -- Helper routines ------------------------------------------------------ */
/* ========================================================================= */

static char
extract_single_character(PyObject* object, const char variable[],
                         const char allowed[])
{
    Py_UCS4 ch;
    Py_ssize_t n;
    if (!PyUnicode_Check(object)) {
        PyErr_Format(PyExc_ValueError, "%s should be a string", variable);
        return 0;
    }
    if (PyUnicode_READY(object) == -1) return 0;
    n = PyUnicode_GET_LENGTH(object);
    if (n != 1) {
        PyErr_Format(PyExc_ValueError,
                     "%s should be a single character", variable);
        return 0;
    }
    ch = PyUnicode_READ_CHAR(object, 0);
    if (ch < 128) {
        const char c = ch;
        if (strchr(allowed, c)) return c;
    }
    PyErr_Format(PyExc_ValueError,
                 "unknown %s function specified (should be one of '%s')",
                 variable, allowed);
    return 0;
}

static int
distance_converter(PyObject* object, void* pointer)
{
    char c;

    c = extract_single_character(object, "dist", "ebcauxsk");
    if (c == 0) return 0;
    *((char*)pointer) = c;
    return 1;
}

static int
method_kcluster_converter(PyObject* object, void* pointer)
{
    char c;

    c = extract_single_character(object, "method", "am");
    if (c == 0) return 0;
    *((char*)pointer) = c;
    return 1;
}

/* -- data ----------------------------------------------------------------- */

typedef struct {
    int nrows;
    int ncols;
    double** values;
    Py_buffer view;
} Data;

static int
data_converter(PyObject* object, void* pointer)
{
    Data* data = pointer;
    int nrows;
    int ncols;
    int i;
    double** values = data->values;
    Py_buffer* view = &data->view;
    const char* p;
    Py_ssize_t stride;
    const int flag = PyBUF_ND | PyBUF_STRIDES;

    if (object == NULL) goto exit;
    if (object == Py_None) return 1;

    if (PyObject_GetBuffer(object, view, flag) == -1) {
        PyErr_SetString(PyExc_RuntimeError,
                        "data matrix has unexpected format.");
        return 0;
    }

    if (view->ndim != 2) {
        PyErr_Format(PyExc_RuntimeError,
                     "data matrix has incorrect rank %d (expected 2)",
                     view->ndim);
        goto exit;
    }
    if (view->itemsize != sizeof(double)) {
        PyErr_SetString(PyExc_RuntimeError,
                        "data matrix has incorrect data type");
        goto exit;
    }
    nrows = (int) view->shape[0];
    ncols = (int) view->shape[1];
    if (nrows != view->shape[0] || ncols != view->shape[1]) {
        PyErr_Format(PyExc_ValueError,
            "data matrix is too large (dimensions = %zd x %zd)",
            view->shape[0], view->shape[1]);
        goto exit;
    }
    if (nrows < 1 || ncols < 1) {
        PyErr_SetString(PyExc_ValueError, "data matrix is empty");
        goto exit;
    }
    stride = view->strides[0];
    if (view->strides[1] != view->itemsize) {
        PyErr_SetString(PyExc_RuntimeError, "data is not contiguous");
        goto exit;
    }
    values = PyMem_Malloc(nrows*sizeof(double*));
    if (!values) {
        PyErr_NoMemory();
        goto exit;
    }
    for (i = 0, p = view->buf; i < nrows; i++, p += stride)
        values[i] = (double*)p;
    data->values = values;
    data->nrows = nrows;
    data->ncols = ncols;
    return Py_CLEANUP_SUPPORTED;

exit:
    if (values) PyMem_Free(values);
    PyBuffer_Release(view);
    return 0;
}

/* -- mask ----------------------------------------------------------------- */

typedef struct {
    int** values;
    Py_buffer view;
} Mask;

static int
mask_converter(PyObject* object, void* pointer)
{
    Mask* mask = pointer;
    int nrows;
    int ncols;
    int i;
    int** values = mask->values;
    Py_buffer* view = &mask->view;
    const char* p;
    Py_ssize_t stride;
    const int flag = PyBUF_ND | PyBUF_STRIDES;

    if (object == NULL) goto exit;
    if (object == Py_None) return 1;

    if (PyObject_GetBuffer(object, view, flag) == -1) {
        PyErr_SetString(PyExc_RuntimeError, "mask has unexpected format.");
        return 0;
    }

    if (view->ndim != 2) {
        PyErr_Format(PyExc_ValueError,
                     "mask has incorrect rank %d (expected 2)", view->ndim);
        goto exit;
    }
    if (view->itemsize != sizeof(int)) {
        PyErr_SetString(PyExc_RuntimeError, "mask has incorrect data type");
        goto exit;
    }
    nrows = (int) view->shape[0];
    ncols = (int) view->shape[1];
    if (nrows != view->shape[0] || ncols != view->shape[1]) {
        PyErr_Format(PyExc_ValueError,
                     "mask is too large (dimensions = %zd x %zd)",
                     view->shape[0], view->shape[1]);
        goto exit;
    }
    stride = view->strides[0];
    if (view->strides[1] != view->itemsize) {
        PyErr_SetString(PyExc_RuntimeError, "mask is not contiguous");
        goto exit;
    }
    values = PyMem_Malloc(nrows*sizeof(int*));
    if (!values) {
        PyErr_NoMemory();
        goto exit;
    }
    for (i = 0, p = view->buf; i < nrows; i++, p += stride)
        values[i] = (int*)p;
    mask->values = values;
    return Py_CLEANUP_SUPPORTED;

exit:
    if (values) PyMem_Free(values);
    PyBuffer_Release(view);
    return 0;
}

/* -- 1d array ------------------------------------------------------------- */

static int
vector_converter(PyObject* object, void* pointer)
{
    Py_buffer* view = pointer;
    int ndata;
    const int flag = PyBUF_ND | PyBUF_C_CONTIGUOUS;

    if (object == NULL) goto exit;

    if (PyObject_GetBuffer(object, view, flag) == -1) {
        PyErr_SetString(PyExc_RuntimeError, "unexpected format.");
        return 0;
    }

    if (view->ndim != 1) {
        PyErr_Format(PyExc_ValueError, "incorrect rank %d (expected 1)",
                     view->ndim);
        goto exit;
    }
    if (view->itemsize != sizeof(double)) {
        PyErr_SetString(PyExc_RuntimeError, "array has incorrect data type");
        goto exit;
    }
    ndata = (int) view->shape[0];
    if (ndata != view->shape[0]) {
        PyErr_Format(PyExc_ValueError,
                     "array is too large (size = %zd)", view->shape[0]);
        goto exit;
    }
    return Py_CLEANUP_SUPPORTED;

exit:
    PyBuffer_Release(view);
    return 0;
}

/* -- clusterid ------------------------------------------------------------ */

static int
check_clusterid(Py_buffer clusterid, int nitems) {
    int i, j;
    int *p = clusterid.buf;
    int nclusters = 0;
    int* number;

    if (nitems != clusterid.shape[0]) {
        PyErr_Format(PyExc_ValueError, "incorrect size (%zd, expected %d)",
                     clusterid.shape[0], nitems);
        return 0;
    }
    for (i = 0; i < nitems; i++) {
        j = p[i];
        if (j > nclusters) nclusters = j;
        if (j < 0) {
            PyErr_SetString(PyExc_ValueError, "negative cluster number found");
            return 0;
        }
    }
    nclusters++;
    /* -- Count the number of items in each cluster --------------------- */
    number = calloc(nclusters, sizeof(int));
    if (!number) {
        PyErr_NoMemory();
        return 0;
    }
    for (i = 0; i < nitems; i++) {
        j = p[i];
        number[j]++;
    }
    for (j = 0; j < nclusters; j++) if (number[j] == 0) break;
    PyMem_Free(number);
    if (j < nclusters) {
        PyErr_Format(PyExc_ValueError, "cluster %d is empty", j);
        return 0;
    }
    return nclusters;
}

/* -- distance ----------------------------------------------------------- */

typedef struct {
    int n;
    double** values;
    Py_buffer* views;
    Py_buffer view;
} Distancematrix;


/* -- celldata ------------------------------------------------------------- */

typedef struct {
    int nx;
    int ny;
    int nz;
    double*** values;
    Py_buffer view;
} Celldata;


/* -- index ---------------------------------------------------------------- */

static int
index_converter(PyObject* argument, void* pointer)
{
    Py_buffer* view = pointer;
    int n;
    const int flag = PyBUF_ND | PyBUF_C_CONTIGUOUS;

    if (argument == NULL) goto exit;

    if (PyObject_GetBuffer(argument, view, flag) == -1) {
        PyErr_SetString(PyExc_RuntimeError, "unexpected format.");
        return 0;
    }

    if (view->ndim != 1) {
        PyErr_Format(PyExc_ValueError, "incorrect rank %d (expected 1)",
                     view->ndim);
        goto exit;
    }
    if (view->itemsize != sizeof(int)) {
        PyErr_SetString(PyExc_RuntimeError,
                        "argument has incorrect data type");
        goto exit;
    }
    n = (int) view->shape[0];
    if (n != view->shape[0]) {
        PyErr_Format(PyExc_ValueError,
            "array size is too large (size = %zd)", view->shape[0]);
        goto exit;
    }
    return Py_CLEANUP_SUPPORTED;

exit:
    PyBuffer_Release(view);
    return 0;
}

static int
centers_converter(PyObject* argument, void* pointer)
{
    Py_buffer* view = pointer;
    int n;
    const int flag = PyBUF_ND | PyBUF_C_CONTIGUOUS;

    if (argument == NULL) goto exit;

    if (PyObject_GetBuffer(argument, view, flag) == -1) {
        PyErr_SetString(PyExc_RuntimeError, "unexpected format.");
        return 0;
    }

    if (view->ndim != 1) {
        PyErr_Format(PyExc_ValueError, "incorrect rank %d (expected 1)",
                     view->ndim);
        goto exit;
    }
    if (view->itemsize != sizeof(double)) {
        PyErr_SetString(PyExc_RuntimeError,
                        "argument has incorrect data type");
        goto exit;
    }
    n = (int) view->shape[0];
    if (n != view->shape[0]) {
        PyErr_Format(PyExc_ValueError,
            "array size is too large (size = %zd)", view->shape[0]);
        goto exit;
    }
    return Py_CLEANUP_SUPPORTED;

exit:
    PyBuffer_Release(view);
    return 0;
}


/* -- index2d ------------------------------------------------------------- */

/* ========================================================================= */
/* -- Classes -------------------------------------------------------------- */
/* ========================================================================= */

typedef struct {
    PyObject_HEAD
    Node node;
} PyNode;

static int
PyNode_init(PyNode *self, PyObject *args, PyObject *kwds)
{
    int left, right;
    double distance = 0.0;
    static char *kwlist[] = {"left", "right", "distance", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ii|d", kwlist,
                                      &left, &right, &distance))
        return -1;
    self->node.left = left;
    self->node.right = right;
    self->node.distance = distance;
    return 0;
}

static PyObject*
PyNode_repr(PyNode* self)
{
    char string[64];

    sprintf(string, "(%d, %d): %g",
                   self->node.left, self->node.right, self->node.distance);
    return PyUnicode_FromString(string);
}

static char PyNode_left__doc__[] =
"integer representing the first member of this node";

static PyObject*
PyNode_getleft(PyNode* self, void* closure)
{
    int left = self->node.left;

    return PyLong_FromLong((long)left);
}

static int
PyNode_setleft(PyNode* self, PyObject* value, void* closure)
{
    long left = PyLong_AsLong(value);

    if (PyErr_Occurred()) return -1;
    self->node.left = (int) left;
    return 0;
}

static char PyNode_right__doc__[] =
"integer representing the second member of this node";

static PyObject*
PyNode_getright(PyNode* self, void* closure)
{
    int right = self->node.right;

    return PyLong_FromLong((long)right);
}

static int
PyNode_setright(PyNode* self, PyObject* value, void* closure)
{
    long right = PyLong_AsLong(value);

    if (PyErr_Occurred()) return -1;
    self->node.right = (int) right;
    return 0;
}

static PyObject*
PyNode_getdistance(PyNode* self, void* closure)
{
    return PyFloat_FromDouble(self->node.distance);
}

static int
PyNode_setdistance(PyNode* self, PyObject* value, void* closure)
{
    const double distance = PyFloat_AsDouble(value);

    if (PyErr_Occurred()) return -1;
    self->node.distance = distance;
    return 0;
}

static char PyNode_distance__doc__[] =
"the distance between the two members of this node\n";

static PyGetSetDef PyNode_getset[] = {
    {"left",
     (getter)PyNode_getleft,
     (setter)PyNode_setleft,
     PyNode_left__doc__, NULL},
    {"right",
     (getter)PyNode_getright,
     (setter)PyNode_setright,
     PyNode_right__doc__, NULL},
    {"distance",
     (getter)PyNode_getdistance,
     (setter)PyNode_setdistance,
     PyNode_distance__doc__, NULL},
    {NULL}  /* Sentinel */
};

static char PyNode_doc[] =
"A Node object describes a single node in a hierarchical clustering tree.\n"
"The integer attributes 'left' and 'right' represent the two members that\n"
"make up this node; the floating point attribute 'distance' contains the\n"
"distance between the two members of this node.\n";

static PyTypeObject PyNodeType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_cluster.Node",           /* tp_name */
    sizeof(PyNode),            /* tp_basicsize */
    0,                         /* tp_itemsize */
    0,                         /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_compare */
    (reprfunc)PyNode_repr,     /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,          /*tp_flags*/
    PyNode_doc,                /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    0,                         /* tp_methods */
    0,                         /* tp_members */
    PyNode_getset,             /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)PyNode_init,     /* tp_init */
};

typedef struct {
    PyObject_HEAD
    Node* nodes;
    int n;
} PyTree;

static void
PyTree_dealloc(PyTree* self)
{
    if (self->n) PyMem_Free(self->nodes);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject*
PyTree_new(PyTypeObject *type, PyObject* args, PyObject* kwds)
{
    int i, j;
    int n;
    Node* nodes;
    PyObject* arg = NULL;
    int* flag;
    PyTree* self;

    self = (PyTree *)type->tp_alloc(type, 0);
    if (!self) return NULL;

    if (!PyArg_ParseTuple(args, "|O", &arg)) {
        Py_DECREF(self);
        return NULL;
    }

    if (arg == NULL) {
        self->n = 0;
        self->nodes = NULL;
        return (PyObject*)self;
    }

    if (!PyList_Check(arg)) {
        Py_DECREF(self);
        PyErr_SetString(PyExc_TypeError,
                        "Argument should be a list of Node objects");
        return NULL;
    }

    n = (int) PyList_GET_SIZE(arg);
    if (n != PyList_GET_SIZE(arg)) {
        Py_DECREF(self);
        PyErr_Format(PyExc_ValueError,
                     "List is too large (size = %zd)", PyList_GET_SIZE(arg));
        return NULL;
    }
    if (n < 1) {
        Py_DECREF(self);
        PyErr_SetString(PyExc_ValueError, "List is empty");
        return NULL;
    }
    nodes = PyMem_Malloc(n*sizeof(Node));
    if (!nodes) {
        Py_DECREF(self);
        return PyErr_NoMemory();
    }
    for (i = 0; i < n; i++) {
        PyNode* p;
        PyObject* row = PyList_GET_ITEM(arg, i);
        if (!PyType_IsSubtype(Py_TYPE(row), &PyNodeType)) {
            PyMem_Free(nodes);
            Py_DECREF(self);
            PyErr_Format(PyExc_TypeError,
                         "Row %d in list is not a Node object", i);
            return NULL;
        }
        p = (PyNode*)row;
        nodes[i] = p->node;
    }
    /* --- Check if this is a bona fide tree ------------------------------- */
    flag = PyMem_Malloc((2*n+1)*sizeof(int));
    if (!flag) {
        PyMem_Free(nodes);
        Py_DECREF(self);
        return PyErr_NoMemory();
    }
    for (i = 0; i < 2*n+1; i++) flag[i] = 0;
    for (i = 0; i < n; i++) {
        j = nodes[i].left;
        if (j < 0) {
            j = -j-1;
            if (j >= i) break;
        }
        else j += n;
        if (flag[j]) break;
        flag[j] = 1;
        j = nodes[i].right;
        if (j < 0) {
          j = -j-1;
          if (j >= i) break;
        }
        else j += n;
        if (flag[j]) break;
        flag[j] = 1;
    }
    PyMem_Free(flag);
    if (i < n) {
        /* break encountered */
        PyMem_Free(nodes);
        Py_DECREF(self);
        PyErr_SetString(PyExc_ValueError, "Inconsistent tree");
        return NULL;
    }
    self->n = n;
    self->nodes = nodes;
    return (PyObject*)self;
}

static PyObject*
PyTree_str(PyTree* self)
{
    int i;
    const int n = self->n;
    char string[128];
    Node node;
    PyObject* line;
    PyObject* output;
    PyObject* temp;

    output = PyUnicode_FromString("");
    for (i = 0; i < n; i++) {
        node = self->nodes[i];
        sprintf(string, "(%d, %d): %g", node.left, node.right, node.distance);
        if (i < n-1) strcat(string, "\n");
        line = PyUnicode_FromString(string);
        if (!line) {
            Py_DECREF(output);
            return NULL;
        }
        temp = PyUnicode_Concat(output, line);
        if (!temp) {
            Py_DECREF(output);
            Py_DECREF(line);
            return NULL;
        }
        output = temp;
    }
    return output;
}

static int
PyTree_length(PyTree *self)
{
    return self->n;
}

static PyObject*
PyTree_subscript(PyTree* self, PyObject* item)
{
    if (PyIndex_Check(item)) {
        PyNode* result;
        Py_ssize_t i;
        i = PyNumber_AsSsize_t(item, PyExc_IndexError);
        if (i == -1 && PyErr_Occurred())
            return NULL;
        if (i < 0)
            i += self->n;
        if (i < 0 || i >= self->n) {
            PyErr_SetString(PyExc_IndexError, "tree index out of range");
            return NULL;
        }
        result = (PyNode*) PyNodeType.tp_alloc(&PyNodeType, 0);
        if (!result) return PyErr_NoMemory();
        result->node = self->nodes[i];
        return (PyObject*) result;
    }
    else if (PySlice_Check(item)) {
        Py_ssize_t i, j;
        Py_ssize_t start, stop, step, slicelength;
        if (PySlice_GetIndicesEx(item, self->n, &start, &stop, &step,
                                 &slicelength) == -1) return NULL;
        if (slicelength == 0) return PyList_New(0);
        else {
            PyNode* node;
            PyObject* result = PyList_New(slicelength);
            if (!result) return PyErr_NoMemory();
            for (i = 0, j = start; i < slicelength; i++, j += step) {
                node = (PyNode*) PyNodeType.tp_alloc(&PyNodeType, 0);
                if (!node) {
                    Py_DECREF(result);
                    return PyErr_NoMemory();
                }
                node->node = self->nodes[j];
                PyList_SET_ITEM(result, i, (PyObject*)node);
            }
            return result;
        }
    }
    else {
        PyErr_Format(PyExc_TypeError,
                     "tree indices must be integers, not %.200s",
                     item->ob_type->tp_name);
        return NULL;
    }
}

static PyMappingMethods PyTree_mapping = {
    (lenfunc)PyTree_length,           /* mp_length */
    (binaryfunc)PyTree_subscript,     /* mp_subscript */
};

static char PyTree_scale__doc__[] =
"mytree.scale()\n"
"\n"
"Scale the node distances in the tree such that they are all between one\n"
"and zero.\n";

static PyObject*
PyTree_scale(PyTree* self)
{
    int i;
    const int n = self->n;
    Node* nodes = self->nodes;
    double maximum = DBL_MIN;

    for (i = 0; i < n; i++) {
        double distance = nodes[i].distance;
        if (distance > maximum) maximum = distance;
    }
    if (maximum != 0.0)
        for (i = 0; i < n; i++) nodes[i].distance /= maximum;
    Py_INCREF(Py_None);
    return Py_None;
}

static char PyTree_cut__doc__[] =
"mytree.cut(nclusters) -> array\n"
"\n"
"Divide the elements in a hierarchical clustering result mytree into\n"
"clusters, and return an array with the number of the cluster to which each\n"
"element was assigned. The number of clusters is given by nclusters.\n";

static PyObject*
PyTree_cut(PyTree* self, PyObject* args)
{
    int ok = -1;
    int nclusters;
    const int n = self->n + 1;
    Py_buffer indices = {0};

    if (!PyArg_ParseTuple(args, "O&i",
                          index_converter, &indices, &nclusters)) goto exit;
    if (nclusters < 1) {
        PyErr_SetString(PyExc_ValueError,
                        "requested number of clusters should be positive");
        goto exit;
    }
    if (nclusters > n) {
        PyErr_SetString(PyExc_ValueError,
                        "more clusters requested than items available");
        goto exit;
    }
    if (indices.shape[0] != n) {
        PyErr_SetString(PyExc_RuntimeError,
                        "indices array inconsistent with tree");
        goto exit;
    }
    ok = cuttree(n, self->nodes, nclusters, indices.buf);

exit:
    index_converter(NULL, &indices);
    if (ok == -1) return NULL;
    if (ok == 0) return PyErr_NoMemory();
    Py_INCREF(Py_None);
    return Py_None;
}

static char PyTree_sort__doc__[] =
"mytree.sort(order) -> array\n"
"\n"
"Sort a hierarchical clustering tree by switching the left and right\n"
"subnode of nodes such that the elements in the left-to-right order of the\n"
"tree tend to have increasing order values.\n"
"\n"
"Return the indices of the elements in the left-to-right order in the\n"
"hierarchical clustering tree, such that the element with index indices[i]\n"
"occurs at position i in the dendrogram.\n";

static PyObject*
PyTree_sort(PyTree* self, PyObject* args)
{
    int ok = -1;
    Py_buffer indices = {0};
    const int n = self->n;
    Py_buffer order = {0};

    if (n == 0) {
        PyErr_SetString(PyExc_ValueError, "tree is empty");
        return NULL;
    }
    if (!PyArg_ParseTuple(args, "O&O&",
                          index_converter, &indices,
                          vector_converter, &order)) goto exit;
    if (indices.shape[0] != n + 1) {
        PyErr_SetString(PyExc_RuntimeError,
                        "indices array inconsistent with tree");
        goto exit;
    }
    if (order.shape[0] != n + 1) {
        PyErr_Format(PyExc_ValueError,
            "order array has incorrect size %zd (expected %d)",
            order.shape[0], n + 1);
        goto exit;
    }
    ok = sorttree(n, self->nodes, order.buf, indices.buf);
exit:
    index_converter(NULL, &indices);
    vector_converter(NULL, &order);
    if (ok == -1) return NULL;
    if (ok == 0) return PyErr_NoMemory();
    Py_INCREF(Py_None);
    return Py_None;
}

static PyMethodDef PyTree_methods[] = {
    {"scale", (PyCFunction)PyTree_scale, METH_NOARGS, PyTree_scale__doc__},
    {"cut", (PyCFunction)PyTree_cut, METH_VARARGS, PyTree_cut__doc__},
    {"sort", (PyCFunction)PyTree_sort, METH_VARARGS, PyTree_sort__doc__},
    {NULL}  /* Sentinel */
};

static char PyTree_doc[] =
"Tree objects store a hierarchical clustering solution.\n"
"Individual nodes in the tree can be accessed with tree[i], where i is\n"
"an integer. Whereas the tree itself is a read-only object, tree[:]\n"
"returns a list of all the nodes, which can then be modified. To create\n"
"a new Tree from this list, use Tree(list).\n"
"See the description of the Node class for more information.";

static PyTypeObject PyTreeType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_cluster.Tree",             /* tp_name */
    sizeof(PyTree),              /* tp_basicsize */
    0,                           /* tp_itemsize */
    (destructor)PyTree_dealloc,  /* tp_dealloc */
    0,                           /* tp_print */
    0,                           /* tp_getattr */
    0,                           /* tp_setattr */
    0,                           /* tp_compare */
    0,                           /* tp_repr */
    0,                           /* tp_as_number */
    0,                           /* tp_as_sequence */
    &PyTree_mapping,             /* tp_as_mapping */
    0,                           /* tp_hash */
    0,                           /* tp_call */
    (reprfunc)PyTree_str,        /* tp_str */
    0,                           /* tp_getattro */
    0,                           /* tp_setattro */
    0,                           /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,          /*tp_flags*/
    PyTree_doc,                  /* tp_doc */
    0,                           /* tp_traverse */
    0,                           /* tp_clear */
    0,                           /* tp_richcompare */
    0,                           /* tp_weaklistoffset */
    0,                           /* tp_iter */
    0,                           /* tp_iternext */
    PyTree_methods,              /* tp_methods */
    NULL,                        /* tp_members */
    0,                           /* tp_getset */
    0,                           /* tp_base */
    0,                           /* tp_dict */
    0,                           /* tp_descr_get */
    0,                           /* tp_descr_set */
    0,                           /* tp_dictoffset */
    0,                           /* tp_init */
    0,                           /* tp_alloc */
    (newfunc)PyTree_new,         /* tp_new */
};

/* ========================================================================= */
/* -- Methods -------------------------------------------------------------- */
/* ========================================================================= */

/* version */
static char version__doc__[] =
"version() -> string\n"
"\n"
"Return the version number of the C Clustering Library as a string.\n";

static PyObject*
py_version(PyObject* self)
{
    return PyUnicode_FromString( CLUSTERVERSION );
}

/* kcluster */
static char kcluster__doc__[] =
"kcluster(data, nclusters, mask, weight, transpose, npass, method,\n"
"         dist, clusterid, centers) -> None\n"
"\n"
"This function implements k-means clustering.\n"
"\n"
"Arguments:\n"
"\n"
" - data: nrows x ncols array containing the data to be clustered\n"
"\n"
" - nclusters: number of clusters (the 'k' in k-means)\n"
"\n"
" - mask: nrows x ncols array of integers, showing which data are\n"
"   missing. If mask[i,j] == 0, then data[i,j] is missing.\n"
"\n"
" - weight: the weights to be used when calculating distances\n"
" - transpose:\n"
"\n"
"   - if equal to 0, rows are clustered;\n"
"   - if equal to 1, columns are clustered.\n"
"\n"
" - npass: number of times the k-means clustering algorithm is\n"
"   performed, each time with a different (random) initial\n"
"   condition. If npass == 0, then the assignments in clusterid\n"
"   are used as the initial condition.\n"
"\n"
" - method: specifies how the center of a cluster is found:\n"
"\n"
"   - method == 'a': arithmetic mean\n"
"   - method == 'm': median\n"
"\n"
" - dist: specifies the distance function to be used:\n"
"\n"
"   - dist == 'e': Euclidean distance\n"
"   - dist == 'b': City Block distance\n"
"   - dist == 'c': Pearson correlation\n"
"   - dist == 'a': absolute value of the correlation\n"
"   - dist == 'u': uncentered correlation\n"
"   - dist == 'x': absolute uncentered correlation\n"
"   - dist == 's': Spearman's rank correlation\n"
"   - dist == 'k': Kendall's tau\n"
"\n"
" - clusterid: array in which the final clustering solution will be\n"
"   stored (output variable). If npass == 0, then clusterid is also used\n"
"   as an input variable, containing the initial condition from which\n"
"   the EM algorithm should start. In this case, the k-means algorithm\n"
"   is fully deterministic.\n"
"\n"
" - centers: howmanycenters.\n"
"\n";

static PyObject*
py_kcluster(PyObject* self, PyObject* args, PyObject* keywords)
{
    int nclusters = 2;
    int nrows, ncols;
    int nitems;
    int ndata;
    Data data = {0};
    Mask mask = {0};
    Py_buffer weight = {0};
    int transpose = 0;
    int npass = 1;
    char method = 'a';
    char dist = 'e';
    Py_buffer clusterid = {0};
    Py_buffer centers = {0};
    // double* centers;
    double error;
    int ifound = 0;

    static char* kwlist[] = {"data",
                             "nclusters",
                             "mask",
                             "weight",
                             "transpose",
                             "npass",
                             "method",
                             "dist",
                             "clusterid",
                             "centers",
                              NULL};

    if (!PyArg_ParseTupleAndKeywords(args, keywords, "O&iO&O&iiO&O&O&O&", kwlist,
                                     data_converter, &data,
                                     &nclusters,
                                     mask_converter, &mask,
                                     vector_converter, &weight,
                                     &transpose,
                                     &npass,
                                     method_kcluster_converter, &method,
                                     distance_converter, &dist,
                                     index_converter, &clusterid,
                                     centers_converter, &centers)) return NULL;
    if (!data.values) {
        PyErr_SetString(PyExc_RuntimeError, "data is None");
        goto exit;
    }
    if (!mask.values) {
        PyErr_SetString(PyExc_RuntimeError, "mask is None");
        goto exit;
    }
    if (data.nrows != mask.view.shape[0] ||
        data.ncols != mask.view.shape[1]) {
        PyErr_Format(PyExc_ValueError,
            "mask has incorrect dimensions %zd x %zd (expected %d x %d)",
            mask.view.shape[0], mask.view.shape[1], data.nrows, data.ncols);
        goto exit;
    }
    nrows = data.nrows;
    ncols = data.ncols;
    ndata = transpose ? nrows : ncols;
    nitems = transpose ? ncols : nrows;

    if (weight.shape[0] != ndata) {
        PyErr_Format(PyExc_ValueError,
                     "weight has incorrect size %zd (expected %d)",
                     weight.shape[0], ndata);
        goto exit;
    }
    if (nclusters < 1) {
        PyErr_SetString(PyExc_ValueError, "nclusters should be positive");
        goto exit;
    }
    if (nitems < nclusters) {
        PyErr_SetString(PyExc_ValueError,
                        "more clusters than items to be clustered");
        goto exit;
    }
    if (npass < 0) {
        PyErr_SetString(PyExc_RuntimeError, "expected a non-negative integer");
        goto exit;
    }
    else if (npass == 0) {
        int n = check_clusterid(clusterid, nitems);
        if (n == 0) goto exit;
        if (n != nclusters) {
            PyErr_SetString(PyExc_ValueError,
                            "more clusters requested than found in clusterid");
            goto exit;
        }
    }
    // centers = (double *)malloc(nclusters * ndata);
    kcluster(nclusters,
             nrows,
             ncols,
             data.values,
             mask.values,
             weight.buf,
             transpose,
             npass,
             method,
             dist,
             clusterid.buf,
             centers.buf,
             &error,
             &ifound);
exit:
    data_converter(NULL, &data);
    mask_converter(NULL, &mask);
    vector_converter(NULL, &weight);
    index_converter(NULL, &clusterid);
    centers_converter(NULL, &centers);
    if (ifound) return Py_BuildValue("di", error, ifound);
    return NULL;
}
/* end of wrapper for kcluster */


/* ========================================================================= */
/* -- The methods table ---------------------------------------------------- */
/* ========================================================================= */


static struct PyMethodDef cluster_methods[] = {
    {"version", (PyCFunction) py_version, METH_NOARGS, version__doc__},
    {"kcluster",
     (PyCFunction) py_kcluster,
     METH_VARARGS | METH_KEYWORDS,
     kcluster__doc__
    },
    
    {NULL, NULL, 0, NULL} /* sentinel */
};

/* ========================================================================= */
/* -- Initialization ------------------------------------------------------- */
/* ========================================================================= */

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_cluster",
    "C Clustering Library",
    -1,
    cluster_methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyObject *
PyInit__cluster(void)
{
    PyObject *module;

    PyNodeType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&PyNodeType) < 0)
        return NULL;
    if (PyType_Ready(&PyTreeType) < 0)
        return NULL;

    module = PyModule_Create(&moduledef);
    if (module == NULL) return NULL;

    Py_INCREF(&PyTreeType);
    if (PyModule_AddObject(module, "Tree", (PyObject*) &PyTreeType) < 0) {
        Py_DECREF(module);
        Py_DECREF(&PyTreeType);
        return NULL;
    }

    Py_INCREF(&PyNodeType);
    if (PyModule_AddObject(module, "Node", (PyObject*) &PyNodeType) < 0) {
        Py_DECREF(module);
        Py_DECREF(&PyNodeType);
        return NULL;
    }

    return module;
}
