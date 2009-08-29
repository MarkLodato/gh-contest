from numpy cimport *
import numpy as _np
cimport cython

cdef extern from "stdlib.h":
    void qsort(void *, unsigned long, unsigned long, void*)

cdef int compar(void *_a, void *_b):
    cdef long *a = <long*>_a, *b = <long*>_b
    return b[1] - a[1]

def count_nested_a(ndarray[object, ndim=1] A not None, ignore=None):
    """count_nested_a(A, ignore=None) -> array"""

    cdef:
        unsigned long i, j, v, n, max_size
        ndarray[long, ndim=2, mode='c'] out

    d = count_nested(A)
    if ignore is not None:
        for x in ignore:
            d.pop(x, None)
    out = dict_to_array(d)
    qsort(<void*>out.data, out.shape[0], 2 * sizeof(long), <void*>compar)
    return out


def count_nested(ndarray[object, ndim=1] A not None):
    """count_nested(A) -> dict

    Return a dictionary mapping each element of the nested array A to the
    number of times it occurred.
    
    Parameters
    ----------

    A : ndarray
        A nested np.array of dtype object, with each element being an
        np.array with dtype int.
    """

    cdef:
        unsigned int i, j
        long v
        ndarray[long, ndim=1] B
        dict d

    d = {}
    setdefault = d.setdefault
    with cython.boundscheck(False):
        for i in range(A.shape[0]):
            B = A[i]
            for j in range(B.shape[0]):
                v = B[j]
                d[v] = setdefault(v,0) + 1
    return d


def dict_to_array(dict D):
    """dict_to_array(D)

    Convert an N-long dictionary to an Nx2 np.array of ints."""

    cdef:
        ndarray[long, ndim=2, mode='c'] out
        unsigned int i, size

    size = len(D)
    out = _np.empty((size,2), dtype=int)
    i = 0
    for k,v in D.iteritems():
        if i >= size:
            raise RuntimeError('len(D) not same as D.iteritems()')
        with cython.boundscheck(False):
            out[i,0] = k
            out[i,1] = v
        i += 1

    return out
