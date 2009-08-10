import numpy as np
import scipy.sparse

def save_sparse(filename, x):
    """Save the sparse matrix `x` to the given file, using `numpy.savez()`.
    Only CSR and CSC matrices are supported."""
    if isinstance(x, scipy.sparse.csr_matrix):
        np.savez(filename, type='csr', data=x.data, indices=x.indices,
                 indptr=x.indptr, shape=np.array(x.shape))
    elif isinstance(x, scipy.sparse.csc_matrix):
        np.savez(filename, type='csc', data=x.data, indices=x.indices,
                 indptr=x.indptr, shape=np.array(x.shape))
    else:
        raise ValueError('matrix format not supported')

def load_sparse(filename):
    """Load the sparse matrix from a file created by `save_sparse()`."""
    f = np.load(filename)
    type = f['type']
    if type == 'csr':
        constructor = scipy.sparse.csr_matrix
    elif type == 'csc':
        constructor = scipy.sparse.csc_matrix
    else:
        raise ValueError('matrix type "%s" not supported', type)
    return constructor((f['data'], f['indices'], f['indptr']), f['shape'])

