import random
from scipy.sparse import csr_matrix

def randmul(width, density, iterations):
    shape = (1,width)
    data = [1.0] * density
    ij0 = [0] * density
    sparse = [
        csr_matrix((data, (ij0, random.sample(xrange(width), density))), shape)
        for i in xrange(iterations) ]
    dense = [x.todense() for x in sparse]
    return sparse, dense

# sparse, dense = testing.randmul(r2r.shape[0], 1000, 50)
# %timeit -n1 for x in sparse: x * r2r;
# %timeit -n1 for x in dense: x * r2r;
#
# dense  7.0
#  1000  small
# 10000  1.6
# 30000  4.4
# 50000  6.9
