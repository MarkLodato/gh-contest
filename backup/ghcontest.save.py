#!/usr/bin/python

from __future__ import print_function, division
__metaclass__ = type

# Copyright 2009, Mark Lodato
#
# August 31, 2009 and before:
#
#   All rights reserved.  You may not copy, with or without modification, any
#   of the code in this project.
#
# September 1, 2009 and after:
#
#   MIT License:
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#   THE SOFTWARE.

import itertools
import numpy as np

def read_data(filename):
    with open(filename) as f:
        pairs = [map(int, line.rstrip('\n\r').split(':')) for line in f]
    num_users = max(x[0] for x in pairs) + 1
    num_repos = max(x[1] for x in pairs) + 1
    u2r = [[] for i in range(num_users)]
    r2u = [[] for i in range(num_repos)]
    for u,r in pairs:
        u2r[u].append(r)
        r2u[r].append(u)
    return u2r, r2u

def read_test(filename):
    with open(filename) as f:
        return [int(x) for x in f]

def read_repos(filename):
    r2from = {}
    with open(filename) as f:
        for line in f:
            r,v = line.rstrip('\n\r').split(':',1)
            v = v.split(',')
            if len(v) == 3:
                name, date, forked_from = v
                r2from[int(r)] = int(forked_from)
    return r2from

def setup_matrix(u2r, r2u, stop=None):
    out = []
    for repo,users in enumerate(r2u):
        print(repo)
        tmp = count( r   for user in users   for r in u2r[user] )
        tmp.pop(repo, None)
        tmp = tmp.items()
        tmp.sort(reverse=True, key=lambda x: x[1])
        out.append(tmp)
        if repo == stop:
            break
    return out

def count_sum(iterable):
    d = {}
    for k,v in iterable:
        try:
            d[k] += v
        except KeyError:
            d[k] = v
    return d

def count(iterable):
    return count_sum(itertools.izip(iterable, itertools.repeat(1)))

def suggest(user, u2r, r2u, r2from, n=None, cutoff=None):
    repos = u2r[user]
    all_repos = list(repos)

    if r2from is not None:
        for r in repos:
            try:
                f = r2from[r]
            except KeyError:
                pass
            else:
                all_repos.append(f)

    other_users = count_sum( (u,1.0)
            for r in all_repos
            for u in r2u[r])
    other_users.pop(user, None)

    if cutoff is not None:
        other_users = other_users.items()
        other_users.sort(reverse=True, key=lambda x: x[1])
        other_users = dict(other_users[:cutoff])

    other_repos = count_sum( (r,v)
            for u,v in other_users.iteritems()
            for r in u2r[u] )
    for r in repos:
        other_repos.pop(r, None)

    results = other_repos.items()
    results.sort(reverse=True, key=lambda x: x[1])

    if n is None:
        return results
    else:
        return [x[0] for x in results[:n]]

def run_suggest(users, u2r, r2u, r2from, n=10, cutoff=100):
    results = {}
    for u in users:
        print(u)
        results[u] = suggest(u, u2r, r2u, r2from, n, cutoff)
    return results

def save_suggest(filename, results):
    with open(filename, 'w') as f:
        for k,v in sorted(results.iteritems()):
            f.write('%s:%s\n' % (k, ','.join(map(str,v))))


# ------- numpy implementation --------


from _ghcontest import *

def read_data(filename):
    data = np.loadtxt(filename, delimiter=':', dtype=int)
    return data

def parse_data(data):
    num_users, num_repos = data.max(0) + 1
    u2r = np.empty((num_users,), dtype=object)
    r2u = np.empty((num_repos,), dtype=object)
    u2r[:] = [[] for i in range(len(u2r))]
    r2u[:] = [[] for i in range(len(r2u))]
    for u,r in data:
        u2r[u].append(r)
        r2u[r].append(u)
    u2r[:] = [np.array(x, dtype=int) for x in u2r]
    r2u[:] = [np.array(x, dtype=int) for x in r2u]
    return u2r, r2u

def setup_matrix(u2r, r2u, stop=None):
    out = np.empty(r2u.shape, dtype=object)
    for repo, users in enumerate(r2u[:stop]):
        out[repo] = count_nested_a(u2r[users])
    for repo in range(len(out)):
        x = out[repo]
        if len(x) > 0:
            out[repo] = x[x[:,0] != repo]
    return out

def suggest(user, u2r, r2u, r2from, matrix, n=None, cutoff=None):

    repos = u2r[user]
    forked_from = r2from[repos]
    forked_from = forked_from[ forked_from.nonzero() ]
    all_repos = np.hstack((repos, forked_from))

    neighbors = np.empty((len(all_repos),), dtype=object)
    neighbors[:] = [matrix[r][:cutoff,0] for r in all_repos]

    out = count_nested_a(neighbors, repos)

    if n is None:
        return out
    else:
        return out[:n,0]

def run_suggest(users, u2r, r2u, r2from, matrix, n=10, cutoff=500):
    results = {}
    for u in users:
        print(u)
        results[u] = suggest(u, u2r, r2u, r2from, matrix, n, cutoff)
    return results


# ------- scipy.sparse implementation --------


import sys
import re
import scipy.sparse
from sparse import load_sparse, save_sparse

class Suggestion:

    data_filename = 'download/data.txt'
    followers_filename = 'download/followers.txt'
    lang_filename = 'download/lang.txt'
    repos_filename = 'download/repos.txt'

    def __init__(self):
        self.data = self.load_data()
        self.followers = self.load_followers()
        self.forked = self.load_forked()

    def load_data(self):
        filename = self.data_filename
        return self.sparse_from_text(filename)

    def load_followers(self):
        filename = self.followers_filename
        try:
            n = self.data.shape[0]
            shape = (n,n)
        except AttributeError:
            shape = None
        return self.sparse_from_text(filename, shape=shape)

    def load_forked(self):
        filename = self.repos_filename
        try:
            n = self.data.shape[1]
            shape = (n,n)
        except AttributeError:
            shape = None
        return self.forked_from_text(filename, shape=shape)

    @staticmethod
    def sparse_from_text(filename, shape=None):
        data = np.loadtxt(filename, delimiter=':', dtype=int)
        values = np.ones((len(data),), dtype=float)
        ij = np.transpose(data)
        return scipy.sparse.csr_matrix((values,ij), shape=shape)

    @staticmethod
    def forked_from_text(filename, shape=None):
        r = re.compile(r'^(\d+):.*,.*,(\d+)$')
        r2from = []
        with open(filename) as f:
            for line in f:
                m = r.match(line)
                if m is not None:
                    r2from.append([int(x) for x in m.groups()])
        data = np.array(r2from)
        values = np.ones((len(data),), dtype=float)
        ij = np.transpose(data)
        return scipy.sparse.csr_matrix((values,ij), shape=shape)

    @staticmethod
    def multiply_broadcast(A, m):
        """Multiply A * m (or m * A) by broadcasting m to a diagonal."""
        if scipy.sparse.issparse(m):
            if m.shape[0] == 1 or m.shape[1] == 1:
                m = m.todense()
            else:
                return A.multiply(m)
        m = np.asarray(m)
        if not 1 <= m.ndim <= 2:
            raise ValueError("don't know how to multiply with ndim > 2")
        rows = False
        if m.ndim == 2:
            if m.shape[0] == 1:
                m = m[0]
            elif m.shape[1] == 1:
                m = m[:,0]
                rows = True
            else:
                raise ValueError("m must be 1xN or Nx1")
        n = len(m)
        diag = scipy.sparse.dia_matrix((m[None,:], [0]), shape=(n,n))
        if rows:
            return diag * A
        else:
            return A * diag

    @classmethod
    def normalize_rows(cls, A):
        """Normalize each row of the sparse matrix A so each row sums to 1."""
        inverse = 1 / A.sum(1)
        inverse[np.isinf(inverse)] = 0
        return cls.multiply_broadcast(A, inverse)


    def prepare(self, forked_weight = 0.5):
        nusers, nrepos = self.data.shape

        # r2r = # of users watching both i and j
        D = self.data.tocsc()
        r2r = D.transpose() * D

        # Add links to parent repositories.  The weight of the parent
        # repository is equal to the weight of itself times forked_weight.
        weights = r2r.diagonal() * forked_weight
        r2r = r2r + self.multiply_broadcast(self.forked, weights[:,None])

        # r2r[i,j] = probability of going from repo i to repo j on each
        # iteration
        self.r2r = self.normalize_rows(r2r).tocsr()

        # Normalize the data so we sum to 1.
        self.u2r = self.normalize_rows(D).tocsr()


    def run(self, users, verbose = True, **kwargs):
        results = {}
        for u in users:
            if verbose:
                print(u)
            results[u] = self.suggest(u, **kwargs)
        self.results = results
        return results


    def suggest(self, u, follow_factor = 0.25, iterations = 2, top = 10,
            sparse_cutoff = 0.33, falloff = 1.0):

        # Give all of the followees a total weight of `follow_factor` and the
        # user a weight of `1 - follow_factor`.
        f = self.followers[u].todense()
        if follow_factor >= 1.0:
            f[0,u] = 1.0
            f /= f.sum()
        elif f.sum() > 0:
            f /= f.sum()
            f *= follow_factor
            f[0,u] = 1.0 - follow_factor
        else:
            f[0,u] = 1.0

        # Starting with the repositories watched by f, iterate to spread out
        # probability, and add it up.
        #state = f * self.u2r
        state = scipy.sparse.csr_matrix(f) * self.u2r
        accumulator = (state * 0).todense()
        threshold = int(state.shape[1] * sparse_cutoff)
        for i in range(iterations):
            # Once the state becomes partially-filled, it's faster to use a
            # dense matrix.
            try:
                if state.nnz > threshold:
                    state = state.todense()
            except AttributeError:
                pass
            state = state * self.r2r
            accumulator = accumulator + state * (falloff ** i)

        # Sort the scores, ignoring those that are already being watched.
        # Take the top results that are not already being watched.
        watched = self.data[u].nonzero()
        accumulator[watched] = 0
        accumulator = np.asarray(accumulator)[0]
        ranking = (-accumulator).argsort()
        results = ranking[:top]
        scores = accumulator[results]
        top = np.sum(scores > 0)

        return ranking[:top].tolist()

    def print_results(self, results=None, file=None):
        if results is None:
            results = self.results
        if file is None:
            file = sys.stdout
        keys = results.keys()
        keys.sort()
        for k in keys:
            print(k, ','.join(map(str,results[k])), sep=':', file=file)

