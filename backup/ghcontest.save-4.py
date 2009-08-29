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
import sys
import re
import scipy.sparse
import numpy as np
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


    def prepare(self, forked_weight = 1.0):
        nusers, nrepos = self.data.shape

        # r2r = # of users watching both i and j
        D = self.data.tocsc()
        r2r = D.transpose() * D
        self.common_watchers = r2r.copy()
        r2r.data **= 1.1

        # Add links to parent repositories.  The weight of the parent
        # repository is equal to the weight of itself times forked_weight.
        weights = r2r.diagonal() * forked_weight
        r2r = r2r + self.multiply_broadcast(self.forked, weights[:,None])

        # r2r[i,j] = probability of going from repo i to repo j on each
        # iteration
        self.r2r = self.normalize_rows(r2r).tocsr()

        # Normalize the data so we sum to 1.
        self.u2r = self.normalize_rows(D).tocsr()

        # Number of watchers of each repo.
        self.watchers = np.asarray(self.data.sum(0))[0]
        self.watchers /= self.watchers.sum()
        self.most_popular = self.watchers.argsort()[::-1]
        self.top_ten = self.most_popular.tolist()[:10]


    def run(self, users, verbose = True, **kwargs):
        results = {}
        for u in users:
            if verbose:
                print(u)
            results[u] = self.suggest(u, **kwargs)
        self.results = results
        return results


    def suggest(self, u, follow_factor = 0.25, top = 10, sorted=True):

        watched = self.u2r[u]
        #fork1 = watched * self.forked
        #fork2 = fork1 * self.forked
        #state = (watched + fork1 * .25 + fork2 * .25) * self.r2r
        state = watched * self.r2r
        #follows = self.followers[u] * self.u2r
        #state = ((watched * self.r2r) * (1 - follow_factor)
        #        + follows * follow_factor)
        accumulator = state.todense()

        # Sort the scores, ignoring those that are already being watched.
        # Take the top results that are not already being watched.
        #accumulator += accumulator.max() * self.watchers
        watched = self.data[u].nonzero()
        accumulator[watched] = 0
        accumulator = np.asarray(accumulator)[0]
        ranking = accumulator.argsort()[::-1]

        # Make sure we are not outputting crap.
        results = ranking[:top]
        scores = accumulator[results]
        l = np.sum(scores > 0)
        answer = ranking[:l].tolist()

        # If we have no other guess, default to the most popular.
        # This only added two results; doesn't seem worth doing.
        if l < top:
            answer += self.top_ten[:top-l]

        if sorted:
            answer.sort()
        return answer

    def print_results(self, results=None, file=None):
        if results is None:
            results = self.results
        if file is None:
            file = sys.stdout
        keys = results.keys()
        keys.sort()
        for k in keys:
            print(k, ','.join(map(str,results[k])), sep=':', file=file)

    def create_graph(self, user, filename, top=10, inner=False, forked=True):

        with open(filename, 'w') as f:

            f.write('digraph user_%s {\n' % user);
            f.write(' node [shape=ellipse];\n')
            f.write(' edge [dir=none];\n')

            # Color all of the watched repositories.
            repos = self.data[user].nonzero()[1]
            for r in repos:
                f.write(' "%s" [style=filled, fillcolor="#33cc33"];\n' % r)

            w = self.common_watchers
            done = set()

            if forked:
                for r in repos:
                    x = self.forked[r]
                    if x.nnz > 0:
                        o = x.nonzero()[1][0]
                        if (o,r) in done:
                            continue
                        done.add((r,o))
                        f.write(' "%s" -> "%s" [label="%i", '
                                'color=red, dir=forward];\n'
                                % (r, o, w[r,o]))

            if inner:
                for a,b in itertools.combinations(repos, 2):
                    if w[a,b] > 1:
                        if (o,r) in done or (r,o) in done:
                            continue
                        done.add((r,o))
                        f.write(' "%s" -> "%s" [label="%i"];\n'
                                % (a, b, w[a,b]))

            # Add all of the linked repositories.
            if top > 0:
                for r in repos:
                    scores = np.asarray( w[r].todense(), dtype=int )[0]
                    scores[r] = 0
                    t = min(np.sum(scores > 1), top)
                    others = scores.argsort()[:-t-1:-1]
                    for o in others:
                        if (o,r) in done or (r,o) in done:
                            continue
                        done.add((r,o))
                        f.write(' "%s" -> "%s" [label="%s"];\n'
                                % (r, o, scores[o]))

            f.write('}\n')
