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

def run_suggest(users, u2r, r2u, r2from, n=10, cutoff=None):
    results = {}
    for u in users:
        print(u)
        results[u] = suggest(u, u2r, r2u, r2from, n, cutoff)
    return results

def save_suggest(filename, results):
    with open(filename, 'w') as f:
        for k,v in sorted(results.iteritems()):
            f.write('%s:%s\n' % (k, ','.join(map(str,v))))
