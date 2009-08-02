#!/usr/bin/python

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

def run_suggest(users, u2r, r2u, n=10):
    return dict( (u, suggest(u, u2r, r2u, n)) for u in users )

def save_suggest(filename, results):
    with open(filename, 'w') as f:
        for k,v in sorted(results.iteritems()):
            f.write('%s:%s\n' % (k, ','.join(map(str,v))))
