====================
GitHub Contest Entry
====================

:Author: Mark Lodato
:Date: August, 2009


Introduction
------------

My algorithm is very simple, and I was surprised how well it did.  One evening
I thought of the simplest thing I could (a naive and partial implementation of
the Base Algorithm below) and I got 20% on my first shot, half of the leader
at that time!  I did some tweaking and my second shot got 30%.  So, I figured
maybe I'd continue down this path.  My program no where near as sophisticated
as other entires, but it is really simple and runs reasonably fast (on one CPU
of my Core 2 Duo 6300, it takes about a minute to train and makes about 100
suggests/sec.)

The basic idea is to represent repositories as states in a Markov process.
Start by setting with equal probability all states that the user is watching
(the rest set to zero), step the process once, and list the 10 most likely
states that the user is not already watching.  (If this doesn't make sense,
read the algorithm below.  It sounds fancier than it really is.)

How do you determine the probability of going from repository `i` to
repository `j`?  Well, I got the best results by using how many users are
watching both `i` and `j`, and then adding in information on whether `i` was
forked from `j`.


Rundown
-------

The following is pseudo-code description using pseudo-NumPy syntax.

init()

1.  data = load('data.txt')
2.  forked = load_forked_from('repos.txt')

prepare()

1.  common_watchers = data.tranpose() * data  (matrix multiply)
2.  r2r = common_watchers.copy()
3.  for child, parent in forked: r2r[child,parent] += r2r[child,child]
4.  r2r = common_watchers / common_watchers.sum(1)[...,newaxis]
5.  u2r = data / data.sum(1)[...,newaxis]

suggest(u)

1.  state = u2r[u]
2.  next = state * r2r   (matrix multiply)
3.  for r in state.nonzero():  next[r] = 0
4.  return next.argsort(reverse=True)


Base Algorithm
--------------

1.  To start, load `data.txt` from disk into sparse matrix, ``D``, where
    ``D[u,r]`` is 1 if user ``u`` is watching repository ``r``, 0 otherwise.
    [``self.data``]

2.  Compute ``W[a,b]``, the number of users watching both repositories ``a``
    and ``b``.  This is simply ``D' * D``.  [``self.common_watchers``]

3.  Compute ``P[a,b]``, the probability of transitioning from repository ``a``
    to repository ``b`` in our Markov chain.  (``P`` is right stochastic.)
    This is just diving each row of ``W`` by the row sum.  [``self.r2r``]

4.  Compute ``S[u,r]``, the probability of selecting repository ``r`` at
    random from those being watched by user ``u``.  This is just dividing each
    row of ``D`` by the row sum.  [``self.u2r``]

5.  For each user ``u`` for which we wish to make suggestions:

    a.  Let ``state = S[u]``, which is ``1/n`` for each of the ``n``
        repositories user ``u`` is watching, 0 otherwise.

    b.  Step the Markov chain once by computing ``next = state * D``.

    c.  Set ``next[r] = 0`` for each repository ``r`` that the user ``u`` is
        watching.  (This is so we don't suggest something the user is already
        watching.)

    d.  Return the top 10 highest values.


Adding Fork Information
-----------------------

It ends up a lot of repositories are only watched by one person: the owner.
But, many of these are forked, and it is obvious that an owner of a child
repository may watch the parent.  So, between steps 2 and 3, if ``p`` is a
parent of ``c``, add the number of users watching ``c`` (``W[c,c]``) to
``W[c,p]``.  This resulted in a huge improvement.


Fallback
--------

If the above returns `n` < 10 results, append the 10 - `n` most popular
repositories.  This did not seem to help at all.


Failed Ideas
------------

The following seem like they should produce better results, but when I tried
them, I got worse results.  Perhaps I did it wrong.

* Followers: It seems like if user `i` is following user `j`, then user `j`'s
  repositories should perhaps be weighted higher.

* Multiple steps:  I would think that stepping multiple times, weighting each
  step ``n`` in some way (I tried ``exp(-n)``) would give better results, but
  this worked worse for me. 

* Fork chains:  It seemed to me that it would be useful to relate a repository
  to *all* other repositories in its fork graph, not just its parent.  I tried
  doing something like this, but I got worse results.


When giving suggestions for a particular user, I tried giving weights to every
other user based on how much overlap there was between the repositories being
watched, the thought being that I want to find other users that have similar
interests.  From here, for each user `u`, I added `u`'s weight to each
repository he was watching, and then I took the top 10 repos.  This worked
*much* worse than the above algorithm.  But maybe there is promise here...


Other Ideas
-----------

If a user is only watching a single repository, and other users are watching
that repository, I think my algorithm should be pretty good, since I'm
calculating ``Pr(r|w)``, the probability of watching ``r`` given that the user
is already watching ``w``.  But, our users are watching multiple repositories,
and I'm basically computing ``Pr(r|w0) + Pr(r|w1) + ...``.  What I really want
to find out is ``Pr(r|w0 w1 w2 ...)``, which is not same since these are not
independent.  If we had *tons* of data, then we could just estimate this
empirically.  But there is almost zero chance that any other users are
watching exactly ``w0``, ``w1``, ``w2``, etc.  But maybe we can use pairwise
probabilities somehow, as in ``Pr(r|wi wj)`` for all ``i``/``j``?

Somehow tie in information about the owner of each repository.  For example,
treat repositories owned by the user different than repositories owned by other
people.


