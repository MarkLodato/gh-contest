CFLAGS = -g3 -O3
CPPFLAGS = -I/usr/include/python2.6
CC = gcc
CYTHON = cython
CYTHON_FLAGS = --line-directives

all : _ghcontest.so

%.c : %.pyx
	$(CYTHON) $(CYTHON_FLAGS) $<

%.so : %.c
	$(CC) -fPIC -shared $(CPPFLAGS) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

clean:
	$(RM) _ghcontest.c _ghcontest.so

.PRECIOUS: %.c
