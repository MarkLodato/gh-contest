/* Copyright 2009, Mark Lodato
 * 
 * August 31, 2009 and before:
 * 
 *   All rights reserved.  You may not copy, with or without modification, any
 *   of the code in this project.
 * 
 * September 1, 2009 and after:
 * 
 *   MIT License:
 * 
 *   Permission is hereby granted, free of charge, to any person obtaining a
 *   copy of this software and associated documentation files (the
 *   "Software"), to deal in the Software without restriction, including
 *   without limitation the rights to use, copy, modify, merge, publish,
 *   distribute, sublicense, and/or sell copies of the Software, and to permit
 *   persons to whom the Software is furnished to do so, subject to the
 *   following conditions:
 * 
 *   The above copyright notice and this permission notice shall be included
 *   in all copies or substantial portions of the Software.
 * 
 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 *   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 *   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
 *   NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 *   DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 *   OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
 *   USE OR OTHER DEALINGS IN THE SOFTWARE.
 */


#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <error.h>


#define MAX_USERS   65536
#define MAX_REPOS   131072
#define MAX_U2R     1024
#define MAX_R2U     4096
#define N           10


struct int_list {
    int value;
    struct item *next;
};

static struct int_list u2r[MAX_USERS];
static struct int_list r2u[MAX_REPOS];
static int r2from[MAX_REPOS];
static int test[MAX_USERS];
static int num_test;


static void
append(struct int_list **list, int value)
{
    struct int_list *item;
    item = malloc(sizeof(*item));
    if (!item)
        error(EXIT_FAILURE, errno, "malloc");
    item->value = value;
    item->next = NULL;
    while (*list)
        list = &(*list)->next;
    *list = item;
}


static struct int_list
copy_list(struct int_list *list)
{
    struct int_list *p, *r;
    if (!list)
        return NULL;
    r = malloc(sizeof(*r));
    if (!r)
        error(EXIT_FAILURE, errno, "malloc");
    *r = *list;
    list = list->next;
    for (p = r; list != NULL; p = p->next, list = list->next) {
        struct int_list *n = malloc(sizeof(*n));
        if (!n)
            error(EXIT_FAILURE, errno, "malloc");
        *n = *list;
        p->next = n;
    }
    return r;
}


static void
read_data(const char *filename)
{
    FILE *f;
    int user, repo, c;

    f = fopen(filename, "r");
    if (f == NULL)
        error(EXIT_FAILURE, errno, "error opening %s", filename);

    memset(u2r, 0, sizeof u2r);
    memset(r2u, 0, sizeof r2u);
    while ((c = fscanf(f, "%d:%d\n", &user, &repo)) != EOF) {
        if (c == 2) {
            append(&u2r[user], repo);
            append(&r2u[repo], user);
        }
    }

    fclose(f);
}


static void
read_test(const char *filename)
{
    FILE *f;
    int user, c;

    f = fopen(filename, "r");
    if (f == NULL)
        error(EXIT_FAILURE, errno, "error opening %s", filename);

    memset(test, 0, sizeof test);
    num_test = 0;
    while ((c = fscanf(f, "%d", &user)) != EOF) {
        if (c == 1) {
            test[num_test++] = user;
        }
    }

    fclose(f);
}


static void
read_repos(const char *filename)
{
    char line[2048], name[2048], date[2048];
    FILE *f;
    int user, from, c;

    f = fopen(filename, "r");
    if (f == NULL)
        error(EXIT_FAILURE, errno, "error opening %s", filename);

    memset(r2from, 0, sizeof r2from);
    while (fread(line, sizeof line, f) != EOF) {
        if (sscanf(line, "%d:[^,\n],[^,\n],%d\n", &repo, name, date, &from)
            == 4) {
            r2from[repo] = from;
        }
    }

    fclose(f);
}


static int
suggest(int user, int cutoff)
{
    struct int_list *repos, *p;

    repos = copy_list(u2r[user]);

    // TODO
}
