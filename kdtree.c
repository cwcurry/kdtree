/* kdtree.c -- KD-Tree implementation for spatial indexing.

   This software is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty
   of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.


   This is an implementation of a KD-Tree, which is a space-partitioning data
 structure for storing k-dimensional data.  This data structure can be used for
 efficient multidimensional searchs (such as nearest-neighbor searches).  Typically
 the median of a set of points is chosen as the axis-aligned split plane.  This
 implementation instead will choose the Nth-ordered point such that the result
 is a complete binary tree (not necessarily full).  If the tree is a complete
 tree, then it can be represented as an array with node pointers implicitly
 defined by the array indices (i.e., a binary heap).  Therefore no additional
 memory is needed to store the tree structure.  In one dimension an array can
 be sorted and a binary search applied to it.  This implementation applies that
 principle in higher dimensions.

 References:
   Bentley, J. L. Multidimensional Binary Search Trees used for Associative Searching.
 Communications of the ACM, 18(9): 509, 1975. */

#include <stdlib.h>  /* for NULL    */
#include <string.h>  /* for memcpy  */
#include <math.h>    /* for fabs    */
#include <float.h>   /* for DBL_MAX */
#include <assert.h>  /* for assert  */
#include "kdtree.h"


/* The threshold between heap and array insertion for k-nearest-neighbor searches. */
#define KNN_INSERT_THRESHOLD  20

/* Heap Pointer Functions */
#define HEAP_PARENT(i)  ((i)/2)
#define HEAP_LEFT(i)    (2*(i))
#define HEAP_RIGHT(i)   (2*(i)+1)
#define HEAP_OTHER(i)   ((i)^1)

#define KD_MIN(x, y) ((x) < (y) ? (x) : (y))


/* Type of function used to insert a value into a knn_result structure. */
typedef void (* knn_insert_fn_t) (knn_workspace *w, const knn_result *p);


/* KD-Tree Structure */
struct kd_tree_struct {
    void *root;                     /* Root of tree (uses 1-based indexing). */
    size_t n;                       /* Number of nodes in the tree.          */
    size_t elsize;                  /* Size of each node in bytes.           */
    int dim;                        /* Dimension of the data.                */
    kd_compare_fn_t cmp_fn;
    kd_dist_fn_t dist_fn;
    kd_dist1_fn_t dist1_fn;
    kd_swap_fn_t swap_fn;
    kd_set_axis_fn_t set_axis_fn;
    kd_find_axis_fn_t find_axis_fn;
};

/* K-Nearest Neighbor Search Workspace Structure */
struct knn_workspace_struct {
    knn_result *r;                /* Pointer to an array of result elements. */
    size_t k;                     /* Number of knn_result elements stored.   */
    size_t n;                     /* Maximum (allocated) number of results.  */
    knn_insert_fn_t insert_fn;    /* Function to insert results into r.      */
};


/* Private Function Prototypes */
static void kd_build_recursive (kd_tree *, void *, const size_t, const size_t, const int);
static void * kd_select (kd_tree *, void *, const size_t, const size_t, const int);
static void kd_search_recursive (double *, void **, const void *, const kd_tree *, const size_t, const int);
static void knn_search_recursive (knn_workspace *, const void *, const kd_tree *, const size_t, const int);
static void knn_insert_heap (knn_workspace *, const knn_result *);
static void knn_insert_array (knn_workspace *, const knn_result *);
static int log2_floor (const unsigned int);


/******************************************************************************
*
* kd_alloc - Allocate memory for a KD-Tree object and initialize it for N data
*   points.
*
* Return : A pointer to the newly-allocated tree, or
*          NULL if memory could not be allocated.
*
******************************************************************************/

kd_tree * kd_alloc (
    const size_t n) {

    kd_tree *tree;


    assert (n > 0);

    /* Allocate the kd_tree object. */
    tree = (kd_tree *) malloc (sizeof (kd_tree));
    if (!tree) return NULL;

    /* Delay allocation of the data array until building. */
    tree->root = NULL;

    /* Initialize the kd_tree object. */
    tree->n = n;
    tree->elsize = sizeof (kd_point);
    tree->dim = 2;

    /* Initialize default function pointers. */
    tree->cmp_fn = kd_greater_than;
    tree->dist_fn = kd_dist_l2;
    tree->dist1_fn = kd_dist1;
    tree->swap_fn = kd_swap;

    tree->set_axis_fn = NULL;
    tree->find_axis_fn = NULL;

    return tree;
}


/******************************************************************************
*
* kd_free - Free memory allocated for a KD-Tree given by TREE.
*
******************************************************************************/

void kd_free (
    kd_tree *tree) {

    if (tree) {
        if (tree->root) {
            /* Set the pointer to zero-based indexing before free'ing it. */
            free ((char *) (tree->root) + 1 * tree->elsize);
            tree->root = NULL;
        }
        free (tree);
    }
}


/******************************************************************************
*
* kd_set_point_type - Set the point element size and offset of the position array
*   in the node for the KD-Tree given by TREE.
*
******************************************************************************/

void kd_set_point_type (
    kd_tree *tree,
    const size_t elsize,
    const int dim) {

    assert (tree);
    assert (dim > 0);
    assert (elsize > 0);

    tree->elsize = elsize;
    tree->dim = dim;
}


/******************************************************************************
*
* kd_set_cmp_fn - Set the comparison function of the KD-Tree given by TREE.
*
******************************************************************************/

void kd_set_cmp_fn (
    kd_tree *tree,
    kd_compare_fn_t cmp_fn) {

    assert (tree);
    assert (cmp_fn);

    tree->cmp_fn = cmp_fn;
}


/******************************************************************************
*
* kd_set_swap_fn - Set the function to swap two node values in memory in the
*   KD-Tree given by TREE.
*
******************************************************************************/

void kd_set_swap_fn (
    kd_tree *tree,
    kd_swap_fn_t swap_fn) {

    assert (tree);
    assert (swap_fn);

    tree->swap_fn = swap_fn;
}


/******************************************************************************
*
* kd_set_dist_fn - Set the distance functions of the KD-Tree given by TREE.
*
******************************************************************************/

void kd_set_dist_fn (
    kd_tree *tree,
    kd_dist_fn_t dist_fn,
    kd_dist1_fn_t dist1_fn) {

    assert (tree);
    assert (dist_fn);
    assert (dist1_fn);

    tree->dist_fn = dist_fn;
    tree->dist1_fn = dist1_fn;
}


/******************************************************************************
*
* kd_set_axis_fn - Set the functions to find and set the axis-aligned split
*   plane value of a node in the KD-Tree given by TREE.
*
******************************************************************************/

void kd_set_axis_fn (
    kd_tree *tree,
    kd_set_axis_fn_t set_axis_fn,
    kd_find_axis_fn_t find_axis_fn) {

    assert (tree);
    assert (set_axis_fn);
    assert (find_axis_fn);

    tree->set_axis_fn = set_axis_fn;
    tree->find_axis_fn = find_axis_fn;
}


/******************************************************************************
*
* kd_get_data - Return a pointer to the array of data objects in the KD-Tree
*   TREE.  If there are no points or TREE is NULL, then NULL is returned.  The
*   data is owned by the kd_tree object and should not be free'd by the caller.
*   This should be used only after kd_build has been called.
*
******************************************************************************/

void * kd_get_data (
    const kd_tree *tree) {

    if (tree) {
        assert (tree->root);
        return (char *) (tree->root) + 1 * tree->elsize;
    }

    return NULL;
}


/******************************************************************************
*
* kd_get_count - Return the number of elements in the KD-Tree TREE or return
*   zero if TREE is NULL.
*
******************************************************************************/

size_t kd_get_count (
    const kd_tree *tree) {

    return tree ? tree->n : 0;
}


/******************************************************************************
*
* kd_get_dim - Return the dimensions of the data in the KD-Tree TREE or return
*   zero if TREE is NULL.
*
******************************************************************************/

int kd_get_dim (
    const kd_tree *tree) {

    return tree ? tree->dim : 0;
}


/******************************************************************************
*
* kd_build - Build the KD-Tree as an array.  The tree is built recursively from
*   the array of points P.  Each element of P should be of type kd_point or
*   should be defined from kd_set_point_type with the dimensions and element size.
*   If each element of P is not of type kd_point, then a valid comparison and swap
*   function are required to be set by kd_set_swap_fn and kd_set_cmp_fn.  The
*   order of the array P will be modified by this call.  This runs in O(N lg (N)).
*
******************************************************************************/

void kd_build (
    kd_tree *tree,
    void *p) {

    assert (p);
    assert (tree);
    assert (tree->n > 0);
    assert (tree->dim > 0);
    assert (tree->elsize > 0);
    assert (tree->cmp_fn);
    assert (tree->swap_fn);

    /* Allocate the data array and set the root node to 1-based indexing
       to simplify the heap-pointer indexing. */
    tree->root = malloc (tree->n * tree->elsize);
    tree->root = (char *) (tree->root) - 1 * tree->elsize;

    /* Recursively build the KD-Tree. */
    kd_build_recursive (tree, p, 1, tree->n, 0);
}


/******************************************************************************
*
* kd_build_recursive - Recursively insert a node into a KD-Tree TREE at node
*   index I from the array P of length N elements.  The split axis is first
*   chosen, then a pivot location is selected such that the result is a complete
*   binary tree.  The pivot node is then copied into the new tree.
*
******************************************************************************/

static void kd_build_recursive (
    kd_tree *tree,
    void *p,
    const size_t i,
    const size_t n,
    const int level) {

    const size_t m = tree->elsize;
    size_t h, nl, nr;
    void *k, *q;
    int axis;


    if (n == 0) return;

    /* Calculate the pointer to the node given by the node index i. */
    q = (char *) (tree->root) + i * m;

    if (tree->find_axis_fn) {
        /* Choose the axis to split the nodes on (if provided). */
        axis = tree->find_axis_fn (p, n, level);
    } else {
        /* Otherwise the split-axis alternates at each level. */
        axis = level % tree->dim;
    }

    /* Base case. */
    if (n == 1) {
        memcpy (q, p, m);
        if (tree->set_axis_fn) {
            tree->set_axis_fn (q, axis);
        }
        return;
    }

    /* Choose the pivot location to create a complete tree.  Calculate the number
       of nodes that would be in the left and right subtree of a complete binary
       tree with n nodes. */
    h = 1 << log2_floor (n);
    nl = KD_MIN (n - h / 2, h - 1);
    nr = n - nl - 1;

    /* Partition the array p and return the nl-th ordered element as the split node. */
    k = kd_select (tree, p, n, nl, axis);

    /* Copy the splitting node value into the new tree. */
    memcpy (q, k, m);
    if (tree->set_axis_fn) {
        tree->set_axis_fn (q, axis);
    }

    /* Recursively insert nodes into each subtree. */
    kd_build_recursive (tree, p, HEAP_LEFT (i), nl, level + 1);
    kd_build_recursive (tree, (char *) p + (nl + 1) * m, HEAP_RIGHT (i), nr, level + 1);
}


/******************************************************************************
*
* kd_select - Return the Kth-ordered element in the array P of size N.  Elements
*   less than K will be placed in P[0..K-1] in an arbitrary order.  Elements
*   greater than K will be placed in P[K+1..N-1] in an arbitrary order.  The
*   median will be chosen if K = (N-1)/2.  This algorithm is Quick Select by
*   C.A.R. Hoare with a modification of the pivot selection.  It's average run
*   time is O(N).  Note that the first ordered element is K=0.
*
* References:
*   C.A.R. Hoare, 1961.  Algorithm 63 (Partition) and Algorithm 65 (Find).
*     Communications of the ACM, v. 4, no. 7, pp. 321 - 322.
*
******************************************************************************/

static void * kd_select (
    kd_tree *tree,
    void *pp,
    const size_t n,
    const size_t k,
    const int axis) {

    char * const p = (char * const) pp;
    kd_compare_fn_t cmp = tree->cmp_fn;
    kd_swap_fn_t swap = tree->swap_fn;
    const size_t m = tree->elsize;
    size_t low, mid, high;
    size_t ll, hh;


    assert (pp);
    assert (n > 0);
    assert (k < n);

    low = 0;
    high = m * (n-1);

    for (;;) {
        if (high <= low) return p + k * m; /* One element.  */

        if (high == low + m) {             /* Two elements. */
            if (cmp (p + low, p + high, axis) > 0) swap (p + low, p + high);
            return p + k * m;
        }

        /* Find the median of low, middle and high items; Swap into position low. */
        mid = low + m * (((high - low) / m) / 2);

        if (cmp (p + mid, p + high, axis) > 0) swap (p + mid, p + high);
        if (cmp (p + low, p + high, axis) > 0) swap (p + low, p + high);
        if (cmp (p + mid, p + low,  axis) > 0) swap (p + mid, p + low);

        /* Swap low item (now in position middle) into position (low+1). */
        swap (p + mid, p + low + m);

        /* Nibble from each end towards the middle, swapping items when stuck. */
        ll = low + m;
        hh = high;

        for (;;) {
            do ll += m; while (cmp (p + low, p + ll,  axis) > 0);
            do hh -= m; while (cmp (p + hh,  p + low, axis) > 0);

            if (hh < ll) break;

            swap (p + ll, p + hh);
        }

        /* Swap middle item (in position low) back into correct position. */
        swap (p + low, p + hh);

        /* Re-set active partition. */
        if (hh >= k * m) high = hh - m;
        if (hh <= k * m) low = ll;
    }

    /* Not reached. */
    return NULL;
}


/******************************************************************************
*
* kd_search - Search for the nearest-neighbor point to P in the KD-Tree TREE and
*   return a pointer to it.  If DIST_MAX is not zero, then a maximum (squared)
*   distance of DIST_MAX will be searched.  If DIST is not NULL, then the (squared)
*   distance between both points will be returned in it.  This runs in O(lg (N)).
*
******************************************************************************/

void * kd_search (
    double *dist,
    const void *p,
    const kd_tree *tree,
    const double dist_max) {

    double dist_min;
    void *nn = NULL;


    assert (p);
    assert (tree);
    assert (tree->root);
    assert (tree->dist_fn);
    assert (tree->dist1_fn);
    assert (tree->elsize > 0);
    assert (dist_max >= 0.0);

    if (dist_max > 0.0) {
        dist_min = dist_max;
    } else {
        dist_min = DBL_MAX;
    }

    /* Recursively search the KD-Tree starting at the root node (i=1). */
    kd_search_recursive (&dist_min, &nn, p, tree, 1, 0);

    /* Return the minimum (squared) distance found if requested. */
    if (dist) *dist = dist_min;

    return nn;
}


/******************************************************************************
*
* kd_search_xy - Search for the nearest-neighbor point to (X,Y) in the KD-Tree
*   TREE and return a pointer to it.  If DIST_MAX is not zero, then a maximum
*   (squared) distance of DIST_MAX will be searched.  If DIST is not NULL, then
*   the (squared) distance between both points will be returned in it.  This
*   runs in O(lg (N)).
*
******************************************************************************/

void * kd_search_xy (
    double *dist,
    const double x,
    const double y,
    const kd_tree *tree,
    const double dist_max) {

    kd_point p;
    double dist_min;
    void *nn = NULL;


    assert (tree);
    assert (tree->root);
    assert (tree->dist_fn);
    assert (tree->dist1_fn);
    assert (tree->elsize > 0);
    assert (dist_max >= 0.0);

    p.x[0] = x;
    p.x[1] = y;

    if (dist_max > 0.0) {
        dist_min = dist_max;
    } else {
        dist_min = DBL_MAX;
    }

    /* Recursively search the KD-Tree starting at the root node (i=1). */
    kd_search_recursive (&dist_min, &nn, &p, tree, 1, 0);

    /* Return the minimum (squared) distance found if requested. */
    if (dist) *dist = dist_min;

    return nn;
}


/******************************************************************************
*
* kd_search_recursive - Recursively search for the nearest-neighbor point to
*   P in the KD-Tree TREE.
*
******************************************************************************/

static void kd_search_recursive (
    double *dist_min,
    void **nn,
    const void *p,
    const kd_tree *tree,
    const size_t i,
    const int level) {

    double d, dp;
    void *q;
    size_t next;


    /* The array index i is equivalent to a node pointer.  If this value is
       greater than n, then this means the pointer is NULL, or a leaf node. */
    if (i > tree->n) return;

    /* Calculate the pointer to the node given by the node index i. */
    q = (char *) (tree->root) + i * tree->elsize;

    /* Calculate the distance from the query point to the axis split plane. */
    dp = tree->dist1_fn (p, q, level);

    /* Calculate the distance metric from the query point P to the current node Q. */
    d = tree->dist_fn (p, q);

    /* Update the best distance so far. */
    if (d <= *dist_min) {
        *dist_min = d;
        *nn = q;
    }

    /* Select the next node (subtree) that contains the query point. */
    if (dp <= 0.0) {
        next = HEAP_LEFT (i);
    } else {
        next = HEAP_RIGHT (i);
    }

    /* Recursively descend down the subtree that is closest to the query point. */
    kd_search_recursive (dist_min, nn, p, tree, next, level + 1);

    /* If the candidate hypersphere crosses the splitting plane, then look on the
       other side of the plane (subtree) also. */
    if (dp * dp < *dist_min) {
        kd_search_recursive (dist_min, nn, p, tree, HEAP_OTHER (next), level + 1);
    }
}


/******************************************************************************
*
* knn_alloc - Allocate memory and initialize a knn_workspace object.  Space is
*   allocated to hold a maximum return of N results of a nearest-neighbor search.
*
******************************************************************************/

knn_workspace * knn_alloc (
    const size_t n) {

    knn_workspace *w;


    assert (n > 0);

    w = (knn_workspace *) malloc (sizeof (knn_workspace));

    w->r = (knn_result *) malloc (n * sizeof (knn_result));
    /* Offset pointer to give 1-based indexing.  This simplifies heap array indexing. */
    w->r = w->r - 1;
    w->n = n;
    w->k = 0;

    /* Set the result insertion function.  Array insertion is O(N) compared
       to heap insertion O(lg (N)), but is faster for small n. */
    if (n > KNN_INSERT_THRESHOLD) {
        w->insert_fn = knn_insert_heap;
    } else {
        w->insert_fn = knn_insert_array;
    }

    return w;
}


/******************************************************************************
*
* knn_free - Free memory allocated for the knn_workspace object W.
*
******************************************************************************/

void knn_free (
    knn_workspace *w) {

    if (w) {
        if (w->r) {
            /* Set the pointer to zero-based indexing before free'ing it. */
            free (w->r + 1);
            w->r = NULL;
        }
        free (w);
    }
}


/******************************************************************************
*
* knn_get_result_data - Return a pointer to the knn_result array for the
*   knn_workspace object W or return NULL if W is NULL.
*
******************************************************************************/

knn_result * knn_get_result_data (
    const knn_workspace *w) {

    return w ? w->r + 1 : NULL;
}


/******************************************************************************
*
* knn_get_result_count - Return the number of elements in the knn_result array
*   for the knn_workspace object W or return zero if W is NULL.
*
******************************************************************************/

size_t knn_get_result_count (
    const knn_workspace *w) {

    return w ? w->k : 0;
}


/******************************************************************************
*
* knn_search - Search for the K-Nearest Neighbor points to the point P using
*   the KD-Tree TREE.  All points found are stored in the workspace W.  This
*   runs in O(lg (N)).
*
******************************************************************************/

void knn_search (
    knn_workspace *w,
    const void *p,
    const kd_tree *tree) {

    assert (p);
    assert (w);
    assert (w->r);
    assert (w->insert_fn);
    assert (w->n > 0);
    assert (tree);
    assert (tree->root);
    assert (tree->dist_fn);
    assert (tree->dist1_fn);
    assert (tree->elsize > 0);

    /* Initialize the number of results to zero. */
    w->k = 0;

    /* Recursively search the KD-Tree starting at the root node (i=1). */
    knn_search_recursive (w, p, tree, 1, 0);
}


/******************************************************************************
*
* knn_search_xy - Search for the K-Nearest Neighbor points to the point (X,Y)
*   using the KD-Tree TREE.  All points found are stored in the workspace W.
*   This runs in O(lg (N)).
*
******************************************************************************/

void knn_search_xy (
    knn_workspace *w,
    const double x,
    const double y,
    const kd_tree *tree) {

    kd_point p;


    assert (w);
    assert (w->r);
    assert (w->insert_fn);
    assert (w->n > 0);
    assert (tree);
    assert (tree->root);
    assert (tree->dist_fn);
    assert (tree->dist1_fn);
    assert (tree->elsize > 0);

    /* Initialize the number of results to zero. */
    w->k = 0;

    p.x[0] = x;
    p.x[1] = y;

    /* Recursively search the KD-Tree starting at the root node (i=1). */
    knn_search_recursive (w, &p, tree, 1, 0);
}


/******************************************************************************
*
* knn_search_recursive - Recursively search for the K-Nearest Neighbor points
*   of the point P in the KD-Tree TREE.
*
******************************************************************************/

static void knn_search_recursive (
    knn_workspace *w,
    const void *p,
    const kd_tree *tree,
    const size_t i,
    const int level) {

    knn_result r;
    double dp;
    size_t next;
    void *q;


    /* The array index i is equivalent to a node pointer.  If this value is
       greater than n, then this means the pointer is NULL, or a leaf node. */
    if (i > tree->n) return;

    /* Calculate the pointer to the node given by the node index i. */
    q = (char *) (tree->root) + i * tree->elsize;

    /* Calculate the distance from the query point to the axis split plane. */
    dp = tree->dist1_fn (p, q, level);

    /* Calculate the distance metric from the query point P to the current node Q. */
    r.dist = tree->dist_fn (p, q);
    r.p = q;

    /* Update the best distance so far. */
    w->insert_fn (w, &r);

    /* Select the next node (subtree) that contains the query point. */
    if (dp <= 0.0) {
        next = HEAP_LEFT (i);
    } else {
        next = HEAP_RIGHT (i);
    }

    /* Recursively descend down the subtree that is closest to the query point. */
    knn_search_recursive (w, p, tree, next, level + 1);

    /* If the candidate hypersphere crosses the splitting plane, then look on the
       other side of the plane (subtree) also. */
    if (dp * dp < w->r[1].dist) {
        knn_search_recursive (w, p, tree, HEAP_OTHER (next), level + 1);
    }
}


/******************************************************************************
*
* knn_insert_heap - Insert the point P into the result array stored in the
*   workspace W.  The insertion uses a priority max-heap.  This is O(lg (N)).
*
******************************************************************************/

static void knn_insert_heap (
    knn_workspace *w,
    const knn_result *p) {

    size_t i, j;


    assert (p);
    assert (w);
    assert (w->r);
    assert (w->n > 0);
    assert (p->dist >= 0.0);
    assert (w->k <= w->n);

    /* Perform heap insert if the heap is not full. */
    if (w->k < w->n) {

        w->k++;

        for (i=w->k; i>1; i=j) {
            j = i / 2;
            if (p->dist <= w->r[j].dist) break;
            w->r[i] = w->r[j];
        }

        w->r[i] = *p;
        return;
    }

    /* At this point the heap is full. */

    /* w->r[1] is the top of the max heap, i.e., the maximum value in the heap.
       If the new distance is greater than the max, then discard it. */
    if (p->dist < w->r[1].dist) {

        /* Otherwise, replace the top with the new point p and heapify. */
        w->r[1] = *p;

        i = 1;

        while (1) {
            size_t left = 2 * i;
            size_t right = 2 * i + 1;
            size_t largest = i;

            if ((left <= w->n) && (w->r[left].dist >= p->dist))
                largest = left;
            if ((right <= w->n) && (w->r[right].dist >= w->r[largest].dist))
                largest = right;
            if (largest == i)
                break;

            /* Push p down the heap one level and bump one up. */
            w->r[i] = w->r[largest];
            w->r[largest] = *p;
            i = largest;
        }
    }
}


/******************************************************************************
*
* knn_insert_array - Insert the point P into the result array stored in the
*   workspace W.  The insertion uses a linear array search.  This is O(N).
*
******************************************************************************/

static void knn_insert_array (
    knn_workspace *w,
    const knn_result *p) {

    assert (p);
    assert (w);
    assert (w->r);
    assert (w->n > 0);
    assert (p->dist >= 0.0);
    assert (w->k <= w->n);

    /* Perform array insert if the array is not full. */
    if (w->k < w->n) {

        w->k++;

        if (p->dist > w->r[1].dist) {
            /* If the point is larger than the top, then swap. */
            w->r[w->k] = w->r[1];
            w->r[1] = *p;
        } else {
            /* Otherwise copy it to the last location. */
            w->r[w->k] = *p;
        }
        return;
    }

    /* At this point the array is full. */

    /* If the new distance is greater than the max, then discard it. */
    if (p->dist < w->r[1].dist) {
        size_t i, max_index;
        double max;

        /* Otherwise, replace the top with the new point p. */
        w->r[1] = *p;

        /* Search the array for a new maximum element. */
        max = p->dist;
        max_index = 1;

        for (i=2; i<=w->k; i++) {
            if (w->r[i].dist > max) {
                max = w->r[i].dist;
                max_index = i;
            }
        }

        /* Swap the top element with the new maximum. */
        if (max_index != 1) {
            knn_result swap;

            swap = w->r[max_index];
            w->r[max_index] = w->r[1];
            w->r[1] = swap;
        }
    }
}


/******************************************************************************
*
* kd_find - Find the point P in the KD-Tree TREE and return a pointer to it or
*   NULL if the point is not in the tree.  The comparison function must implement
*   greater than, less than, and equal (i.e., kd_compare).  The points must use
*   the default axis split plane.  This runs in O(lg (N)).
*
******************************************************************************/

void * kd_find (
    const kd_tree *tree,
    const void *p) {

    size_t i, next;
    int level = 0;


    assert (p);
    assert (tree);
    assert (tree->root);

    for (i=1; i<=tree->n; i=next) {
        void *q = (char *) (tree->root) + i * tree->elsize;
        const int axis = level % tree->dim;
        const int cmp = tree->cmp_fn (p, q, axis);

        if (cmp < 0) {
            next = HEAP_LEFT (i);
        } else if (cmp > 0) {
            next = HEAP_RIGHT (i);
        } else {
            return q;
        }
        level++;
    }

    return NULL;
}


/******************************************************************************
*
* kd_dist_l1 - Calculate the squared Manhattan distance (squared L1-Norm)
*   between points P and Q.
*
******************************************************************************/

double kd_dist_l1 (
    const void *a,
    const void *b) {

    const kd_point *p = (const kd_point *) a;
    const kd_point *q = (const kd_point *) b;
    double x = fabs (p->x[0] - q->x[0]);
    double y = fabs (p->x[1] - q->x[1]);

    return (x + y) * (x + y);
}


/******************************************************************************
*
* kd_dist_l2 - Calculate the squared Euclidean distance (squared L2-Norm)
*   between the points P and Q.
*
******************************************************************************/

double kd_dist_l2 (
    const void *a,
    const void *b) {

    const kd_point *p = (const kd_point *) a;
    const kd_point *q = (const kd_point *) b;
    double x = (p->x[0] - q->x[0]);
    double y = (p->x[1] - q->x[1]);

    return x * x + y * y;
}


/******************************************************************************
*
* kd_dist1 - Calculate the distance between the points P and Q that is
*   orthogonal to the splitting axis.
*
******************************************************************************/

double kd_dist1 (
    const void *a,
    const void *b,
    const int level) {

    const kd_point *p = (const kd_point *) a;
    const kd_point *q = (const kd_point *) b;
    const int axis = level % 2;

    return p->x[axis] - q->x[axis];
}


/******************************************************************************
*
* kd_compare - Compare the metric distance between two KD-Points P and Q for
*   the dimension given by AXIS.  Can be used with kd_find.
*
* Return: +1 if p > q,
*         -1 if p < q,
*          0 if p = q.
*
******************************************************************************/

int kd_compare (
    const void *a,
    const void *b,
    const int axis) {

    const kd_point *p = (const kd_point *) a;
    const kd_point *q = (const kd_point *) b;


    if (p->x[axis] > q->x[axis]) return  1;
    if (p->x[axis] < q->x[axis]) return -1;

    return 0;
}


/******************************************************************************
*
* kd_greater_than - Compare the metric distance between two KD-Points P and Q
*   for the dimension given by AXIS using the comparison 'greater than'.
*
* Return: +1 if p > q,
*          0 if p <= q.
*
******************************************************************************/

int kd_greater_than (
    const void *a,
    const void *b,
    const int axis) {

    const kd_point *p = (const kd_point *) a;
    const kd_point *q = (const kd_point *) b;

    return p->x[axis] > q->x[axis];
}


/******************************************************************************
*
* kd_swap - Swap two KD-Points P and Q.
*
******************************************************************************/

void kd_swap (
    void *a,
    void *b) {

    kd_point *p = (kd_point *) a;
    kd_point *q = (kd_point *) b;
    kd_point t;

    /* Casting the pointer to the kd_point type allows the compiler to make assumptions
       of the memory alignment and optimize the following memory swap.  Otherwise, it
       would require a slow byte for byte swap. */
    t = *p;
    *p = *q;
    *q = t;
}


/******************************************************************************
*
* kd_find_spread - Return the dimension of the data with the maximum estimated
*   variance.  This will be the axis to choose the split plane.
*
******************************************************************************/

int kd_find_spread (
    const void *pp,
    const size_t n,
    const int level) {

    kd_point *p = (kd_point *) pp;
    int d, md, ndim;
    double ms, s;
    size_t i;


    ndim = 2;

    md = 0;
    ms = -1.0;

    for (d=0; d<ndim; d++) {
        double min, max;

        min = max = p[0].x[d];

        for (i=1; i<n; i++) {
            if (p[i].x[d] > max) max = p[i].x[d];
            if (p[i].x[d] < min) min = p[i].x[d];
        }
        s = max - min;

        if (s > ms) {
            md = d;
            ms = s;
        }
    }

    return md;
}


/******************************************************************************
*
* kd_search_linear - Search for the nearest neighbor to the point P in the array
*   given by Q of length N using a sequential (linear) search and return it.
*   If DIST_MAX is not zero, then a maximum (squared) distance of DIST_MAX will
*   be searched.  If DIST is not NULL, then the (squared) distance between both
*   points will be returned in it.  This runs in O(N).
*
******************************************************************************/

kd_point * kd_search_linear (
    double *dist,
    const kd_point *p,
    const kd_point *q,
    const size_t n,
    kd_dist_fn_t dist_fn,
    const double dist_max) {

    const kd_point *nn = NULL;
    double dist_min;
    size_t i;


    assert (p);
    assert (q);
    assert (dist_fn);
    assert (dist_max >= 0.0);

    if (dist_max > 0.0) {
        dist_min = dist_max;
    } else {
        dist_min = DBL_MAX;
    }

    for (i=0; i<n; i++) {
        const double d = dist_fn (p, q + i);

        if (d < dist_min) {
            dist_min = d;
            nn = q + i;
        }
    }

    /* Return the distance to the nearest neighbor. */
    if (dist) *dist = dist_min;

    return (kd_point *) nn;
}


/******************************************************************************
*
* knn_search_linear - Search for the K nearest neighbors to the point P in the
*   array given by Q of length N search using a sequential (linear) search.
*   If DIST_MAX is not zero, then a maximum (squared) distance of DIST_MAX will
*   be searched.  This runs in O(N).
*
******************************************************************************/

void knn_search_linear (
    knn_workspace *w,
    const kd_point *p,
    const kd_point *q,
    const size_t n,
    kd_dist_fn_t dist_fn,
    const double dist_max) {

    double dist_min;
    size_t i;


    assert (w);
    assert (p);
    assert (q);
    assert (dist_fn);
    assert (dist_max >= 0.0);

    if (dist_max > 0.0) {
        dist_min = dist_max;
    } else {
        dist_min = DBL_MAX;
    }

    for (i=0; i<n; i++) {
        const double d = dist_fn (p, q + i);

        if (d < dist_min) {
            knn_result rp;
            rp.dist = d;
            rp.p = (void *) (q + i);

            w->insert_fn (w, &rp);
        }
    }
}


/******************************************************************************
*
* log2_floor - Return floor (lg (N)), which is the smallest exponent e such
*   that 2^e is less than N.  N must be positive.
*
* Timing: no opt (12-220 clks), -O2 (12-64 clks), builtin (4 clks)
*
* References: http://en.wikipedia.org/wiki/Find_first_set
*
******************************************************************************/

static int log2_floor (
    const unsigned int n) {

#if CPL_GNUC_PREREQ (3, 4)
    assert (n > 0);
    return (8 * sizeof (unsigned int) - 1) - __builtin_clz (n);
#else
    unsigned int m = n;
    int e = 0;


    assert (n > 0);

    while (m>>=1) e++;

    return e;
#endif
}
