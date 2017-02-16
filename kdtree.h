/* kdtree.h -- Header file for kdtree.c

   This software is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty
   of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. */

#ifndef KDTREE_H
#define KDTREE_H

#define CPL_NEED_SIZE_T  /* Tell cpl_spec.h we need size_t. */
#include "cpl_spec.h"


/* KD-Tree Point Structure */
typedef struct {
    double x[2];
    double data;
} kd_point;

/* K-Nearest Neighbor Result Structure */
typedef struct {
    double dist;
    void *p;
} knn_result;


/* Type of function that implements GREATER THAN between two nodes. */
typedef int (* kd_compare_fn_t) (const void *p, const void *q, const int axis);

/* Type of function used to calculate the distance between two nodes. */
typedef double (* kd_dist_fn_t) (const void *p, const void *q);

/* Type of function to calculate distance orthogonal to the splitting axis. */
typedef double (* kd_dist1_fn_t) (const void *p, const void *q, const int level);

/* Type of function to set the node-splitting axis. */
typedef void (* kd_set_axis_fn_t) (void *p, const int axis);

/* Type of function used to find axis to split on. */
typedef int (* kd_find_axis_fn_t) (const void *p, const size_t n, const int level);

/* Type of function used to swap two nodes. */
typedef void (* kd_swap_fn_t) (void *p, void *q);


/* Opaque KD-Tree Structure */
typedef struct kd_tree_struct kd_tree;

/* Opaque K-Nearest Neighbor Workspace Structure */
typedef struct knn_workspace_struct knn_workspace;


/*************************** KD-Tree API Functions ***************************/

CPL_CLINKAGE_START

/* KD-Tree Constructor/Destructor */
kd_tree * kd_alloc (const size_t) CPL_ATTRIBUTE_WARN_UNUSED_RESULT;
void kd_free (kd_tree *);

/* KD-Tree Set/Get Functions */
void kd_set_point_type (kd_tree *, const size_t, const int);
void kd_set_cmp_fn (kd_tree *, kd_compare_fn_t);
void kd_set_swap_fn (kd_tree *, kd_swap_fn_t);
void kd_set_dist_fn (kd_tree *, kd_dist_fn_t, kd_dist1_fn_t);
void kd_set_axis_fn (kd_tree *, kd_set_axis_fn_t, kd_find_axis_fn_t);
void * kd_get_data (const kd_tree *) CPL_ATTRIBUTE_PURE;
size_t kd_get_count (const kd_tree *) CPL_ATTRIBUTE_PURE;
int kd_get_dim (const kd_tree *) CPL_ATTRIBUTE_PURE;

/* KD-Tree Initialization */
void kd_build (kd_tree *, void *);

/* Nearest-neighbor Search */
void * kd_search (double *, const void *, const kd_tree *, const double) CPL_ATTRIBUTE_WARN_UNUSED_RESULT;
void * kd_search_xy (double *, const double, const double, const kd_tree *, const double) CPL_ATTRIBUTE_WARN_UNUSED_RESULT;

/* K-Nearest Neighbor Search O(lg (N)) */
knn_workspace * knn_alloc (const size_t) CPL_ATTRIBUTE_WARN_UNUSED_RESULT;
void knn_free (knn_workspace *);
knn_result * knn_get_result_data (const knn_workspace *) CPL_ATTRIBUTE_PURE;
size_t knn_get_result_count (const knn_workspace *) CPL_ATTRIBUTE_PURE;
void knn_search (knn_workspace *, const void *, const kd_tree *);
void knn_search_xy (knn_workspace *, const double, const double, const kd_tree *);

/* Exact Search O(ln (N)) */
void * kd_find (const kd_tree *, const void *) CPL_ATTRIBUTE_WARN_UNUSED_RESULT;

/* Linear Search O(N) */
kd_point * kd_search_linear (double *, const kd_point *, const kd_point *, const size_t, kd_dist_fn_t, const double) CPL_ATTRIBUTE_WARN_UNUSED_RESULT;
void knn_search_linear (knn_workspace *, const kd_point *, const kd_point *, const size_t, kd_dist_fn_t, const double);

/* Comparison and Distance Functions for kd_point Structure */
int kd_compare (const void *, const void *, const int) CPL_ATTRIBUTE_PURE;
int kd_greater_than (const void *, const void *, const int) CPL_ATTRIBUTE_PURE;
double kd_dist_l1 (const void *, const void *) CPL_ATTRIBUTE_PURE;
double kd_dist_l2 (const void *, const void *) CPL_ATTRIBUTE_PURE;
double kd_dist1 (const void *, const void *, const int) CPL_ATTRIBUTE_PURE;
int kd_find_spread (const void *, const size_t, const int) CPL_ATTRIBUTE_PURE;
void kd_swap (void *, void *);


CPL_CLINKAGE_END

#endif /* KDTREE_H */
