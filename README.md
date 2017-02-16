# kdtree
KD-Tree and Nearest Neighbor Search

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
