/* cpl_spec.h -- Portable function and variable attribute support.

   This software is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. */

/* Provide definition of size_t type if requested. */
#ifdef CPL_NEED_SIZE_T
# define __need_size_t  /* Tell gcc stddef.h we only want size_t. */
# ifdef __cplusplus
#  include <cstddef>
# else
#  include <stddef.h>
# endif
# undef __need_size_t
#endif

/* Provide definition of FILE type if requested. */
#ifdef CPL_NEED_FILE_T
# ifdef __cplusplus
#  include <cstdio>
# else
#  include <stdio.h>
# endif
#endif

/* Provide definition of time_t type if requested. */
#ifdef CPL_NEED_TIME_T
# ifdef __cplusplus
#  include <ctime>
# else
#  include <time.h>
# endif
#endif

/* Provide definition of fixed-width integer types if requested. */
#ifdef CPL_NEED_FIXED_WIDTH_T
/* Define types for fixed-width integers (C99). */
# if defined (_MSC_VER) && (_MSC_VER <= 1500) /* Fix for MSVC */
typedef signed __int8  int8_t;
typedef signed __int16 int16_t;
typedef signed __int32 int32_t;
typedef signed __int64 int64_t;
typedef unsigned __int8 uint8_t;
typedef unsigned __int16 uint16_t;
typedef unsigned __int32 uint32_t;
typedef unsigned __int64 uint64_t;
# else
#  include <stdint.h>
# endif
#endif

#ifndef CPL_SPEC_H
#define CPL_SPEC_H

/* Macros for C Linkage specification. */
#ifdef __cplusplus
#  define CPL_CLINKAGE_START  extern "C" {
#  define CPL_CLINKAGE_END    }
#else
#  define CPL_CLINKAGE_START /* empty */
#  define CPL_CLINKAGE_END   /* empty */
#endif

/* Test for gcc >= maj.min. */
#if defined (__GNUC__) && defined (__GNUC_MINOR__)
# define CPL_GNUC_PREREQ(major, minor) ((__GNUC__ << 16) + __GNUC_MINOR__ >= ((major) << 16) + (minor))
#else
# define CPL_GNUC_PREREQ(major, minor)  0
#endif

/* Add GNU attribute support. */
#ifndef __attribute__
# if defined (__STRICT_ANSI__) || !CPL_GNUC_PREREQ (2, 5)
#  define __attribute__(spec) /* empty */
# endif
#endif

/* Use __extension__ to suppress -pedantic warnings about GCC extensions. */
#if !CPL_GNUC_PREREQ (2, 8)
# define __extension__ /* empty */
#endif

/* Tell the compiler to use the smallest alignment or space for the struct, union, or variable.
   This may be used with __gcc_struct__ on windows platforms. */
#if CPL_GNUC_PREREQ (2, 7)
# define CPL_ATTRIBUTE_PACKED __attribute__ ((__packed__))
#else
# define CPL_ATTRIBUTE_PACKED /* empty */
#endif

/* On windows platforms tell the compiler to pack the struct differently from the Microsoft ABI,
   which may add extra padding.  Same as -mno-ms-bitfields. */
#if CPL_GNUC_PREREQ (3, 4)
# define CPL_ATTRIBUTE_GCC_STRUCT __attribute__ ((__gcc_struct__))
#else
# define CPL_ATTRIBUTE_GCC_STRUCT /* empty */
#endif

/* Tell the compiler that the nth function parameter should be a non-NULL pointer.
   This allows some optimization and a warning if a NULL pointer is used if -Wnonnull is enabled. */
#if CPL_GNUC_PREREQ (3, 3)
# define CPL_ATTRIBUTE_NONNULL(n)  __attribute__ ((__nonnull__ (n)))
#else
# define CPL_ATTRIBUTE_NONNULL(n) /* empty */
#endif

/* Tell the compiler that the nth and mth function parameters should be non-NULL pointers. */
#if CPL_GNUC_PREREQ (3, 3)
# define CPL_ATTRIBUTE_NONNULL2(n, m)  __attribute__ ((__nonnull__ (n, m)))
#else
# define CPL_ATTRIBUTE_NONNULL2(n, m) /* empty */
#endif

/* Tell the compiler that all function parameters should be non-NULL pointers.
   This should be used instead of the previous macros to avoid compiler warnings
   about empty macro arguments. */
#if CPL_GNUC_PREREQ (3, 3)
# define CPL_ATTRIBUTE_NONNULL_ALL  __attribute__ ((__nonnull__))
#else
# define CPL_ATTRIBUTE_NONNULL_ALL /* empty */
#endif

/* Tell the compiler that the function return value should be a non-NULL pointer. */
#if CPL_GNUC_PREREQ (4, 9)
# define CPL_ATTRIBUTE_RETURNS_NONNULL  __attribute__ ((__returns_nonnull__))
#else
# define CPL_ATTRIBUTE_RETURNS_NONNULL /* empty */
#endif

/* Tell the compiler that the last argument of the function should be NULL. */
#if CPL_GNUC_PREREQ (3, 5)
# define CPL_ATTRIBUTE_SENTINEL  __attribute__ ((__sentinel__))
#else
# define CPL_ATTRIBUTE_SENTINEL /* empty */
#endif

/* Tell the compiler that the function, type, or variable is possibly unused and not to issue a warning. */
#if CPL_GNUC_PREREQ (3, 4)
# define CPL_ATTRIBUTE_UNUSED  __attribute__ ((__unused__))
#else
# define CPL_ATTRIBUTE_UNUSED /* empty */
#endif

/* Tell the compiler to inline the function even if no optimization level was specified. */
#if CPL_GNUC_PREREQ (3, 1)
# define CPL_ATTRIBUTE_ALWAYS_INLINE  __attribute__ ((__always_inline__))
#else
# define CPL_ATTRIBUTE_ALWAYS_INLINE /* empty */
#endif

/* Tell the compiler not to inline the function even if no optimization level was specified. */
#if CPL_GNUC_PREREQ (3, 1)
# define CPL_ATTRIBUTE_NOINLINE  __attribute__ ((__noinline__))
#else
# define CPL_ATTRIBUTE_NOINLINE /* empty */
#endif

/* Issue a warning if the caller of the function does not use its return value. */
#if CPL_GNUC_PREREQ (3, 4)
# define CPL_ATTRIBUTE_WARN_UNUSED_RESULT  __attribute__ ((__warn_unused_result__))
#else
# define CPL_ATTRIBUTE_WARN_UNUSED_RESULT /* empty */
#endif

/* "pure" means that the function has no effects except the return value, which
   depends only on the function parameters and/or global variables. */
#if CPL_GNUC_PREREQ (3, 0)
# define CPL_ATTRIBUTE_PURE  __attribute__ ((__pure__)) CPL_ATTRIBUTE_WARN_UNUSED_RESULT
#else
# define CPL_ATTRIBUTE_PURE /* empty */
#endif

/* "const" means a function does nothing but examine its arguments and gives
   a return value, it doesn't read or write any memory (neither global nor
   pointed to by arguments), and has no other side-effects.  This is more
   restrictive than "pure". */
#if CPL_GNUC_PREREQ (2, 96)
# define CPL_ATTRIBUTE_CONST  __attribute__ ((__const__)) CPL_ATTRIBUTE_WARN_UNUSED_RESULT
#else
# define CPL_ATTRIBUTE_CONST /* empty */
#endif

/* Tell the compiler that the return value of the function points to memory,
   where the size is given by one of the function parameters. */
#if CPL_GNUC_PREREQ (4, 3)
# define CPL_ATTRIBUTE_ALLOC_SIZE(n)  __attribute__ ((__alloc_size__ (n)))
#else
# define CPL_ATTRIBUTE_ALLOC_SIZE(n) /* empty */
#endif

/* Tell the compiler that the return value of the function points to memory,
   where the size is given by the product of two function parameters. */
#if CPL_GNUC_PREREQ (4, 3)
# define CPL_ATTRIBUTE_ALLOC_SIZE2(n, elsize)  __attribute__ ((__alloc_size__ (n, elsize)))
#else
# define CPL_ATTRIBUTE_ALLOC_SIZE2(n, elsize) /* empty */
#endif

/* Tell the compiler that any non-NULL pointer returned by the function cannot
   alias any other pointer when the function returns and the memory has undefined
   content.  realloc-like functions do not have this property as the memory pointed
   to does not have undefined content. */
#if CPL_GNUC_PREREQ (3, 0)
# define CPL_ATTRIBUTE_MALLOC  __attribute__ ((__malloc__))
#else
# define CPL_ATTRIBUTE_MALLOC /* empty */
#endif

/* Tell the compiler that the function is to be optimized more aggresively and
   grouped together to improve locality. */
#if CPL_GNUC_PREREQ (4, 3)
# define CPL_ATTRIBUTE_HOT  __attribute__ ((__hot__))
#else
# define CPL_ATTRIBUTE_HOT /* empty */
#endif

/* Tell the compiler that the function is unlikely to execute and to optimize
   for size, group functions together to improve locality, and optimize branch
   predictions. */
#if CPL_GNUC_PREREQ (4, 3)
# define CPL_ATTRIBUTE_COLD  __attribute__ ((__cold__))
#else
# define CPL_ATTRIBUTE_COLD /* empty */
#endif

/* Tell the compiler to assume that the function does not return. */
#if CPL_GNUC_PREREQ (2, 7)
# define CPL_ATTRIBUTE_NORETURN  __attribute__ ((__noreturn__))
#else
# define CPL_ATTRIBUTE_NORETURN /* empty */
#endif

/* Tell the compiler that the function takes printf style arguments which should
   be type-checked against a format string.  m is the number of the "format string"
   parameter and n is the number of the first variadic parameter. */
#if CPL_GNUC_PREREQ (2, 96)
# define CPL_ATTRIBUTE_PRINTF(m, n)  __attribute__ ((__format__ (__printf__, m, n))) CPL_ATTRIBUTE_NONNULL (m)
#else
# define CPL_ATTRIBUTE_PRINTF(m, n) /* empty */
#endif

/* The type attribute aligned specifies the minimum alignment in bytes for
   the variables of the given type. */
#if CPL_GNUC_PREREQ (2, 7)
# define CPL_ATTRIBUTE_ALIGNED(x)  __attribute__ ((__aligned__ (x)))
#else
# define CPL_ATTRIBUTE_ALIGNED(x) /* empty */
#endif

/* Allows the compiler to assume that the pointer PTR is at least N-bytes aligned. */
#if CPL_GNUC_PREREQ (4, 7)
# define CPL_ASSUME_ALIGNED(ptr, n) __builtin_assume_aligned (ptr, n)
#else
# define CPL_ASSUME_ALIGNED(ptr, n) (ptr)
#endif

/* An empty "throw ()" means the function doesn't throw any C++ exceptions,
   which can save some stack frame info in applications.

   Note that CPL_ATTRIBUTE_NOTHROW must be given on any inlines the same as on
   their prototypes (for g++ at least, where they're used together).  Note also
   that g++ 3.0 requires that CPL_ATTRIBUTE_NOTHROW is given before other
   attributes like CPL_ATTRIBUTE_PURE. */

/* Tell the compiler that the function does not throw an exception. */
#ifdef __GNUC__
# if !defined (__cplusplus) && CPL_GNUC_PREREQ (3, 3)
#  define CPL_ATTRIBUTE_NOTHROW  __attribute__ ((__nothrow__))
# else
#  if defined (__cplusplus) && CPL_GNUC_PREREQ (2, 8)
#   define CPL_ATTRIBUTE_NOTHROW throw ()
#  else
#   define CPL_ATTRIBUTE_NOTHROW /* empty */
#  endif
# endif
#else
# define CPL_ATTRIBUTE_NOTHROW /* empty */
#endif

/* Use __builtin_constant_p to determine if a value is known to be a constant at compile-time. */
#if !CPL_GNUC_PREREQ (2, 0)
# define __builtin_constant_p(x) 0
#endif

/* Give the compiler branch prediction information. */
#if CPL_GNUC_PREREQ (3, 0)
# define CPL_LIKELY(cond)   __builtin_expect ((cond) != 0, 1)
# define CPL_UNLIKELY(cond) __builtin_expect ((cond) != 0, 0)
#else
# define CPL_LIKELY(cond)   (cond)
# define CPL_UNLIKELY(cond) (cond)
#endif

/* Tell the compiler that for the life of the pointer, only it or a value derived
   from it will be used to access the object to which it points. */
#if defined (_MSC_VER)
# define CPL_RESTRICT __restrict
#elif CPL_GNUC_PREREQ (3, 1)
# define CPL_RESTRICT __restrict__
#else
# define CPL_RESTRICT /* empty */
#endif

/* CPL_PTR_CAST allows use of static_cast in C/C++ headers, so macros are clean to "g++ -Wold-style-cast". */
#ifdef __cplusplus
#define CPL_PTR_CAST(T, expr)  (static_cast<T> (expr))
#else
#define CPL_PTR_CAST(T, expr)  ((T) (expr))
#endif

#endif /* CPL_SPEC_H */
