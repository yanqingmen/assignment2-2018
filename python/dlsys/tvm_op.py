from __future__ import absolute_import, print_function

import tvm
import numpy as np
import topi

# Global declarations of environment.

# llvm
tgt_host="llvm"
# llvm, cuda, opencl, metal
# Change it to respective GPU if gpu is enabled Ex: cuda, opencl
tgt="llvm"


def make_elemwise_add(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.placeholder(shape, dtype=dtype, name="B")
    C = tvm.compute(A.shape, lambda *i: A(*i) + B(*i))

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_elemwise_mul(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.placeholder(shape, dtype=dtype, name="B")
    C = tvm.compute(A.shape, lambda *i: A(*i) * B(*i))

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_elemwise_add_by_const(shape, const_k, tgt, tgt_host, func_name,
                               dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    C = tvm.compute(A.shape, lambda *i: A(*i) + const_k)

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_elemwise_mul_by_const(shape, const_k, tgt, tgt_host, func_name,
                            dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    C = tvm.compute(A.shape, lambda *i: A(*i) * const_k)

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_relu(shape, tgt, tgt_host, func_name, dtype="float32"):
    """Hint: use tvm.max, tvm.const(0, A.dtype)"""
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.const(0, A.dtype)
    C = tvm.compute(A.shape, lambda *i: tvm.max(A(*i), B))

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_relu_gradient(shape, tgt, tgt_host, func_name, dtype="float32"):
    """Hint: use tvm.select"""
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.placeholder(shape, dtype=dtype, name="B")
    C = tvm.compute(A.shape, lambda *i: B(*i) * tvm.select(A(*i) > 0, 1, 0))

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_matrix_mul(shapeA, transposeA, shapeB, transposeB, tgt, tgt_host,
                    func_name, dtype="float32"):
    """Hint: treat 4 cases of transposeA, transposeB separately"""
    """Hint: for tvm schedule, use split, reorder, vectorize, parallel"""
    """Hint: debug tvm schedule using tvm.lower"""
    assert len(shapeA) == 2 and len(shapeB) == 2
    A = tvm.placeholder(shapeA, dtype=dtype, name="A")
    B = tvm.placeholder(shapeB, dtype=dtype, name="B")

    if not transposeA and not transposeB:
        in_a, out_a = shapeA
        in_b, out_b = shapeB
        k = tvm.reduce_axis((0, out_a), name='k')
        trans_b = topi.transpose(B)
        matmul = tvm.compute((in_a, out_b),
                             lambda i, j: tvm.sum(A[i, k] * trans_b[j, k], axis=k))
    elif transposeA and not transposeB:
        out_a, in_a = shapeA
        in_b, out_b = shapeB
        k = tvm.reduce_axis((0, out_a), name='k')
        trans_a = topi.transpose(A)
        trans_b = topi.transpose(B)
        matmul = tvm.compute((in_a, out_b),
                             lambda i, j: tvm.sum(trans_a[i, k] * trans_b[j, k], axis=k))
    elif not transposeA and transposeB:
        in_a, out_a = shapeA
        out_b, in_b = shapeB
        k = tvm.reduce_axis((0, out_a), name='k')
        matmul = tvm.compute((in_a, out_b),
                             lambda i, j: tvm.sum(A[i, k] * B[j, k], axis=k))
    elif transposeA and transposeB:
        out_a, in_a = shapeA
        out_b, in_b = shapeB
        k = tvm.reduce_axis((0, out_a), name='k')
        trans_a = topi.transpose(A)
        matmul = tvm.compute((in_a, out_b),
                             lambda i, j: tvm.sum(trans_a[i, k] * B[j, k], axis=k))

    s = tvm.create_schedule(matmul.op)
    f = tvm.build(s, [A, B, matmul], tgt, target_host=tgt_host, name=func_name)
    return f


def make_conv2d(shapeX, shapeF, tgt, tgt_host, func_name, dtype="float32"):
    assert(shapeX[1] == shapeF[1])
    """Hint: use tvm.reduce_axis, tvm.sum"""
    """Hint: go by conv2d definition. Treat stride=1, padding=0 case only."""
    """For a challenge, treat the general case for stride and padding."""
    A = tvm.placeholder(shapeX, dtype=dtype, name="A")
    conv = tvm.placeholder(shapeF, dtype=dtype, name="conv")
    C = topi.nn.conv2d(A, conv, 1, 0)

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, conv, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_matrix_softmax(shape, tgt, tgt_host, func_name, dtype="float32"):
    """Hint: use tvm.reduce_axis, tvm.sum, tvm.max, tvm.exp"""
    """Hint: do not reuse the same reduction axis j."""
    """Hint: implement the following version for better stability
        e_x = np.exp(x - np.max(x))
        softmax(x)= e_x / e_x.sum()
    """
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    C = topi.nn.softmax(A, axis=1)
    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_matrix_softmax_cross_entropy(shape, tgt, tgt_host, func_name,
                                      dtype="float32"):
    """Hint: output shape should be (1,)"""
    y = tvm.placeholder(shape, dtype=dtype, name="y")
    y_ = tvm.placeholder(shape, dtype=dtype, name="y_")
    t = -topi.sum(y_ * topi.log(topi.nn.softmax(y)), axis=1)
    c = topi.sum(t, keepdims=True)/shape[0]
    s = tvm.create_schedule(c.op)
    f = tvm.build(s, [y, y_, c], tgt, target_host=tgt_host, name=func_name)
    return f


def make_reduce_sum_axis_zero(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    C = topi.sum(A, axis=0, keepdims=False)

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_broadcast_to(shape, to_shape, tgt, tgt_host, func_name,
                      dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    C = topi.broadcast_to(A, to_shape)

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_sgd_update(shape, learning_rate, tgt, tgt_host, func_name,
                    dtype="float32"):
    X = tvm.placeholder(shape, dtype=dtype, name="A")
    grad = tvm.placeholder(shape, dtype=dtype, name="grad")
    Y = tvm.compute(shape, lambda *i: X(*i) - learning_rate * grad(*i))

    s = tvm.create_schedule(Y.op)
    f = tvm.build(s, [X, grad, Y], tgt, target_host=tgt_host, name=func_name)
    return f