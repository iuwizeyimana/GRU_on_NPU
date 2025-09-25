#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.
import argparse
from ml_dtypes import bfloat16
import numpy as np
import sys

from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
import aie.utils.trace as trace_utils
from aie.utils.trace import PortEvent
from aie.helpers.dialects.ext.scf import _for as range_
#from aie.iron.dtype import str_to_dtype


dtype_map = {
    "bf16": bfloat16, 
    "i8"  : np.int8,
    "i16" : np.int16, 
    "f32" : np.float32, 
    "i32" : np.int32,
}
def main():
    argparser = argparse.ArgumentParser(
        prog="AIE GRU design",
        description="Emits MLIR code for a GRU design of the given input size",
    )
    argparser.add_argument("--dev", type=str, choices=["npu", "npu2"], default="npu")
    argparser.add_argument("-M", type=int, default=256)
    argparser.add_argument("-K", type=int, default=256)
    argparser.add_argument("-N", type=int, default=256)
    argparser.add_argument("-m", type=int, default=64)
    argparser.add_argument("-k", type=int, default=64)
    argparser.add_argument("-n", type=int, default=32)
    argparser.add_argument(
        "--dtype_in", type=str, choices=["bf16", "i8", "i16"], default="i16"
    )
    argparser.add_argument(
        "--dtype_out",
        type=str,
        choices=["bf16", "i8", "i16", "f32", "i32"],
        default="i16",
    )
    argparser.add_argument("--trace_size", type=int, default=0)
    args = argparser.parse_args()
    my_matmul(
        args.dev,
        args.M,
        args.K,
        args.N,
        args.m,
        args.k,
        args.n,
        args.dtype_in,
        args.dtype_out,
        args.trace_size,
    )


def ceildiv(a, b):
    return (a + b - 1) // b


def my_matmul(
    dev,
    M,
    K,
    N,
    m,
    k,
    n,
    dtype_in_str,
    dtype_out_str,
    trace_size,
):

    assert M % m == 0
    assert K % k == 0
    assert (K*8) % k == 0
    assert N % n == 0

    # r, s, t are the dimensions required by the microkernel MAC instructions.
    if dev == "npu":
        if dtype_in_str == "bf16":
            r = 4
            s = 8
            t = 4
        elif dtype_in_str == "i8":
            r = 4
            s = 8
            t = 8
        elif dtype_in_str == "i16":
            r = 4
            s = 4
            t = 4
    else:
        if dtype_in_str == "bf16":
            r = 4
            s = 8
            t = 8
        elif dtype_in_str == "i8":
            r = 8
            s = 8
            t = 8
        elif dtype_in_str == "i16":
            r = 4
            s = 4
            t = 8

    assert m % r == 0
    assert k % s == 0
    assert n % t == 0

    vectorized = True
    enable_tracing = True if trace_size > 0 else False

    dtype_in = dtype_map[dtype_in_str]
    dtype_out = dtype_map[dtype_out_str]

    assert np.issubdtype(dtype_in, np.integer) == np.issubdtype(
        dtype_out, np.integer
    ), f"Input dtype ({dtype_in}) and output dtype ({dtype_out}) must either both be integral or both be float"
    assert (
        np.dtype(dtype_out).itemsize >= np.dtype(dtype_in).itemsize
    ), f"Output dtype ({dtype_out}) must be equal or larger to input dtype ({dtype_in})"

    I_sz = (M * K) + (M * K * 8) #ToDO fix this hard number -- K values of x are 8x K values of h
    W_sz = (K * N) + (K * 8 * N)
    O_sz = M * N

    M_div_m = M // m
    K_div_k_x = (K*8) // k
    K_div_k_h = K // k
    N_div_n = N // n
    tiles = M_div_m * N_div_n

    with mlir_mod_ctx() as ctx:

        if dev == "npu":
            dev_ty = AIEDevice.npu1_1col
        else:
            dev_ty = AIEDevice.npu2

        @device(dev_ty)

        def device_body():

            # sizes of input and output params
            i_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
            w_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
            o_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

            # AIE Core Function declarations
            func_type = "" if vectorized else "scalar_"
            zero = external_func(f"zero_{func_type}{dtype_out_str}", inputs=[o_ty])
            store_ = external_func(f"store_{func_type}{dtype_out_str}", 
				     inputs=[i_ty, o_ty])
            matmul_func_name = f"matmul_{func_type}{dtype_in_str}_{dtype_out_str}"
            matmul = external_func(
                matmul_func_name,
                inputs=[i_ty, w_ty, o_ty],
            )

            add_func_name = f"ewise_add_{func_type}{dtype_in_str}_{dtype_out_str}"
            add= external_func(
                add_func_name,
                inputs=[o_ty, o_ty, o_ty],
            )

            tanh_func_name = f"ewise_tanh_{func_type}{dtype_in_str}_{dtype_out_str}"
            tanh= external_func(
                tanh_func_name,
                inputs=[o_ty, o_ty, o_ty],
            )

            sig_func_name = f"ewise_sig_{func_type}{dtype_in_str}_{dtype_out_str}"
            sig= external_func(
                sig_func_name,
                inputs=[o_ty, o_ty, o_ty],
            )

            emul_func_name = f"ewise_mul_{func_type}{dtype_in_str}_{dtype_out_str}"
            emul = external_func(
                emul_func_name,
                inputs=[o_ty, o_ty, o_ty],
            )

            emulmin1_func_name = f"ewise_mulmin1_{func_type}{dtype_in_str}_{dtype_out_str}"
            emulmin1 = external_func(
                emulmin1_func_name,
                inputs=[o_ty, o_ty, o_ty],
            )





            # Tile declarations
            # (col, row)
            # row 0 for shim tiles
            shim_tile_in = tile(0, 0)
            shim_tile_wr = tile(1, 0)
            shim_tile_wn = tile(2, 0)
            shim_tile_wz = tile(3, 0)

            # row 1 for mem tiles
            mem_tile_in = tile(0, 1)
            mem_tile_wr = tile(1, 1)
            mem_tile_wn = tile(2, 1)
            mem_tile_wz = tile(3, 1)

            # rows 2-5 for compute tile
            compute_tile_a  = tile(0, 5)
            compute_tile_b  = tile(1, 5)
            compute_tile_c  = tile(2, 5)
            compute_tile_d  = tile(3, 5)
            compute_tile_r  = tile(0, 4)
            compute_tile_g  = tile(1, 4)
            compute_tile_n  = tile(2, 4)
            compute_tile_e  = tile(0, 3)
            compute_tile_z  = tile(1, 3)
            compute_tile_k  = tile(2, 3)
            compute_tile_f  = tile(0, 2)
            compute_tile_j  = tile(1, 2)
            compute_tile_hp = tile(2, 2)
            #compute_tile_s  = tile(3, 2 ) # stores a H value used by j later -- fix for reading H -- commenting out for issue 

            # AIE-array data movement with object fifos
            # 1. From shim to mem tiles
            #   Input X
            x_shim2mem = object_fifo("x_shim2mem", shim_tile_in, mem_tile_in, 2, i_ty)
            #   Input H
            h_shim2mem = object_fifo("h_shim2mem", shim_tile_in, mem_tile_in, 2, i_ty)
            #   Input w_ir & w_hr
            wir_shim2mem = object_fifo("wir_shim2mem", shim_tile_wr, mem_tile_wr, 2, w_ty)
            whr_shim2mem = object_fifo("whr_shim2mem", shim_tile_wr, mem_tile_wr, 2, w_ty)
            #   Input w_in & w_hn
            win_shim2mem = object_fifo("win_shim2mem", shim_tile_wn, mem_tile_wn, 2, w_ty)
            whn_shim2mem = object_fifo("whn_shim2mem", shim_tile_wn, mem_tile_wn, 2, w_ty)
            #   Input w_iz & w_hz
            wiz_shim2mem = object_fifo("wiz_shim2mem", shim_tile_wz, mem_tile_wz, 2, w_ty)
            whz_shim2mem = object_fifo("whz_shim2mem", shim_tile_wz, mem_tile_wz, 2, w_ty)


            # 2. From mem tiles to compute tiles
            #   Input X -> to cores a, d, e
            I_transformations = [
                (m // r, r * k), # advance to the next row block =>+r(num row blocks)*k(num cols) 
                (k // s, s),	 # advance to the next column block=>+s (num col blocks)
                (r, k),		 # in each r block; next row => +k
                (s, 1),		 # in each s block; next col => +1
            ]
            x_mem2comp = object_fifo(
                "x_mem2comp",
                mem_tile_in,
                [compute_tile_a, compute_tile_d, compute_tile_e],
                2, #depth
                i_ty,
                I_transformations,

            ) 
            object_fifo_link(x_shim2mem, x_mem2comp)

            #   Input H -> to cores b, c, f, j, hp
            h_mem2comp = object_fifo(
                "h_mem2comp",
                mem_tile_in,
                [compute_tile_b, compute_tile_c, compute_tile_f,
		compute_tile_j], #change compute_tile_s fix back to compute_tile_j
                2, #depth
                i_ty,
                I_transformations,

            ) 
            object_fifo_link(h_shim2mem, h_mem2comp)

            #   Input w_ir
            W_transformations = [
                (k // s, s * n), # advance to the next row block => +s(num row blocks)*n(num cols)
                (n // t, t),	 # advance to the next col block =>+t(num col blocks)
                (s, n),		 # in each s block; next row =>+n
                (t, 1),		 # in each t block; next col => +1
            ]
            wir_mem2comp = object_fifo(
                "wir_mem2comp",
                mem_tile_wr,
                compute_tile_a,
                2,
                w_ty,
                W_transformations,
            )
            object_fifo_link(wir_shim2mem, wir_mem2comp)

            #   Input w_hr
            whr_mem2comp = object_fifo(
                "whr_mem2comp",
                mem_tile_wr,
                compute_tile_b,
                2,
                w_ty,
                W_transformations,
            )
            object_fifo_link(whr_shim2mem, whr_mem2comp)

            #   Input w_hn
            whn_mem2comp = object_fifo(
                "whn_mem2comp",
                mem_tile_wn,
                compute_tile_c,
                2,
                w_ty,
                W_transformations,
            )
            object_fifo_link(whn_shim2mem, whn_mem2comp)

            #   Input w_in
            win_mem2comp = object_fifo(
                "win_mem2comp",
                mem_tile_wn,
                compute_tile_d,
                2,
                w_ty,
                W_transformations,
            )
            object_fifo_link(win_shim2mem, win_mem2comp)

            #   Input w_iz
            wiz_mem2comp = object_fifo(
                "wiz_mem2comp",
                mem_tile_wz,
                compute_tile_e,
                2,
                w_ty,
                W_transformations,
            )
            object_fifo_link(wiz_shim2mem, wiz_mem2comp)

            #   Input w_hz
            whz_mem2comp = object_fifo(
                "whz_mem2comp",
                mem_tile_wz,
                compute_tile_f,
                2,
                w_ty,
                W_transformations,
            )
            object_fifo_link(whz_shim2mem, whz_mem2comp)


            # 3. Across Compute tiles
            #   Output A
            memA = object_fifo("memA", compute_tile_a, compute_tile_r, 2, o_ty)
            #   Output B
            memB = object_fifo("memB", compute_tile_b, compute_tile_r, 2, o_ty)
            #   Output C
            memC = object_fifo("memC", compute_tile_c, compute_tile_g, 2, o_ty)
            #   Output D
            memD = object_fifo("memD", compute_tile_d, compute_tile_n, 2, o_ty)
            #   Output R
            memR = object_fifo("memR", compute_tile_r, compute_tile_g, 2, o_ty)
            #   Output G
            memG = object_fifo("memG", compute_tile_g, compute_tile_n, 2, o_ty)
            #   Output N
            memN = object_fifo("memN", compute_tile_n, compute_tile_k, 2, o_ty)
            #   Output E
            memE = object_fifo("memE", compute_tile_e, compute_tile_z, 2, o_ty)
            #   Output F
            memF = object_fifo("memF", compute_tile_f, compute_tile_z, 2, o_ty)
            #   Output Z
            memZ = object_fifo("memZ", compute_tile_z, [compute_tile_k, compute_tile_j], 2, o_ty)
            #   Output K
            memK = object_fifo("memK", compute_tile_k, compute_tile_hp, 2, o_ty)
            #   Output J
            memJ = object_fifo("memJ", compute_tile_j, compute_tile_hp, 2, o_ty)
            ##   Output S  used to transfer H values from tile F to tile J
            #memS = object_fifo("memS", compute_tile_s, compute_tile_j, 2, o_ty)

            
            # 4. Final output O from comp to mem to shim
            O_transformations = [
                (m // r, r * n), # advance to the next row block =>+r(num row blocks)*n(num cols)
                (r, t),
                (n // t, r * t), # advance to the next col block =>
                (t, 1),
            ]
 
            memO = object_fifo("memO", compute_tile_hp, mem_tile_in, 2, o_ty)
            outO = object_fifo(
                "outO",
                mem_tile_in,
                shim_tile_in,
                2,
                o_ty,
                O_transformations,
            )
            object_fifo_link(memO, outO)



            @core(compute_tile_a, f"mm.o", stack_size=0xD00)
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    for _ in range_(tiles) if tiles > 1 else range(1):  # issue #1547

                        elem_a = memA.acquire(ObjectFifoPort.Produce, 1)
                        zero(elem_a)

                        for _ in (
                            range_(K_div_k_x) if K_div_k_x > 1 else range(1)
                        ):  # issue #1547
                            elem_in_x = x_mem2comp.acquire(ObjectFifoPort.Consume, 1)
                            elem_in_wir = wir_mem2comp.acquire(ObjectFifoPort.Consume, 1)
                            matmul(elem_in_x, elem_in_wir, elem_a)
                            x_mem2comp.release(ObjectFifoPort.Consume, 1)
                            wir_mem2comp.release(ObjectFifoPort.Consume, 1)

                        memA.release(ObjectFifoPort.Produce, 1)

            @core(compute_tile_b, f"mm.o", stack_size=0xD00)
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    for _ in range_(tiles) if tiles > 1 else range(1):  # issue #1547

                        elem_b = memB.acquire(ObjectFifoPort.Produce, 1)
                        zero(elem_b)

                        for _ in (
                            range_(K_div_k_h) if K_div_k_h > 1 else range(1)
                        ):  # issue #1547
                            elem_in_h = h_mem2comp.acquire(ObjectFifoPort.Consume, 1)
                            elem_in_whr = whr_mem2comp.acquire(ObjectFifoPort.Consume, 1)
                            matmul(elem_in_h, elem_in_whr, elem_b)
                            h_mem2comp.release(ObjectFifoPort.Consume, 1)
                            whr_mem2comp.release(ObjectFifoPort.Consume, 1)

                        memB.release(ObjectFifoPort.Produce, 1)
            
            @core(compute_tile_c, f"mm.o", stack_size=0xD00)
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    for _ in range_(tiles) if tiles > 1 else range(1):  # issue #1547

                        elem_c = memC.acquire(ObjectFifoPort.Produce, 1)
                        zero(elem_c)

                        for _ in (
                            range_(K_div_k_h) if K_div_k_h > 1 else range(1)
                        ):  # issue #1547
                            elem_in_h = h_mem2comp.acquire(ObjectFifoPort.Consume, 1)
                            elem_in_whn = whn_mem2comp.acquire(ObjectFifoPort.Consume, 1)
                            matmul(elem_in_h, elem_in_whn, elem_c)
                            h_mem2comp.release(ObjectFifoPort.Consume, 1)
                            whn_mem2comp.release(ObjectFifoPort.Consume, 1)

                        memC.release(ObjectFifoPort.Produce, 1)

            @core(compute_tile_d, f"mm.o", stack_size=0xD00)
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    for _ in range_(tiles) if tiles > 1 else range(1):  # issue #1547

                        elem_d = memD.acquire(ObjectFifoPort.Produce, 1)
                        zero(elem_d)

                        for _ in (
                            range_(K_div_k_x) if K_div_k_x > 1 else range(1)
                        ):  # issue #1547
                            elem_in_x = x_mem2comp.acquire(ObjectFifoPort.Consume, 1)
                            elem_in_win = win_mem2comp.acquire(ObjectFifoPort.Consume, 1)
                            matmul(elem_in_x, elem_in_win, elem_d)
                            x_mem2comp.release(ObjectFifoPort.Consume, 1)
                            win_mem2comp.release(ObjectFifoPort.Consume, 1)

                        memD.release(ObjectFifoPort.Produce, 1)


            @core(compute_tile_r, f"sigadd.o", stack_size=0xD00)
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    for _ in range_(tiles) if tiles > 1 else range(1):  # issue #1547

                        elem_r = memR.acquire(ObjectFifoPort.Produce, 1)
                        elem_in_a = memA.acquire(ObjectFifoPort.Consume, 1)
                        elem_in_b = memB.acquire(ObjectFifoPort.Consume, 1)
                        sig(elem_in_a, elem_in_b, elem_r)
                        memA.release(ObjectFifoPort.Consume, 1)
                        memB.release(ObjectFifoPort.Consume, 1)

                        memR.release(ObjectFifoPort.Produce, 1)

            @core(compute_tile_g, f"mul.o", stack_size=0xD00)
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    for _ in range_(tiles) if tiles > 1 else range(1):  # issue #1547

                        elem_g = memG.acquire(ObjectFifoPort.Produce, 1)
                        elem_in_c = memC.acquire(ObjectFifoPort.Consume, 1)
                        elem_in_r = memR.acquire(ObjectFifoPort.Consume, 1)
                        emul(elem_in_c, elem_in_r, elem_g)
                        memC.release(ObjectFifoPort.Consume, 1)
                        memR.release(ObjectFifoPort.Consume, 1)

                        memG.release(ObjectFifoPort.Produce, 1)


            @core(compute_tile_n, f"tanhadd.o", stack_size=0xD00)
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    for _ in range_(tiles) if tiles > 1 else range(1):  # issue #1547

                        elem_n = memN.acquire(ObjectFifoPort.Produce, 1)
                        elem_in_d = memD.acquire(ObjectFifoPort.Consume, 1)
                        elem_in_g = memG.acquire(ObjectFifoPort.Consume, 1)
                        tanh(elem_in_d, elem_in_g, elem_n)
                        memD.release(ObjectFifoPort.Consume, 1)
                        memG.release(ObjectFifoPort.Consume, 1)

                        memN.release(ObjectFifoPort.Produce, 1)


            @core(compute_tile_e, f"mm.o", stack_size=0xD00)
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    for _ in range_(tiles) if tiles > 1 else range(1):  # issue #1547

                        elem_e = memE.acquire(ObjectFifoPort.Produce, 1)
                        zero(elem_e)

                        for _ in (
                            range_(K_div_k_x) if K_div_k_x > 1 else range(1)
                        ):  # issue #1547
                            elem_in_x = x_mem2comp.acquire(ObjectFifoPort.Consume, 1)
                            elem_in_wiz = wiz_mem2comp.acquire(ObjectFifoPort.Consume, 1)
                            matmul(elem_in_x, elem_in_wiz, elem_e)
                            x_mem2comp.release(ObjectFifoPort.Consume, 1)
                            wiz_mem2comp.release(ObjectFifoPort.Consume, 1)

                        memE.release(ObjectFifoPort.Produce, 1)

            @core(compute_tile_f, f"mm.o", stack_size=0xD00)
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    for _ in range_(tiles) if tiles > 1 else range(1):  # issue #1547

                        elem_f = memF.acquire(ObjectFifoPort.Produce, 1)
                        zero(elem_f)

                        for _ in (
                            range_(K_div_k_h) if K_div_k_h > 1 else range(1)
                        ):  # issue #1547
                            elem_in_h = h_mem2comp.acquire(ObjectFifoPort.Consume, 1)
                            elem_in_whz = whz_mem2comp.acquire(ObjectFifoPort.Consume, 1)
                            matmul(elem_in_h, elem_in_whz, elem_f)
                            h_mem2comp.release(ObjectFifoPort.Consume, 1)
                            whz_mem2comp.release(ObjectFifoPort.Consume, 1)

                        memF.release(ObjectFifoPort.Produce, 1)

            #@core(compute_tile_s, f"mm.o", stack_size=0xD00)
            #def core_body():
            #    for _ in range_(0xFFFFFFFF):
            #        for _ in range_(tiles) if tiles > 1 else range(1):  # issue #1547

            #            elem_s = memS.acquire(ObjectFifoPort.Produce, 1)

            #            for _ in (
            #                range_(K_div_k_h) if K_div_k_h > 1 else range(1)
            #            ):  # issue #1547
            #                elem_in_h = h_mem2comp.acquire(ObjectFifoPort.Consume, 1)
            #                store_(elem_in_h, elem_s)
            #                h_mem2comp.release(ObjectFifoPort.Consume, 1)

            #            memS.release(ObjectFifoPort.Produce, 1)

            @core(compute_tile_z, f"sigadd.o", stack_size=0xD00) #ToDo maybe move it beofr Z if needed
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    for _ in range_(tiles) if tiles > 1 else range(1):  # issue #1547

                        elem_z = memZ.acquire(ObjectFifoPort.Produce, 1)
                        elem_in_e = memE.acquire(ObjectFifoPort.Consume, 1)
                        elem_in_f = memF.acquire(ObjectFifoPort.Consume, 1)
                        sig(elem_in_e, elem_in_f, elem_z)
                        memE.release(ObjectFifoPort.Consume, 1)
                        memF.release(ObjectFifoPort.Consume, 1)

                        memZ.release(ObjectFifoPort.Produce, 1)

            @core(compute_tile_k, f"mulmin1.o", stack_size=0xD00) 
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    for _ in range_(tiles) if tiles > 1 else range(1):  # issue #1547

                        elem_k = memK.acquire(ObjectFifoPort.Produce, 1)
                        elem_in_n = memN.acquire(ObjectFifoPort.Consume, 1)
                        elem_in_z = memZ.acquire(ObjectFifoPort.Consume, 1)
                        emulmin1(elem_in_z, elem_in_n, elem_k) # Z must come first
                        memN.release(ObjectFifoPort.Consume, 1)
                        memZ.release(ObjectFifoPort.Consume, 1)

                        memK.release(ObjectFifoPort.Produce, 1)



            @core(compute_tile_j, f"mul.o", stack_size=0xD00)
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    for _ in range_(tiles) if tiles > 1 else range(1):  # issue #1547

                        elem_j = memJ.acquire(ObjectFifoPort.Produce, 1)
                        elem_in_z = memZ.acquire(ObjectFifoPort.Consume, 1)
                        #elem_in_s = memS.acquire(ObjectFifoPort.Consume, 1)
                        elem_in_h = h_mem2comp.acquire(ObjectFifoPort.Consume, 1)
                        #emul(elem_in_z, elem_in_s, elem_j) 
                        emul(elem_in_z, elem_in_h, elem_j) 
                        #memS.release(ObjectFifoPort.Consume, 1)
                        h_mem2comp.release(ObjectFifoPort.Consume, 1)

                        memZ.release(ObjectFifoPort.Consume, 1)
                        memJ.release(ObjectFifoPort.Produce, 1)


            @core(compute_tile_hp, f"add.o", stack_size=0xD00)
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    for _ in range_(tiles) if tiles > 1 else range(1):  # issue #1547

                        elem_out = memO.acquire(ObjectFifoPort.Produce, 1)
                        elem_in_k = memK.acquire(ObjectFifoPort.Consume, 1)
                        elem_in_j = memJ.acquire(ObjectFifoPort.Consume, 1)
                        add(elem_in_k, elem_in_j, elem_out)
                        memK.release(ObjectFifoPort.Consume, 1)
                        memJ.release(ObjectFifoPort.Consume, 1)

                        memO.release(ObjectFifoPort.Produce, 1)



            # To/from AIE-array data movement
            @runtime_sequence(
                np.ndarray[(I_sz,), np.dtype[dtype_in]], # x & h
                np.ndarray[(W_sz,), np.dtype[dtype_in]], # w_ir & w_hr
                np.ndarray[(W_sz,), np.dtype[dtype_in]], # w_in & w_hn
                np.ndarray[(W_sz,), np.dtype[dtype_in]], # w_iz & w_hz
                np.ndarray[(O_sz,), np.dtype[dtype_out]],
            )
            def sequence(I, WR, WN, WZ, O):

                # only do 4 tile rows at a time before synchronizing, so we can reuse BDs
                rows_per_block = 4
                h_offset = M * K * 8
                wh_offset = N * K * 8
                for tile_row_block in range(ceildiv(M_div_m, rows_per_block)):
                    # we only sync on half the BDs before reusing them, so the other half can concurrently keep running
                    # that's what this loop is for
                    for pingpong in [0, 1]:
                        O_row_offset = (
                            tile_row_block * rows_per_block * m * N
                            + pingpong * rows_per_block // 2 * m * N
                        )
                        row_base = (
                            tile_row_block * rows_per_block
                            + pingpong * rows_per_block // 2
                        )
                        bd_id_base = 8 * pingpong
                        num_tile_rows = min([rows_per_block // 2, M_div_m - row_base])
                        if num_tile_rows <= 0:
                            # At the very last iteration, we may not need a 'pong' iteration
                            break
                        npu_dma_memcpy_nd(
                            metadata=outO,
                            bd_id=bd_id_base,
                            mem=O,
                            offsets=[0, 0, 0, O_row_offset],
                            sizes=[num_tile_rows, N // n, m, n],
                            strides=[m * N, n, N, 1],
                        )
                        for tile_row in range(num_tile_rows):
                            I_row_offset_x = (row_base + tile_row) * m * K * 8
                            I_row_offset_h = (row_base + tile_row) * m * K
                            # X
                            npu_dma_memcpy_nd(
                                metadata=x_shim2mem,
                                bd_id=bd_id_base + 2 * tile_row + 1,
                                mem=I,
                                offsets=[0, 0, 0, I_row_offset_x],
                                sizes=[1, (K*8) // k, m, k],
                                strides=[0, k, (K*8), 1],
                            )
                            # H
                            npu_dma_memcpy_nd(
                                metadata=h_shim2mem,
                                bd_id=bd_id_base + 2 * tile_row + 2,
                                mem=I,
                                offsets=[0, 0, 0, (I_row_offset_h+h_offset)],
                                sizes= [1, K // k, m, k],
                                strides=[0, k, K, 1],
                            )



                            Wx_sizes = [N // n, (K*8) // k, k, n]
                            Wx_strides = [n, k * N, N, 1]
                            
                            Wh_sizes = [N // n, K // k, k, n]
                            Wh_strides = [n, k * N, N, 1]
                            
			    # W_IR
                            npu_dma_memcpy_nd(
                                metadata=wir_shim2mem,
                                bd_id=bd_id_base + 2 * tile_row + 1, #weights and inputs has # shim tiles -- can have #BDs
                                mem=WR,
                                sizes=Wx_sizes,
                                strides=Wx_strides,
                            )
			    # W_HR
                            npu_dma_memcpy_nd(
                                metadata=whr_shim2mem,
                                bd_id=bd_id_base + 2 * tile_row + 2,
                                mem=WR,
                                offsets = [0, 0, 0, wh_offset],
                                sizes=Wh_sizes,
                                strides=Wh_strides,
                            )
			    # W_IN
                            npu_dma_memcpy_nd(
                                metadata=win_shim2mem,
                                bd_id=bd_id_base + 2 * tile_row + 1, #weights and inputs has # shim tiles -- can have #BDs
                                mem=WN,
                                sizes=Wx_sizes,
                                strides=Wx_strides,
                            )
			    # W_HN
                            npu_dma_memcpy_nd(
                                metadata=whn_shim2mem,
                                bd_id=bd_id_base + 2 * tile_row + 2,
                                mem=WN,
                                offsets = [0, 0, 0, wh_offset],
                                sizes=Wh_sizes,
                                strides=Wh_strides,
                            )
			    # W_IZ
                            npu_dma_memcpy_nd(
                                metadata=wiz_shim2mem,
                                bd_id=bd_id_base + 2 * tile_row + 1, #weights and inputs has # shim tiles -- can have #BDs
                                mem=WZ,
                                sizes=Wx_sizes,
                                strides=Wx_strides,
                            )
			    # W_HZ
                            npu_dma_memcpy_nd(
                                metadata=whz_shim2mem,
                                bd_id=bd_id_base + 2 * tile_row + 2,
                                mem=WZ,
                                offsets = [0, 0, 0, wh_offset],
                                sizes=Wh_sizes,
                                strides=Wh_strides,
                            )
                        if tile_row_block > 0 or (tile_row_block == 0 and pingpong > 0):
                            dma_wait(outO)
                dma_wait(outO)

    print(ctx.module)


if __name__ == "__main__":
    main()
else:
    print("Not meant to be imported")
    sys.exit(1)
