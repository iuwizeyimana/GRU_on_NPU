//===- scale.cc -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>


#ifndef DIM_M
#define DIM_M 64
#endif
#ifndef DIM_N
#define DIM_N 64
#endif

template <typename T_in, typename T_out, const int N>
void eltwise_vmul(T_in *a, T_in *b, T_out *c) {
  constexpr int vec_factor = 16;
  event0();
  T_in *__restrict pA1 = a;
  T_in *__restrict pB1 = b;
  T_out *__restrict pC1 = c;
  const int F = N / vec_factor;
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_MIN_ITERATION_COUNT(16)
  for (int i = 0; i < F; i++) {
    aie::vector<T_in, vec_factor> A0 = aie::load_v<vec_factor>(pA1);
    pA1 += vec_factor;
    aie::vector<T_in, vec_factor> B0 = aie::load_v<vec_factor>(pB1);
    pB1 += vec_factor;
    //auto acc = aie::mul<acc16>(A0, B0);
    auto acc = aie::mul(A0, B0);
    aie::vector<T_out, vec_factor> cout = acc.template to_vector<T_out>();
    aie::store_v(pC1, cout);
    pC1 += vec_factor;
  }
  event1();
}
extern "C" {

void ewise_mul_i16_i16 (int16 *a, int16 *b, int16 *c){
  eltwise_vmul<int16, int16, DIM_M*DIM_N>(a, b, c);
}

void ewise_mul_bf16_bf16 (bfloat16 *a, bfloat16 *b, bfloat16 *c){
  eltwise_vmul<bfloat16, bfloat16, DIM_M*DIM_N>(a, b, c);
}

} // extern "C"
