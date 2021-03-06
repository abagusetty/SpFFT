/*
 * Copyright (c) 2019 ETH Zurich, Simon Frasch
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef SPFFT_GPU_COMPLEX_CONVERISON_CUH
#define SPFFT_GPU_COMPLEX_CONVERISON_CUH

#include "gpu_util/gpu_fft_api.hpp"

namespace spfft {

template <typename T, typename U>
struct ConvertComplex {
  __device__ __host__ inline static T apply(const U& val) { return val; }
};

template <>
struct ConvertComplex<gpu::fft::ComplexFloatType, gpu::fft::ComplexDoubleType> {
  __device__ __host__ inline static gpu::fft::ComplexFloatType apply(
      const gpu::fft::ComplexDoubleType& val) {
    return gpu::fft::ComplexFloatType{(float)val.x, (float)val.y};
  }
};

template <>
struct ConvertComplex<gpu::fft::ComplexDoubleType, gpu::fft::ComplexFloatType> {
  __device__ __host__ inline static gpu::fft::ComplexDoubleType apply(
      const gpu::fft::ComplexFloatType& val) {
    return gpu::fft::ComplexDoubleType{(double)val.x, (double)val.y};
  }
};

}
#endif
