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
#include <CL/sycl.hpp>
#include <algorithm>
#include <cassert>
#include "gpu_util/complex_conversion.dp.hpp"
#include "gpu_util/gpu_fft_api.hpp"
#include "gpu_util/gpu_kernel_parameter.hpp"
#include "gpu_util/gpu_runtime.hpp"
#include "memory/array_view_utility.hpp"
#include "memory/gpu_array_const_view.hpp"
#include "memory/gpu_array_view.hpp"

namespace spfft {

// Packs z-sticks into buffer for MPI_Alltoall
// Dimension of buffer are (numRanks, maxNumZSticks, maxNumXYPlanes)
// Dimension of freqZData are (numLocalZSticks, dimZ)
template <typename DATA_TYPE, typename BUFFER_TYPE>
static void buffered_pack_backward_kernel(const GPUArrayConstView1D<int> numXYPlanes,
                                                     const GPUArrayConstView1D<int> xyPlaneOffsets,
                                                     const GPUArrayConstView2D<DATA_TYPE> freqZData,
                                                     GPUArrayView3D<BUFFER_TYPE> buffer,
                                                     cl::sycl::nd_item<3>& item_ct) {
    const int xyPlaneIndex = item_ct.get_global_id(2);
    for (int r = 0; r < numXYPlanes.size(); ++r) {
	if (xyPlaneIndex < numXYPlanes(r)) {
	    const int xyOffset = xyPlaneOffsets(r);
	    for (int zStickIndex = item_ct.get_group(1); zStickIndex < freqZData.dim_outer();
		 zStickIndex += item_ct.get_group_range(1)) {
		buffer(r, zStickIndex, xyPlaneIndex) = ConvertComplex<BUFFER_TYPE, DATA_TYPE>::apply(
		    freqZData(zStickIndex, xyPlaneIndex + xyOffset));
	    }
	}
    }
}

template <typename DATA_TYPE, typename BUFFER_TYPE>
static auto buffered_pack_backward_launch(const gpu::StreamType stream, const int maxNumXYPlanes,
                                          const GPUArrayView1D<int>& numXYPlanes,
                                          const GPUArrayView1D<int>& xyPlaneOffsets,
                                          const GPUArrayView2D<DATA_TYPE>& freqZData,
                                          GPUArrayView3D<BUFFER_TYPE> buffer) -> void {
  assert(xyPlaneOffsets.size() == numXYPlanes.size());
  assert(buffer.size() >= freqZData.size());
  assert(buffer.dim_outer() == xyPlaneOffsets.size());
  assert(buffer.dim_inner() == maxNumXYPlanes);
  const cl::sycl::range<3> threadBlock(1, 1, gpu::BlockSizeSmall);
  const cl::sycl::range<3> threadGrid(1, std::min(freqZData.dim_outer(), gpu::GridSizeMedium),
                                  (maxNumXYPlanes + threadBlock[2] - 1) / threadBlock[2]);
  assert(threadGrid.x > 0);
  assert(threadGrid.y > 0);
  launch_kernel(buffered_pack_backward_kernel<DATA_TYPE, BUFFER_TYPE>, threadGrid, threadBlock, 0,
                stream, numXYPlanes, xyPlaneOffsets, freqZData, buffer);
}

auto buffered_pack_backward(
    const gpu::StreamType stream, const int maxNumXYPlanes, const GPUArrayView1D<int>& numXYPlanes,
    const GPUArrayView1D<int>& xyPlaneOffsets,
    const GPUArrayView2D<typename gpu::fft::ComplexType<double>::type>& freqZData,
    GPUArrayView3D<typename gpu::fft::ComplexType<double>::type> buffer) -> void {
  buffered_pack_backward_launch(stream, maxNumXYPlanes, numXYPlanes, xyPlaneOffsets, freqZData,
                                buffer);
}

auto buffered_pack_backward(
    const gpu::StreamType stream, const int maxNumXYPlanes, const GPUArrayView1D<int>& numXYPlanes,
    const GPUArrayView1D<int>& xyPlaneOffsets,
    const GPUArrayView2D<typename gpu::fft::ComplexType<float>::type>& freqZData,
    GPUArrayView3D<typename gpu::fft::ComplexType<float>::type> buffer) -> void {
  buffered_pack_backward_launch(stream, maxNumXYPlanes, numXYPlanes, xyPlaneOffsets, freqZData,
                                buffer);
}

auto buffered_pack_backward(
    const gpu::StreamType stream, const int maxNumXYPlanes, const GPUArrayView1D<int>& numXYPlanes,
    const GPUArrayView1D<int>& xyPlaneOffsets,
    const GPUArrayView2D<typename gpu::fft::ComplexType<double>::type>& freqZData,
    GPUArrayView3D<typename gpu::fft::ComplexType<float>::type> buffer) -> void {
  buffered_pack_backward_launch(stream, maxNumXYPlanes, numXYPlanes, xyPlaneOffsets, freqZData,
                                buffer);
}

// Unpacks z-sticks from buffer after MPI_Alltoall
// Dimension of buffer are (numRanks, maxNumZSticks, maxNumXYPlanes)
// Dimension of freqXYData are (numLocalXYPlanes, dimY, dimX)
template <typename DATA_TYPE, typename BUFFER_TYPE>
static void buffered_unpack_backward_kernel(
    const GPUArrayConstView1D<int> numZSticks, const GPUArrayConstView1D<int> indices,
    const GPUArrayConstView3D<BUFFER_TYPE> buffer, GPUArrayView2D<DATA_TYPE> freqXYDataFlat,
    cl::sycl::nd_item<3>& item_ct) {
    // buffer.dim_mid() is equal to maxNumZSticks
    const int xyPlaneIndex = item_ct.get_global_id(2);
    if (xyPlaneIndex < freqXYDataFlat.dim_outer()) {
	for (int r = 0; r < numZSticks.size(); ++r) {
	    const int numCurrentZSticks = numZSticks(r);
	    for (int zStickIndex = item_ct.get_group(1); zStickIndex < numCurrentZSticks;
		 zStickIndex += item_ct.get_group_range(1)) {
		const int currentIndex = indices(r * buffer.dim_mid() + zStickIndex);
		freqXYDataFlat(xyPlaneIndex, currentIndex) =
		    ConvertComplex<DATA_TYPE, BUFFER_TYPE>::apply(buffer(r, zStickIndex, xyPlaneIndex));
	    }
	}
    }
}

template <typename DATA_TYPE, typename BUFFER_TYPE>
static auto buffered_unpack_backward_launch(const gpu::StreamType stream, const int maxNumXYPlanes,
                                            const GPUArrayView1D<int>& numZSticks,
                                            const GPUArrayView1D<int>& indices,
                                            const GPUArrayView3D<BUFFER_TYPE>& buffer,
                                            GPUArrayView3D<DATA_TYPE> freqXYData) -> void {
  assert(buffer.dim_outer() == numZSticks.size());
  assert(buffer.dim_inner() == maxNumXYPlanes);
  assert(indices.size() == buffer.dim_mid() * numZSticks.size());
  const cl::sycl::range<3> threadBlock(1, 1, gpu::BlockSizeSmall);
  const cl::sycl::range<3> threadGrid(1, std::min(buffer.dim_mid(), gpu::GridSizeMedium),
                                  (freqXYData.dim_outer() + threadBlock[2] - 1) / threadBlock[2]);
  assert(threadGrid.x > 0);
  assert(threadGrid.y > 0);
  launch_kernel(buffered_unpack_backward_kernel<DATA_TYPE, BUFFER_TYPE>, threadGrid, threadBlock, 0,
                stream, numZSticks, indices, buffer,
                GPUArrayView2D<DATA_TYPE>(freqXYData.data(), freqXYData.dim_outer(),
                                          freqXYData.dim_mid() * freqXYData.dim_inner(),
                                          freqXYData.device_id()));
}

auto buffered_unpack_backward(
    const gpu::StreamType stream, const int maxNumXYPlanes, const GPUArrayView1D<int>& numZSticks,
    const GPUArrayView1D<int>& indices,
    const GPUArrayView3D<typename gpu::fft::ComplexType<double>::type>& buffer,
    GPUArrayView3D<typename gpu::fft::ComplexType<double>::type> freqXYData) -> void {
  buffered_unpack_backward_launch(stream, maxNumXYPlanes, numZSticks, indices, buffer, freqXYData);
}

auto buffered_unpack_backward(
    const gpu::StreamType stream, const int maxNumXYPlanes, const GPUArrayView1D<int>& numZSticks,
    const GPUArrayView1D<int>& indices,
    const GPUArrayView3D<typename gpu::fft::ComplexType<float>::type>& buffer,
    GPUArrayView3D<typename gpu::fft::ComplexType<float>::type> freqXYData) -> void {
  buffered_unpack_backward_launch(stream, maxNumXYPlanes, numZSticks, indices, buffer, freqXYData);
}

auto buffered_unpack_backward(
    const gpu::StreamType stream, const int maxNumXYPlanes, const GPUArrayView1D<int>& numZSticks,
    const GPUArrayView1D<int>& indices,
    const GPUArrayView3D<typename gpu::fft::ComplexType<float>::type>& buffer,
    GPUArrayView3D<typename gpu::fft::ComplexType<double>::type> freqXYData) -> void {
  buffered_unpack_backward_launch(stream, maxNumXYPlanes, numZSticks, indices, buffer, freqXYData);
}

// Unpacks z-sticks from buffer after MPI_Alltoall
// Dimension of buffer are (numRanks, maxNumZSticks, maxNumXYPlanes)
// Dimension of freqZData are (numLocalZSticks, dimZ)
template <typename DATA_TYPE, typename BUFFER_TYPE>
static void buffered_unpack_forward_kernel(const GPUArrayConstView1D<int> numXYPlanes,
                                                      const GPUArrayConstView1D<int> xyPlaneOffsets,
                                                      const GPUArrayConstView3D<BUFFER_TYPE> buffer,
                                                      GPUArrayView2D<DATA_TYPE> freqZData,
                                                      cl::sycl::nd_item<3>& item_ct) {
    const int xyPlaneIndex = item_ct.get_global_id(2);
    for (int r = 0; r < numXYPlanes.size(); ++r) {
	if (xyPlaneIndex < numXYPlanes(r)) {
	    const int xyOffset = xyPlaneOffsets(r);
	    for (int zStickIndex = item_ct.get_group(1); zStickIndex < freqZData.dim_outer();
		 zStickIndex += item_ct.get_group_range(1)) {
		freqZData(zStickIndex, xyPlaneIndex + xyOffset) =
		    ConvertComplex<DATA_TYPE, BUFFER_TYPE>::apply(buffer(r, zStickIndex, xyPlaneIndex));
	    }
	}
    }
}

template <typename DATA_TYPE, typename BUFFER_TYPE>
static auto buffered_unpack_forward_launch(const gpu::StreamType stream, const int maxNumXYPlanes,
                                           const GPUArrayView1D<int>& numXYPlanes,
                                           const GPUArrayView1D<int>& xyPlaneOffsets,
                                           const GPUArrayView3D<BUFFER_TYPE>& buffer,
                                           GPUArrayView2D<DATA_TYPE> freqZData) -> void {
  assert(xyPlaneOffsets.size() == numXYPlanes.size());
  assert(buffer.size() >= freqZData.size());
  assert(buffer.dim_outer() == xyPlaneOffsets.size());
  assert(buffer.dim_inner() == maxNumXYPlanes);
  const cl::sycl::range<3> threadBlock(1, 1, gpu::BlockSizeSmall);
  const cl::sycl::range<3> threadGrid(1, std::min(freqZData.dim_outer(), gpu::GridSizeMedium),
                                  (maxNumXYPlanes + threadBlock[2] - 1) / threadBlock[2]);
  assert(threadGrid.x > 0);
  assert(threadGrid.y > 0);
  launch_kernel(buffered_unpack_forward_kernel<DATA_TYPE, BUFFER_TYPE>, threadGrid, threadBlock, 0,
                stream, numXYPlanes, xyPlaneOffsets, buffer, freqZData);
}

auto buffered_unpack_forward(
    const gpu::StreamType stream, const int maxNumXYPlanes, const GPUArrayView1D<int>& numXYPlanes,
    const GPUArrayView1D<int>& xyPlaneOffsets,
    const GPUArrayView3D<typename gpu::fft::ComplexType<double>::type>& buffer,
    GPUArrayView2D<typename gpu::fft::ComplexType<double>::type> freqZData) -> void {
  buffered_unpack_forward_launch(stream, maxNumXYPlanes, numXYPlanes, xyPlaneOffsets, buffer,
                                 freqZData);
}

auto buffered_unpack_forward(
    const gpu::StreamType stream, const int maxNumXYPlanes, const GPUArrayView1D<int>& numXYPlanes,
    const GPUArrayView1D<int>& xyPlaneOffsets,
    const GPUArrayView3D<typename gpu::fft::ComplexType<float>::type>& buffer,
    GPUArrayView2D<typename gpu::fft::ComplexType<float>::type> freqZData) -> void {
  buffered_unpack_forward_launch(stream, maxNumXYPlanes, numXYPlanes, xyPlaneOffsets, buffer,
                                 freqZData);
}

auto buffered_unpack_forward(
    const gpu::StreamType stream, const int maxNumXYPlanes, const GPUArrayView1D<int>& numXYPlanes,
    const GPUArrayView1D<int>& xyPlaneOffsets,
    const GPUArrayView3D<typename gpu::fft::ComplexType<float>::type>& buffer,
    GPUArrayView2D<typename gpu::fft::ComplexType<double>::type> freqZData) -> void {
  buffered_unpack_forward_launch(stream, maxNumXYPlanes, numXYPlanes, xyPlaneOffsets, buffer,
                                 freqZData);
}

// Packs z-sticks into buffer for MPI_Alltoall
// Dimension of buffer are (numRanks, maxNumZSticks, maxNumXYPlanes)
// Dimension of freqXYData are (numLocalXYPlanes, dimY, dimX)
template <typename DATA_TYPE, typename BUFFER_TYPE>
static void buffered_pack_forward_kernel(
    const GPUArrayConstView1D<int> numZSticks, const GPUArrayConstView1D<int> indices,
    const GPUArrayConstView2D<DATA_TYPE> freqXYDataFlat, GPUArrayView3D<BUFFER_TYPE> buffer,
    cl::sycl::nd_item<3>& item_ct) {
    // buffer.dim_mid() is equal to maxNumZSticks
    const int xyPlaneIndex = item_ct.get_global_id(2);
    if (xyPlaneIndex < freqXYDataFlat.dim_outer()) {
	for (int r = 0; r < numZSticks.size(); ++r) {
	    const int numCurrentZSticks = numZSticks(r);
	    for (int zStickIndex = item_ct.get_group(1); zStickIndex < numCurrentZSticks;
		 zStickIndex += item_ct.get_group_range(1)) {
		const int currentIndex = indices(r * buffer.dim_mid() + zStickIndex);
		buffer(r, zStickIndex, xyPlaneIndex) = ConvertComplex<BUFFER_TYPE, DATA_TYPE>::apply(
		    freqXYDataFlat(xyPlaneIndex, currentIndex));
	    }
	}
    }
}

template <typename DATA_TYPE, typename BUFFER_TYPE>
static auto buffered_pack_forward_launch(const gpu::StreamType stream, const int maxNumXYPlanes,
                                         const GPUArrayView1D<int>& numZSticks,
                                         const GPUArrayView1D<int>& indices,
                                         const GPUArrayView3D<DATA_TYPE>& freqXYData,
                                         GPUArrayView3D<BUFFER_TYPE> buffer) -> void {
  assert(buffer.dim_outer() == numZSticks.size());
  assert(buffer.dim_inner() == maxNumXYPlanes);
  assert(indices.size() == buffer.dim_mid() * numZSticks.size());
  const cl::sycl::range<3> threadBlock(1, 1, gpu::BlockSizeSmall);
  const cl::sycl::range<3> threadGrid(1, std::min(buffer.dim_mid(), gpu::GridSizeMedium),
                                  (freqXYData.dim_outer() + threadBlock[2] - 1) / threadBlock[2]);
  assert(threadGrid.x > 0);
  assert(threadGrid.y > 0);
  launch_kernel(buffered_pack_forward_kernel<DATA_TYPE, BUFFER_TYPE>, threadGrid, threadBlock, 0,
                stream, numZSticks, indices,
                GPUArrayConstView2D<DATA_TYPE>(freqXYData.data(), freqXYData.dim_outer(),
                                               freqXYData.dim_mid() * freqXYData.dim_inner(),
                                               freqXYData.device_id()),
                buffer);
}

auto buffered_pack_forward(
    const gpu::StreamType stream, const int maxNumXYPlanes, const GPUArrayView1D<int>& numZSticks,
    const GPUArrayView1D<int>& indices,
    const GPUArrayView3D<typename gpu::fft::ComplexType<double>::type>& freqXYData,
    GPUArrayView3D<typename gpu::fft::ComplexType<double>::type> buffer) -> void {
  buffered_pack_forward_launch(stream, maxNumXYPlanes, numZSticks, indices, freqXYData, buffer);
}

auto buffered_pack_forward(
    const gpu::StreamType stream, const int maxNumXYPlanes, const GPUArrayView1D<int>& numZSticks,
    const GPUArrayView1D<int>& indices,
    const GPUArrayView3D<typename gpu::fft::ComplexType<float>::type>& freqXYData,
    GPUArrayView3D<typename gpu::fft::ComplexType<float>::type> buffer) -> void {
  buffered_pack_forward_launch(stream, maxNumXYPlanes, numZSticks, indices, freqXYData, buffer);
}

auto buffered_pack_forward(
    const gpu::StreamType stream, const int maxNumXYPlanes, const GPUArrayView1D<int>& numZSticks,
    const GPUArrayView1D<int>& indices,
    const GPUArrayView3D<typename gpu::fft::ComplexType<double>::type>& freqXYData,
    GPUArrayView3D<typename gpu::fft::ComplexType<float>::type> buffer) -> void {
  buffered_pack_forward_launch(stream, maxNumXYPlanes, numZSticks, indices, freqXYData, buffer);
}

}  // namespace spfft
