#include "BilateralFilter.hpp"
#include <device_launch_parameters.h>

#include <cmath>
#include <iostream>


namespace
{
constexpr int BLOCK_SIZE = 32;
constexpr int RADIUS = 2;  // Example radius
constexpr int PADDING = 1; // Padding

/**
 * @brief Device function to get the pixel value
 *
 * @tparam  useSharedMemory Flag to indicate whether to use shared memory.
 * @param input Pointer to the input image data.
 * @param sharedInput Pointer to the shared memory data
 * @param globalIdx Index for global memory access.
 * @param sharedIdx Index for shared memory access.
 * @return The pixel value.
 */
template <bool useSharedMemory>
__device__ float getPixel(const float* input, float* sharedInput, const int globalIdx,
                          const int sharedIdx)
{
    if constexpr (useSharedMemory) {
        return sharedInput[sharedIdx];
    }
    return input[globalIdx];
}

/**
 * @brief Kernel function for bilateral filtering.
 *
 * @tparam useSharedMemory Flag to indicate whether to use shared memory.
 * @param input Pointer to the input image data.
 * @param output Pointer to the output image data.
 * @param width The width of the image.
 * @param height The height of the image.
 * @param sigmaColor The standard deviation for the color space.
 * @param sigmaSpace The standard deviation for the coordinate space.
 */
template <bool useSharedMemory>
__global__ void bilateralFilterKernel(const float* input, float* output, const int width, const int height,
                                      const float sigmaColor, const float sigmaSpace)
{
    // Calculate pixel coordinates
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Bounds check
    if (x >= width || y >= height)
        return;

    extern __shared__ float sharedMemory[];
    float* sharedInput = nullptr;

    if constexpr (useSharedMemory) {
        sharedInput = sharedMemory;

        // calculate local and shared memory indices
        const int localX = threadIdx.x;
        const int localY = threadIdx.y;
        const int sharedWidth = blockDim.x + 2 * RADIUS + PADDING;

        const int globalIdx = y * width + x; 
        const int sharedIdx = (localY + RADIUS) * sharedWidth + (localX + RADIUS);

        // load the main data block into shared memory
        sharedInput[sharedIdx] = input[globalIdx];

        // load halo cells
        if (localX < RADIUS) {
            // left halo
            int leftHaloIdx = sharedIdx - RADIUS;
            int leftGlobalIdx = globalIdx - RADIUS;
            sharedInput[leftHaloIdx] = x >= RADIUS ? input[leftGlobalIdx] : input[globalIdx];
        }

        if (localX >= blockDim.x - RADIUS) {
            // right halo
            int rightHaloIdx = sharedIdx + RADIUS;
            int rightGlobalIdx = globalIdx + RADIUS;
            sharedInput[rightHaloIdx] =
                x + RADIUS < width ? input[rightGlobalIdx] : input[globalIdx];
        }

        if (localY < RADIUS) {
            // Top halo
            int topHaloIdx = sharedIdx - RADIUS * sharedWidth;
            int topGlobalIdx = globalIdx - RADIUS * width;
            sharedInput[topHaloIdx] = y >= RADIUS ? input[topGlobalIdx] : input[globalIdx];
        }

        if (localY >= blockDim.y - RADIUS) {
            // bottom halo
            int bottomHaloIdx = sharedIdx + RADIUS * sharedWidth;
            int bottomGlobalIdx = globalIdx + RADIUS * width;
            sharedInput[bottomHaloIdx] =
                y + RADIUS < height ? input[bottomGlobalIdx] : input[globalIdx];
        }

        __syncthreads();
    }

    const int globalIdx = y * width + x;
    const int sharedIdx =
        (threadIdx.y + RADIUS) * (blockDim.x + 2 * RADIUS + PADDING) + threadIdx.x + RADIUS;
    float centerPixel = getPixel<useSharedMemory>(input, sharedInput, globalIdx, sharedIdx);
    float sum{0.0f};
    float normalization{0.0f};

    const float sigmaColorSquared = sigmaColor * sigmaColor;
    const float sigmaSpaceSquared = sigmaSpace * sigmaSpace;

    for (int dy = -RADIUS; dy <= RADIUS; ++dy) {
        for (int dx = -RADIUS; dx <= RADIUS; ++dx) {
            int ix = min(max(x + dx, 0), width - 1);
            int iy = min(max(y + dy, 0), height - 1);

            int globalIx = iy * width + ix;
            int sharedIx = (threadIdx.y + dy + RADIUS) * (blockDim.x + 2 * RADIUS + PADDING) +
                           (threadIdx.x + dx + RADIUS);

            float neighborPixel = getPixel<useSharedMemory>(input, sharedInput, globalIx, sharedIx);
            float colorDistance = (neighborPixel - centerPixel) * (neighborPixel - centerPixel);
            float spaceDistance = (dx * dx + dy * dy);
            float weight = expf(-(colorDistance / (2 * sigmaColorSquared)) -
                                (spaceDistance / (2 * sigmaSpaceSquared)));

            sum += weight * neighborPixel;
            normalization += weight;
        }
    }

    output[y * width + x] = sum / normalization;
}

} // anonymous namespace


namespace cuda
{

BilateralFilter::BilateralFilter(const int width, const int height, const float sigmaColor,
                                 const float sigmaSpace, MemoryMode mode)
    : m_width(width), m_height(height), m_sigmaColor(sigmaColor), m_sigmaSpace(sigmaSpace),
      m_Mode(mode)
{
    // Allocate Memory

    CUDA_CHECK(cudaMalloc(&d_input_, m_width * m_height * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output_, m_width * m_height * sizeof(float)));
}

BilateralFilter::~BilateralFilter() noexcept
{
    // Free memory
    CUDA_CHECK_NO_THROW(cudaFree(d_input_));
    CUDA_CHECK_NO_THROW(cudaFree(d_output_));
}

void BilateralFilter::apply(const float* input, float* output) const
{
    // copy input to GPU memory
    cudaMemcpy(d_input_, input, m_width * m_height * sizeof(float), cudaMemcpyHostToDevice);

    // launch the filter kernel
    launchKernel(d_input_, d_output_);

    // Copy output from GPU memory
    //std::copy(d_output_, d_output_ + m_width * m_height, output);
    cudaMemcpy(output, d_output_, m_width * m_height * sizeof(float), cudaMemcpyDeviceToHost);
}

void BilateralFilter::setParams(const FilterParams& params)
{
    const auto& bilateralParams = dynamic_cast<const BilateralFilterParams&>(params);
    m_sigmaColor = bilateralParams.sigmaColor;
    m_sigmaSpace = bilateralParams.sigmaSpace;
}

void BilateralFilter::launchKernel(const float* input, float* output) const
{
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((m_width + blockSize.x - 1) / blockSize.x,
                  (m_height + blockSize.y - 1) / blockSize.y);

    if (m_Mode == cuda::MemoryMode::SHARED_MEM) {
        size_t sharedMemSize = (BLOCK_SIZE + 2 * RADIUS + PADDING) *
                               (BLOCK_SIZE + 2 * RADIUS + PADDING) * sizeof(float);
        // Launch the kernel with shared memory
        bilateralFilterKernel<true><<<gridSize, blockSize, sharedMemSize>>>(
            const_cast<float*>(input), output, m_width, m_height, m_sigmaColor, m_sigmaSpace);
    } else {
        // launch the naive kernel, which use global memory only
        bilateralFilterKernel<false><<<gridSize, blockSize>>>(
            const_cast<float*>(input), output, m_width, m_height, m_sigmaColor, m_sigmaSpace);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

}

} // namespace cuda