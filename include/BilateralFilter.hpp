#pragma once

#include "cuda_utils.hpp"

#include "Filter.hpp"
#include "FilterParams.hpp"


/**
* @class BilateralFilter
* @brief A class for applying bilateral filter on an image
*/

namespace cuda
{
class BilateralFilter : public Filter
{
  public:
    /**
     * @brief Constructor for BilateralFilter.
     *
     * Initializes the BilateralFilter with the specified image dimensions,
     * standard deviations for color and space, and the filtering mode.
     *
     * @param width The width of the image.
     * @param height The height of the image.
     * @param sigmaColor The standard deviation for the color space.
     * @param sigmaSpace The standard deviation for the coordinate space.
     * @param mode The memory mode of the filter (GLOBAL or SHARED_MEM).
     */
    BilateralFilter(const int width, const int height, float sigmaColor,
                    float sigmaSpace, MemoryMode mode = MemoryMode::GLOBAL);
    /**
     * @brief Destructor for BilateralFilter.
     *
     * Cleans up allocated memory.
     */
    ~BilateralFilter() noexcept override;

    /**
     * @brief Applies the bilateral filter to the input image.
     *
     * This method applies the bilateral filter to the input image data and
     * stores the filtered result in the output image data.
     *
     * @param input The input image data.
     * @param output The output image data.
     */
    void apply(const float* input, float* output) const override;

    void setParams(const FilterParams& params) override;

  private:
    /**
     * @brief Launches the appropriate kernel for the bilateral filter.
     *
     * This method launches either the naive or optimized bilateral filter kernel
     * based on the specified mode.
     *
     * @param input The input image data.
     * @param output The output image data.
     */
    void launchKernel(const float* input, float* output) const;

  private:
    const int m_width{};
    const int m_height{};
    float m_sigmaColor{};
    float m_sigmaSpace{};
    float* d_input_;
    float* d_output_;
    MemoryMode m_Mode{};
};
} // namespace cuda
