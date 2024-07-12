#pragma once

namespace cuda
{
/**
 * @brief Base struct for filter parameters.
 */
struct FilterParams
{
    /**
     * @brief Virtual destructor for the base filter parameters class.
     */
    virtual ~FilterParams() = default;
};

/**
 * @brief Parameters specific to the bilateral filter.
 */
struct BilateralFilterParams : public FilterParams
{
    /**
     * @brief Constructor for BilateralFilterParams.
     *
     * @param sigmaColor The standard deviation for the color space.
     * @param sigmaSpace The standard deviation for the coordinate space.
     */
    BilateralFilterParams(const float sigmaColor, const float sigmaSpace)
        : sigmaColor(sigmaColor), sigmaSpace(sigmaSpace)
    {
    }

    const float sigmaColor{};
    const float sigmaSpace{};
};
} // namespace cuda
