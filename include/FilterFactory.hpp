#pragma once
#include "Filter.hpp"
#include "FilterParams.hpp"
#include <memory>

namespace cuda
{
/**
 * @brief Factory class for creating filters.
 */
class FilterFactory
{
  public:
    /**
     * @brief Create a filter of the specified type.
     *
     * @param type The type of filter to create.
     * @param width The width of the image.
     * @param height The height of the image.
     * @param params The parameters for the filter.
     * @param mode The memory mode for the filter (GLOBAL or SHARED_MEM).
     * @return A unique pointer to the created filter.
     */
    static std::unique_ptr<Filter> createFilter(FilterType type, const int width, const int height,
                                                const FilterParams& params,
                                                MemoryMode mode);
};

} // namespace cuda
