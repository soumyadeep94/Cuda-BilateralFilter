#pragma once

namespace cuda
{
    /* @enum MemoryMode
     @brief Specifies the memory mode of the filter
     */
    enum class MemoryMode
    {
        GLOBAL,       ///< Use global memory.
        SHARED_MEM   ///< Use shared memory.
    };

    /* @enum FilterType
    * @brief Specifies the filters to be added
    */
    enum class FilterType
    {
        BILATERAL_FILTER,
        // Add other filter types
    };

    class FilterParams;
    /**
     * @brief Abstract base class for filters.
     */
    class Filter
    {
      public:
        virtual ~Filter() = default;

        /**
         * @brief Apply the filter to the input data.
         *
         * @param input Pointer to the input data.
         * @param output Pointer to the output data.
         */
        virtual void apply(const float* input, float* output) const = 0;

         /**
         * @brief Set the parameters for the bilateral filter.
         *
         * @param params The filter parameters to be set.
         */
        virtual void setParams(const FilterParams& params) = 0;
    };
} // namespace cuda