#pragma once

#include <vector>
#include <memory>

namespace cuda
{
/**
 * @brief Class to manage a pipeline of filters.
 */

class Filter;

class FilterPipeline
{
  public:
    /**
     * @brief Add a filter to the pipeline.
     *
     * @param filter Unique pointer to the filter to be added.
     */
    void addFilter(std::unique_ptr<Filter> filter);

     /**
     * @brief Apply the pipeline of filters to the input data.
     *
     * @param input Pointer to the input data.
     * @param output Pointer to the output data.
     * @param width Width of the input and output data.
     * @param height Height of the input and output data.
     */
    void apply(const float* input, float* output, const int width, const int height) const;

    private:
        std::vector<std::unique_ptr<Filter>> m_filters; ///< Container for the filters in the pipeline.
};
} //namespace cuda