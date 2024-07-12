#include "FilterPipeline.hpp"
#include "Filter.hpp"
#include <ranges>

namespace cuda
{
void FilterPipeline::addFilter(std::unique_ptr<Filter> filter)
{
    m_filters.push_back(std::move(filter));
}

void FilterPipeline::apply(const float* input, float* output, const int width, const int height) const
{
    if (m_filters.empty()) {
        return;
    }

    //create an intermediate buffer
    std::vector<float> intermediateBuffer(static_cast<size_t>(width) * static_cast<size_t>(height));
    float* currentInput = const_cast<float*>(input);
    float* currentOutput = intermediateBuffer.data();

    for (auto&& filter : m_filters | std::views::all) {

        if (filter == m_filters.back())
            currentOutput = output;

        filter->apply(currentInput, currentOutput);

        if (filter != m_filters.back())
            std::swap(currentInput, currentOutput);
    }
}

}