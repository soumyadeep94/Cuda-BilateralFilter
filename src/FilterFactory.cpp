#include "FilterFactory.hpp"
#include "BilateralFilter.hpp"

namespace cuda
{
    std::unique_ptr<Filter> FilterFactory::createFilter(FilterType type, const int width,
                                                        const int height, const FilterParams& params, MemoryMode mode)
    {
        switch (type) {
        case FilterType::BILATERAL_FILTER: {
            const auto& bilateralParams = dynamic_cast<const BilateralFilterParams&>(params);
            return std::make_unique<BilateralFilter>(width, height, bilateralParams.sigmaColor, bilateralParams.sigmaSpace,
                                                     mode);
        }
        default:
            return nullptr;
        }
    }
}