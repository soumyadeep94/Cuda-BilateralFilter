#include "NdVector.hpp"
#include "Png.hpp"
#include "FilterPipeline.hpp"
#include "FilterFactory.hpp"

#include <fmt/format.h>

#include <ranges>
#include <cstdio>
#include <iostream>

#include "cuda_utils.hpp"


/*Function to process the image data. It takes image as an input, performs a series of transformation and outputs 
the processed image or the relavant data*/
void processImage(const Image& inputImage, Image& resultImage,
                  const std::vector<cuda::FilterType>& filterTypes,
                  const std::vector<std::unique_ptr<cuda::FilterParams>>& filterParams,
                  cuda::MemoryMode mode)
{
    const auto& [width, height] = inputImage.getSize();

    // Create NdVector to wrap the input image data
    NdVector<float, 2> inputNdVector({width, height});
    std::ranges::copy(inputImage.getData(),
                      inputImage.getData() + inputNdVector.getNumberOfElements(),
                      inputNdVector.getData());

    // Allocate memory for the output image
    NdVector<float, 2> resultNdVector({width, height});

    cuda::FilterPipeline pipeline;
    for (size_t i = 0; i < filterTypes.size(); ++i) {
        pipeline.addFilter(cuda::FilterFactory::createFilter(filterTypes[i], width, height,
                                                             *filterParams[i], mode));
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    // compute with the applied filter
    pipeline.apply(inputNdVector.getData(), resultNdVector.getData(), width, height);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Filter execution time: " << milliseconds << " ms" << std::endl;

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaDeviceSynchronize();

    auto resultData = resultImage.getData();
    auto resultFloatData = resultNdVector.getData();

    // Convert float data back to uint8_t for the result image
    std::ranges::transform(
        std::views::iota(0u, width * height), resultData, [resultFloatData](auto i) {
            return static_cast<uint8_t>(std::clamp(resultFloatData[i], 0.0f, 255.0f));
        });
}

void app()
{
    const auto inputImage = png::load(std::filesystem::path{RESOURCE_DIR} / "clara.png");

    Image resultImage{inputImage.getSize()};

    //Define parameters for the bilateral filter
    constexpr float sigmaColor{25.0f};
    constexpr float sigmaSpace{25.0f};

    // Create filter parameters
    auto bilateralParams = std::make_unique<cuda::BilateralFilterParams>(sigmaColor, sigmaSpace);

    //Define filters to be added
    std::vector<cuda::FilterType> filterTypes{cuda::FilterType::BILATERAL_FILTER};
    std::vector<std::unique_ptr<cuda::FilterParams>> filterParams;
    filterParams.push_back(std::move(bilateralParams));

    //With shared memory mode
    {
        std::cout << "Bilateral filter using Shared memory" << std::endl;
        constexpr cuda::MemoryMode mode =
            cuda::MemoryMode::SHARED_MEM; // Toggle mode (GLOBAL or SHARED_MEM)

        // Process the image using bilateral filter
        processImage(inputImage, resultImage, filterTypes, filterParams, mode);

        // Save the result image
        png::save(resultImage, std::filesystem::path{RESOURCE_DIR} / "result.png");
    }
}

int main()
{
    try {
        app();
    } catch (const std::exception& ex) {
        fmt::print(stderr, "Error while processing: {}\n", ex.what());
    }
}