
#include <catch2/catch_all.hpp>

#include "BilateralFilter.hpp"
#include "NdVector.hpp"

using ImageVector = lv::NdVector<float, 2>;

constexpr int WIDTH = 1024;
constexpr int HEIGHT = 1024;
constexpr float SIGMA_COLOR = 25.0f;
constexpr float SIGMA_SPACE = 25.0f;

//Helper function to generate test data
void generateTestData(ImageVector& inputNdVector)
{
    for (int i = 0; i < WIDTH * HEIGHT; ++i) {
        inputNdVector.getData()[i] = static_cast<float>(i % 256);
    }
}

TEST_CASE("Benchmark Bilateral Filter", "[bilateral]")
{
    ImageVector inputNdVector({WIDTH, HEIGHT});
    ImageVector resultNdVector({WIDTH, HEIGHT});
    generateTestData(inputNdVector);

    cuda::BilateralFilter filter_naive(WIDTH, HEIGHT, SIGMA_COLOR, SIGMA_SPACE,
                                 cuda::MemoryMode::GLOBAL);
    cuda::BilateralFilter filter_optimized(WIDTH, HEIGHT, SIGMA_COLOR, SIGMA_SPACE,
                                     cuda::MemoryMode::SHARED_MEM);

    BENCHMARK("Bilateral Filter: Shared Memory")
    {
        filter_optimized.apply(inputNdVector.getData(), resultNdVector.getData());
        return resultNdVector.getData()[0];
    };

    BENCHMARK("Bilateral Filter: Standard")
    {
        filter_naive.apply(inputNdVector.getData(), resultNdVector.getData());
        return resultNdVector.getData()[0]; // return a value to ensure the compiler doesn't
                                            // optimize
    };
};