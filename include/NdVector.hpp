#pragma once

#include <fmt/format.h>
#include <gsl/gsl-lite.hpp>

#include <array>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace lv
{

/**
 * @brief Simple/minimal n-dimensional vector.
 *
 * @tparam ElemType Data type for an element in this @b NdVector
 * @Dim Number of dimensions.
 */
template <typename ElemType, size_t Dim> class NdVector
{
    static_assert(Dim > 0, "NdVector must at least be one-dimensional");

  public:
    using ElementType = ElemType;

    /**
     * @brief Create a zero-initialized object with the dimensions given in @a size.
     *
     * @param size Size for each of the dimension.
     * @throws std::invalid_argument if @a size contains a 0 in any dimension.
     * @throws std::bad_alloc if resulting buffer is too larget to fit in available memory.
     */
    explicit NdVector(std::array<std::uint32_t, Dim> size)
        : size_{size}, buffer_(calculateNumberOfElements(size))
    {
    }

    NdVector(const NdVector&) = default;
    NdVector& operator=(const NdVector&) = default;
    NdVector(NdVector&&) noexcept = default;
    NdVector& operator=(NdVector&&) noexcept = default;

    /**
     * @brief Get the overall number of elements in this object.
     *
     * @return Overall number of elements.
     */
    [[nodiscard]] size_t getNumberOfElements() const noexcept
    {
        return calculateNumberOfElements(getSize());
    }

    /**
     * @brief Get the size of this object.
     *
     * @return Sizes for each dimension.
     */
    [[nodiscard]] std::array<std::uint32_t, Dim> getSize() const noexcept { return size_; }

    /**
     * @brief Gets access to an element in this object.
     *
     * @pre @index must be inside this object's bounds! Undefined behavior otherwise.
     *
     * @param index Index location to access in this object.
     * @return Reference to the element in this object.
     */
    [[nodiscard]] ElementType& operator()(std::array<std::uint32_t, Dim> index) noexcept
    {
        return buffer_[calculateLinearIndex(index)];
    }

    /**
     * @brief Gets access to an element in this object.
     *
     * @pre @index must be inside this object's bounds! Undefined behavior otherwise.
     *
     * @param index Index location to access in this object.
     * @return Reference to the element in this object.
     */
    [[nodiscard]] const ElementType& operator()(std::array<std::uint32_t, Dim> index) const noexcept
    {
        return buffer_[calculateLinearIndex(index)];
    }

    /**
     * @brief Get access to the underlying buffer.
     *
     * @return Pointer to the first element in this object.
     */
    [[nodiscard]] const ElementType* getData() const noexcept { return buffer_.data(); }

    /**
     * @brief Get access to the underlying buffer.
     *
     * @return Pointer to the first element in this object.
     */
    [[nodiscard]] ElementType* getData() noexcept { return buffer_.data(); }

  private:
    [[nodiscard]] static size_t calculateNumberOfElements(
        std::array<std::uint32_t, Dim> size) noexcept
    {
        static const constexpr size_t START_SIZE = 1;
        return std::accumulate(size.begin(), size.end(), START_SIZE,
                               [](size_t accumulator, size_t value) {
                                   if (value == 0) {
                                       throw std::invalid_argument{"No dimension of size may be 0"};
                                   }
                                   return accumulator * value;
                               });
    }

    [[nodiscard]] ptrdiff_t calculateLinearIndex(std::array<uint32_t, Dim> index) const noexcept
    {
        std::ptrdiff_t linear_index = 0;
        for (std::ptrdiff_t dimension = Dim - 1; dimension >= 0; dimension--) {
            linear_index *= gsl::narrow_cast<std::ptrdiff_t>(size_[dimension]);
            linear_index += gsl::narrow_cast<std::ptrdiff_t>(index[dimension]);
        }
        return linear_index;
    }

    std::array<std::uint32_t, Dim> size_;
    std::vector<ElementType> buffer_;
};

using Image = NdVector<std::uint8_t, 2>;

} // namespace lv