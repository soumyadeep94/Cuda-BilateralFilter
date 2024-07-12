/*
 * Copyright 2023 LUMA Vision Limited
 * Company Number 606427, Ireland
 * All rights reserved
 */

#include "Png.hpp"

#include <fmt/std.h>
#include <gsl/gsl-lite.hpp>
#include <png.h>

namespace lv::png
{

Image load(const std::filesystem::path& path)
{
    auto* fileHandle = fopen(path.string().c_str(), "rb");
    if (fileHandle == nullptr) {
        throw std::runtime_error{fmt::format("Error opening {} for reading", path)};
    }
    const auto closeFile = gsl::finally([fileHandle]() { fclose(fileHandle); });

    auto* png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (png == nullptr) {
        throw std::runtime_error{"Error creating PNG read struct"};
    }
    auto* pngInfo = png_create_info_struct(png);
    const auto destroyPng =
        gsl::finally([&png, &pngInfo]() { png_destroy_read_struct(&png, &pngInfo, nullptr); });
    if (pngInfo == nullptr) {
        throw std::runtime_error{"Error creating PNG info struct"};
    }

    if (setjmp(png_jmpbuf(png))) { // NOLINT
        throw std::runtime_error("Error setting up PNG error handling");
    }

    png_init_io(png, fileHandle);
    png_read_info(png, pngInfo);

    const auto width = png_get_image_width(png, pngInfo);
    const auto height = png_get_image_height(png, pngInfo);

    if (png_get_color_type(png, pngInfo) != PNG_COLOR_TYPE_GRAY) {
        throw std::runtime_error{"Unsupported color type"};
    }
    if (png_get_bit_depth(png, pngInfo) != sizeof(Image::ElementType) * 8) {
        throw std::runtime_error{"Unsupported bit depth"};
    }
    if (png_get_rowbytes(png, pngInfo) != sizeof(Image::ElementType) * width) {
        throw std::runtime_error{"Unsupported row stride"};
    }

    Image image{{width, height}};
    std::vector<png_bytep> rows(height);
    for (std::uint32_t row = 0; row < height; row++) {
        rows[row] = &image({0, row});
    }

    png_read_image(png, rows.data());

    return image;
}


void save(const Image& image, const std::filesystem::path& path)
{
    const auto [width, height] = image.getSize();

    auto* fileHandle = fopen(path.string().c_str(), "wb");
    if (fileHandle == nullptr) {
        throw std::runtime_error{fmt::format("Error opening {} for writing", path)};
    }
    const auto closeFile = gsl::finally([fileHandle]() { fclose(fileHandle); });

    auto* png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (png == nullptr) {
        throw std::runtime_error{"Error creating PNG write struct"};
    }
    auto* pngInfo = png_create_info_struct(png);
    const auto destroyPng =
        gsl::finally([&png, &pngInfo]() { png_destroy_write_struct(&png, &pngInfo); });
    if (pngInfo == nullptr) {
        throw std::runtime_error{"Error creating PNG info struct"};
    }

    if (setjmp(png_jmpbuf(png))) { // NOLINT
        throw std::runtime_error("Error setting up PNG error handling");
    }

    png_init_io(png, fileHandle);

    png_set_IHDR(png, pngInfo, width, height, sizeof(Image::ElementType) * 8, PNG_COLOR_TYPE_GRAY,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    std::vector<png_bytep> rows(height);
    for (std::uint32_t row = 0; row < height; row++) {
        rows[row] = const_cast<Image::ElementType*>(&image({0, row}));
    }
    png_set_rows(png, pngInfo, rows.data());
    png_write_png(png, pngInfo, PNG_TRANSFORM_IDENTITY, nullptr);
}

} // namespace op