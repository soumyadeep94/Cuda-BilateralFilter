/*
 * Copyright 2023 LUMA Vision Limited
 * Company Number 606427, Ireland
 * All rights reserved
 */

#pragma once

#include "NdVector.hpp"

#include <filesystem>

namespace lv::png
{
/**
 * @brief Loads the image at the given @a path.
 *
 * @param path Path to load image from.
 * @return Loaded image (first dimension width, second dimension height).
 * @throws std::runtime_error on file i/o error.
 */
[[nodiscard]] Image load(const std::filesystem::path& path);

/**
 * @brief Writes the given @a image to the given @a path.
 *
 * Overwrites existing data.
 *
 * @param image Image to save (first dimension width, second dimension height).
 * @param path Path to write file to.
 * @throws std::runtime_error on file i/o error.
 */
void save(const Image& image, const std::filesystem::path& path);

} // namespace lv::png