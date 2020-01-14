// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines DataBox tags for dual foliation scalar wave system

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"

class DataVector;

namespace DualFoliationScalarWave {
struct RescaledPsi : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "RescaledPsi"; }
};

struct RescaledPi : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "RescaledPi"; }
};

template <size_t Dim>
struct RescaledPhi : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
  static std::string name() noexcept { return "RescaledPhi"; }
};
}  // namespace DualFoliationScalarWave
