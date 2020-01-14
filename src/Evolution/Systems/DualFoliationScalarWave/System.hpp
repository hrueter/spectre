// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class DualFoliationScalarWaveSystem.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "Evolution/Systems/DualFoliationScalarWave/Equations.hpp"
#include "Evolution/Systems/ScalarWave/Equations.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace Tags {
template <class>
struct Variables;
}  // namespace Tags

/*!
 * \ingroup EvolutionSystemsGroup
 * \brief Items related to evolving the scalar wave equation:
 */
namespace DualFoliationScalarWave {

template <size_t Dim>
struct System {
  static constexpr bool is_in_flux_conservative_form = false;
  static constexpr bool has_primitive_and_conservative_vars = false;
  static constexpr size_t volume_dim = Dim;

  using variables_tag = Tags::Variables<
      tmpl::list<::ScalarWave::Pi, ::ScalarWave::Phi<Dim>, ::ScalarWave::Psi>>;
  // Typelist of which subset of the variables to take the gradient of.
  using gradients_tags = tmpl::list<::ScalarWave::Pi, ::ScalarWave::Phi<Dim>>;

  // fixme: implement dual foliation version
  using compute_time_derivative = ::ScalarWave::ComputeDuDt<Dim>;
  using normal_dot_fluxes = ::ScalarWave::ComputeNormalDotFluxes<Dim>;
  using compute_largest_characteristic_speed =
      ::ScalarWave::ComputeLargestCharacteristicSpeed;

  template <typename Tag>
  using magnitude_tag = Tags::EuclideanMagnitude<Tag>;
};
}  // namespace DualFoliationScalarWave
