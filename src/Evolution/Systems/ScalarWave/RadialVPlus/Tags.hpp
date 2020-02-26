// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines DataBox tags for ScalarWave::RadialVPlus system

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/ScalarWave/RadialVPlus/TagsDeclarations.hpp"

class DataVector;

namespace ScalarWave {
namespace RadialVPlus {
namespace Tags {
struct Phi : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "Phi"; }
};

struct PhiPlus : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "PhiPlus"; }
};

template <size_t Dim>
struct Lambda : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
  static std::string name() noexcept { return "Lambda"; }
};

// @{
/*!
 * \brief rescaling function
 */
struct RescalingChi : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "RescalingChi"; }
};

struct RescalingSigma : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "RescalingSigma"; }
};
// @}

/*!
 * \brief Radius
 *
 * \f$r = \sum_i x^i x^i\f$
 *
 * ToCheck: Should we use EuclideanMagnitude tags?
 */
struct Radius : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "Radius"; }
};

/*!
 * \brief Radial normal vector
 *
 * Normal vector orthogonal to constant radius surfaces.
 * \f$\hat x^i = x^i / r\f$
 *
 * ToCheck: Should we use Normalized / NormalizedCompute tags?
 * Should this be an array instead? Index up or down does not matter for
 * coordinates
 */
template <size_t Dim>
struct RadialNormal : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim, Frame::Inertial>;
  static std::string name() noexcept { return "RadialNormal"; }
};

/*!
 * \brief Constraint damping parameter
 */
struct ConstraintGamma2 : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "ConstraintGamma2"; }
};

/*!
 * \brief Tag for the one-index constraint of the ScalarWave system
 *
 * For details on how this is defined and computed, see
 * `OneIndexConstraintCompute`.
 */
template <size_t Dim>
struct OneIndexConstraint : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
};
/*!
 * \brief Tag for the two-index constraint of the ScalarWave system
 *
 * For details on how this is defined and computed, see
 * `TwoIndexConstraintCompute`.
 */
template <size_t Dim>
struct TwoIndexConstraint : db::SimpleTag {
  using type = tnsr::ij<DataVector, Dim, Frame::Inertial>;
};

// @{
/*!
 * \brief Tags corresponding to the characteristic fields of the flat-spacetime
 * scalar-wave system.
 *
 * \details For details on how these are defined and computed, \see
 * CharacteristicSpeedsCompute
 */
struct VPhi : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "VPhi"; }
};
template <size_t Dim>
struct VZero : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
  static std::string name() noexcept { return "VZero"; }
};
struct VPlus : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "VPlus"; }
};
struct VMinus : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "VMinus"; }
};
// @}

template <size_t Dim>
struct CharacteristicSpeeds : db::SimpleTag {
  using type = std::array<DataVector, 4>;
  static std::string name() noexcept { return "CharacteristicSpeeds"; }
};

template <size_t Dim>
struct CharacteristicFields : db::SimpleTag {
  using type = Variables<tmpl::list<VPhi, VZero<Dim>, VPlus, VMinus>>;
  static std::string name() noexcept { return "CharacteristicFields"; }
};

template <size_t Dim>
struct EvolvedFieldsFromCharacteristicFields : db::SimpleTag {
  using type = Variables<tmpl::list<Phi, PhiPlus, Lambda<Dim>>>;
  static std::string name() noexcept {
    return "EvolvedFieldsFromCharacteristicFields";
  }
};
}  // namespace Tags
}  // namespace RadialVPlus
}  // namespace ScalarWave
