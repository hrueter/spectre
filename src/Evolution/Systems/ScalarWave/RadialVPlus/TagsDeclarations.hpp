// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

namespace ScalarWave {
namespace RadialVPlus {
/// \brief Tags for the ScalarWave::RadialVPlus evolution system
namespace Tags {
struct Phi;
struct PhiPlus;
template <size_t Dim>
struct Lambda;

struct ConstraintGamma2;

struct RescalingChi;
struct RescalingSigma;
struct Radius;
template <size_t Dim>
struct RadialNormal;

template <size_t Dim>
struct OneIndexConstraint;
template <size_t Dim>
struct TwoIndexConstraint;

struct VPhi;
template <size_t Dim>
struct VZero;
struct VPlus;
struct VMinus;

template <size_t Dim>
struct CharacteristicSpeeds;
template <size_t Dim>
struct CharacteristicFields;
template <size_t Dim>
struct EvolvedFieldsFromCharacteristicFields;
}  // namespace Tags
}  // namespace RadialVPlus
}  // namespace ScalarWave
