// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/ScalarWave/RadialVPlus/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"

/// \cond
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace ScalarWave {
namespace RadialVPlus {

struct ConstraintGamma2Compute : Tags::ConstraintGamma2, db::ComputeTag {
  using argument_tags = tmpl::list<Tags::Phi>;
  static auto function(const Scalar<DataVector>& phi) noexcept {
    return make_with_value<type>(phi, 0.);
  }
  using base = Tags::ConstraintGamma2;
};

// @{
/**
 * @copydoc OneIndexConstraintCompute
 */
template <size_t Dim>
tnsr::i<DataVector, Dim, Frame::Inertial> one_index_constraint(
    const Scalar<DataVector>& phi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& d_phi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& lambda,
    const Scalar<DataVector>& chi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& d_chi) noexcept;

template <size_t Dim>
void one_index_constraint(
    gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> constraint,
    const Scalar<DataVector>& phi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& d_phi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& lambda,
    const Scalar<DataVector>& chi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& d_chi) noexcept;
// @}

// @{
/**
 * @copydoc TwoIndexConstraintCompute
 */
template <size_t Dim>
tnsr::ij<DataVector, Dim, Frame::Inertial> two_index_constraint(
    const tnsr::i<DataVector, Dim, Frame::Inertial>& lambda,
    const tnsr::ij<DataVector, Dim, Frame::Inertial>& d_lambda,
    const Scalar<DataVector>& chi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& d_chi) noexcept;

template <size_t Dim>
void two_index_constraint(
    gsl::not_null<tnsr::ij<DataVector, Dim, Frame::Inertial>*> constraint,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& lambda,
    const tnsr::ij<DataVector, Dim, Frame::Inertial>& d_lambda,
    const Scalar<DataVector>& chi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& d_chi) noexcept;
// @}

/*!
 * \brief Compute the scalar-wave one-index constraint.
 *
 * \details Computes the scalar-wave one-index constraint,
 * \f$C_{i} = \partial_i \Phi - \Lambda_{i}
 *            - \Phi \frac{\partial_i \chi}{\chi} .\f$
 */
template <size_t Dim>
struct OneIndexConstraintCompute : Tags::OneIndexConstraint<Dim>,
                                   db::ComputeTag {
  using argument_tags = tmpl::list<
      Tags::Phi, ::Tags::deriv<Tags::Phi, tmpl::size_t<Dim>, Frame::Inertial>,
      Tags::Lambda<Dim>, Tags::RescalingChi,
      ::Tags::deriv<Tags::RescalingChi, tmpl::size_t<Dim>, Frame::Inertial>>;
  using return_type = tnsr::i<DataVector, Dim, Frame::Inertial>;
  static constexpr void (*function)(
      const gsl::not_null<return_type*> result, const Scalar<DataVector>&,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&,
      const Scalar<DataVector>&,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&) =
      &one_index_constraint<Dim>;
  using base = Tags::OneIndexConstraint<Dim>;
};

/*!
 * \brief Compute the scalar-wave 2-index constraint.
 *
 * \details Computes the scalar-wave 2-index constraint
 * \f$C_{ij} = \partial_i\Lambda_j - \partial_j\Lambda_i
 *           - \Lambda_j \frac{\partial_i \chi}{\chi}
 *           + \Lambda_i \frac{\partial_j \chi}{\chi} ,\f$
 * where \f$\Lambda_i\f$ is the rescaled reduction
 * variable.
 *
 * \note We do not support custom storage for antisymmetric tensors yet.
 */
template <size_t Dim>
struct TwoIndexConstraintCompute : Tags::TwoIndexConstraint<Dim>,
                                   db::ComputeTag {
  using argument_tags = tmpl::list<
      Tags::Lambda<Dim>,
      ::Tags::deriv<Tags::Lambda<Dim>, tmpl::size_t<Dim>, Frame::Inertial>,
      Tags::RescalingChi,
      ::Tags::deriv<Tags::RescalingChi, tmpl::size_t<Dim>, Frame::Inertial>>;
  using return_type = tnsr::ij<DataVector, Dim, Frame::Inertial>;
  static constexpr void (*function)(
      const gsl::not_null<return_type*> result,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&,
      const tnsr::ij<DataVector, Dim, Frame::Inertial>&,
      const Scalar<DataVector>&,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&) =
      &two_index_constraint<Dim>;
  using base = Tags::TwoIndexConstraint<Dim>;
};

}  // namespace RadialVPlus
}  // namespace ScalarWave
