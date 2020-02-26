// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/FaceNormal.hpp"
#include "Evolution/Systems/ScalarWave/RadialVPlus/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <typename>
class Variables;

namespace Tags {
template <typename Tag>
struct Normalized;
}  // namespace Tags
/// \endcond

namespace ScalarWave {
namespace RadialVPlus {

// @{
/**
 * @copydoc CharacteristicSpeedsCompute
 */
template <size_t Dim>
std::array<DataVector, 4> characteristic_speeds(
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        unit_normal_one_form) noexcept;

template <size_t Dim>
void characteristic_speeds(
    gsl::not_null<std::array<DataVector, 4>*> char_speeds,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        unit_normal_one_form) noexcept;
// @}

/*!
 * \brief Compute the characteristic speeds for the ScalarWave::RadialVplus
 * system.
 *
 * The characteristic fields are \f$v^{\hat \Phi}\f$, \f$v^{\hat 0}_{i}\f$ and
 * \f$v^{\hat \pm}\f$, with corresponding characteristic speeds
 * \f$\lambda_{\hat \alpha}\f$ given by:
 *
 * \f{align*}
 * \lambda_{\hat \psi} =& 0 \\
 * \lambda_{\hat 0} =& 0 \\
 * \lambda_{\hat \pm} =& \pm 1.
 * \f}
 */
namespace Tags {
template <size_t Dim>
struct CharacteristicSpeedsCompute : Tags::CharacteristicSpeeds<Dim>,
                                     db::ComputeTag {
  using base = Tags::CharacteristicSpeeds<Dim>;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>>;

  static void function(gsl::not_null<return_type*> char_speeds,
                       const tnsr::i<DataVector, Dim, Frame::Inertial>&
                           unit_normal_one_form) noexcept {
    characteristic_speeds(char_speeds, unit_normal_one_form);
  };
};
}  // namespace Tags

// @{
/**
 * @copydoc CharacteristicFieldsCompute
 */
template <size_t Dim>
Variables<tmpl::list<Tags::VPhi, Tags::VZero<Dim>, Tags::VPlus, Tags::VMinus>>
characteristic_fields(const Scalar<DataVector>& phi,
                      const Scalar<DataVector>& phiplus,
                      const tnsr::i<DataVector, Dim, Frame::Inertial>& lambda,
                      const Scalar<DataVector>& gamma_2,
                      const Scalar<DataVector>& chi,
                      const Scalar<DataVector>& sigma,
                      const tnsr::I<DataVector, Dim, Frame::Inertial>& xhat,
                      const tnsr::i<DataVector, Dim, Frame::Inertial>&
                          unit_normal_one_form) noexcept;

template <size_t Dim>
void characteristic_fields(
    gsl::not_null<Variables<
        tmpl::list<Tags::VPhi, Tags::VZero<Dim>, Tags::VPlus, Tags::VMinus>>*>
        char_fields,
    const Scalar<DataVector>& phi, const Scalar<DataVector>& phiplus,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& lambda,
    const Scalar<DataVector>& gamma_2, const Scalar<DataVector>& chi,
    const Scalar<DataVector>& sigma,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& xhat,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        unit_normal_one_form) noexcept;
// @}

/*!
 * \brief Compute characteristic fields from evolved fields
 *
 * \ref Tags::CharacteristicFieldsCompute and
 * \ref Tags::EvolvedFieldsFromCharacteristicFieldsCompute convert between
 * characteristic and evolved fields for the scalar-wave system.
 *
 * \ref Tags::CharacteristicFieldsCompute computes characteristic fields:
 *
 * \f{align*}
 * v^{\hat \Phi}  =& \Phi \\
 * v^{\hat 0}_{i} =& (\delta^k_i - n_i n^k) \Lambda_{k} \\
 * v^{\hat \pm} =& (\chi \gamma_2 ( 1 \mp \hat x^i n_i )  - \sigma) \Phi
 *               + \Phi^{+}
 *               + \chi (\pm n^i - \hat x^i) \Lambda_i ,
 * \f}
 *
 * where \f$\Phi = \chi \Psi\f$ is the rescaled scalar field,
 * \f$\Phi^{+} = \chi^2 (\partial_t \Psi + \hat x^i \partial_i \Psi)
 * + \sigma \chi \Psi\f$ is the rescaled outgoing characteristic, and
 * \f$\Lambda_i=\chi \partial_i \Psi\f$ is the rescaled reduction variable.
 * If \f$ \sigma = 0\f$ and \f$n^i\f$ being identical to the outward pointing
 * radial normal \f$\hat x^i\f$, then \f$v^{\hat +} = \Phi^{+}\f$.
 * (And \f$v^{\hat -} = \Phi^{+}\f$, if \f$n^i = -\hat x^i\f$ .)
 *
 * \ref Tags::EvolvedFieldsFromCharacteristicFieldsCompute computes evolved
 * fields \f$u_\alpha\f$ in terms of the characteristic fields. This uses the
 * inverse of above relations:
 *
 * \f{align*}
 * \Phi =& v^{\hat \Phi}, \\
 * \Phi^{+} =& (\sigma  + \chi \gamma_2 ((\hat x^i n_i)^2 -1)) v^{\hat \Phi}
 *           + \chi \hat x^i v^{\hat 0}_{i}
 *           + \frac{1}{2}( (1 + \hat x^i n_i) v^{\hat +}
 *                        + (1 - \hat x^i n_i) v^{\hat -}) \\
 * \Lambda_{i} =& n_i \gamma_2 \hat x^k n_k v^{\hat \Phi}
 *              + v^{\hat 0}_{i}
 *              + n_i \frac{1}{2 \chi} (v^{\hat +} - v^{\hat -})
 * \f}
 *
 * The corresponding characteristic speeds \f$\lambda_{\hat \alpha}\f$
 * are computed by \ref Tags::CharacteristicSpeedsCompute .
 */
namespace Tags {
template <size_t Dim>
struct CharacteristicFieldsCompute : Tags::CharacteristicFields<Dim>,
                                     db::ComputeTag {
  using base = Tags::CharacteristicFields<Dim>;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<Tags::Phi, Tags::PhiPlus, Tags::Lambda<Dim>,
                 Tags::ConstraintGamma2, Tags::RescalingChi,
                 Tags::RescalingSigma, Tags::RadialNormal<Dim>,
                 ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>>;

  static void function(const gsl::not_null<return_type*> char_fields,
                       const Scalar<DataVector>& phi,
                       const Scalar<DataVector>& phiplus,
                       const tnsr::i<DataVector, Dim, Frame::Inertial>& lambda,
                       const Scalar<DataVector>& gamma_2,
                       const Scalar<DataVector>& chi,
                       const Scalar<DataVector>& sigma,
                       const tnsr::I<DataVector, Dim, Frame::Inertial>& xhat,
                       const tnsr::i<DataVector, Dim, Frame::Inertial>&
                           unit_normal_one_form) noexcept {
    characteristic_fields(char_fields, phi, phiplus, lambda, gamma_2, chi,
                          sigma, xhat, unit_normal_one_form);
  };
};
}  // namespace Tags

// @{
/**
 * @copydoc EvolvedFieldsFromCharacteristicFieldsCompute
 */
template <size_t Dim>
Variables<tmpl::list<Tags::Phi, Tags::PhiPlus, Tags::Lambda<Dim>>>
evolved_fields_from_characteristic_fields(
    const Scalar<DataVector>& v_phi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& v_zero,
    const Scalar<DataVector>& v_plus, const Scalar<DataVector>& v_minus,
    const Scalar<DataVector>& gamma_2, const Scalar<DataVector>& chi,
    const Scalar<DataVector>& sigma,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& xhat,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        unit_normal_one_form) noexcept;

template <size_t Dim>
void evolved_fields_from_characteristic_fields(
    gsl::not_null<
        Variables<tmpl::list<Tags::Phi, Tags::PhiPlus, Tags::Lambda<Dim>>>*>
        evolved_fields,
    const Scalar<DataVector>& v_phi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& v_zero,
    const Scalar<DataVector>& v_plus, const Scalar<DataVector>& v_minus,
    const Scalar<DataVector>& gamma_2, const Scalar<DataVector>& chi,
    const Scalar<DataVector>& sigma,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& xhat,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        unit_normal_one_form) noexcept;
// @}

/*!
 * \brief Compute evolved fields from characteristic fields.
 *
 * For expressions used here to compute evolved fields from characteristic ones,
 * see \ref Tags::CharacteristicFieldsCompute.
 */
namespace Tags {
template <size_t Dim>
struct EvolvedFieldsFromCharacteristicFieldsCompute
    : Tags::EvolvedFieldsFromCharacteristicFields<Dim>,
      db::ComputeTag {
  using base = Tags::EvolvedFieldsFromCharacteristicFields<Dim>;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<Tags::VPhi, Tags::VZero<Dim>, Tags::VPlus, Tags::VMinus,
                 Tags::ConstraintGamma2, Tags::RescalingChi,
                 Tags::RescalingSigma, Tags::RadialNormal<Dim>,
                 ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>>;

  static void function(const gsl::not_null<return_type*> evolved_fields,
                       const Scalar<DataVector>& v_phi,
                       const tnsr::i<DataVector, Dim, Frame::Inertial>& v_zero,
                       const Scalar<DataVector>& v_plus,
                       const Scalar<DataVector>& v_minus,
                       const Scalar<DataVector>& gamma_2,
                       const Scalar<DataVector>& chi,
                       const Scalar<DataVector>& sigma,
                       const tnsr::I<DataVector, Dim, Frame::Inertial>& xhat,
                       const tnsr::i<DataVector, Dim, Frame::Inertial>&
                           unit_normal_one_form) noexcept {
    evolved_fields_from_characteristic_fields(
        evolved_fields, v_phi, v_zero, v_plus, v_minus, gamma_2, chi, sigma,
        xhat, unit_normal_one_form);
  };
};
}  // namespace Tags

}  // namespace RadialVPlus
}  // namespace ScalarWave
