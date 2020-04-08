// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/ScalarWave/RadialVPlus/System.hpp"
#include "Evolution/Systems/ScalarWave/System.hpp"

namespace ScalarWave {

// @{
/** @brief Convert between ScalarWave and ScalarWave::RadialVPlus
 *
 * Standard system:
 * \f{align*}
 * \Psi \phantom{=}&       \\
 * \Pi  =& - \partial_t \Psi \\
 * \Phi_i =& \partial_i \Psi
 * \f}
 *
 * RadialVPlus system:
 * \f{align*}
 * \Phi      =& \chi \Psi \\
 * \Phi^{+}  =& \chi^2 (\partial_t \Psi + \hat x^i \partial_i \Psi)
 *                + \sigma \chi \Psi \\
 *           =& \chi^2 (- \Pi + \hat x^i \Phi_i)
 *                + \sigma \chi \Psi \\
 * \Lambda_i =& \chi \partial_i \Psi \\
 *           =& \chi \Phi_i
 * \f}
 *
 * The same equations also convert between the respective time derivatives.
 */
template <size_t Dim>
void get_radialvplus_from_standard_variables(
    gsl::not_null<Scalar<DataVector>*> radialvplus_phi,
    gsl::not_null<Scalar<DataVector>*> radialvplus_phiplus,
    gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        radialvplus_lambda,
    const Scalar<DataVector>& standard_psi,
    const Scalar<DataVector>& standard_pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& standard_phi,
    const Scalar<DataVector>& chi, const Scalar<DataVector>& sigma,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& xhat) noexcept;

template <size_t Dim>
void get_standard_from_radialvplus_variables(
    gsl::not_null<Scalar<DataVector>*> standard_psi,
    gsl::not_null<Scalar<DataVector>*> standard_pi,
    gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> standard_phi,
    const Scalar<DataVector>& radialvplus_phi,
    const Scalar<DataVector>& radialvplus_phiplus,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& radialvplus_lambda,
    const Scalar<DataVector>& chi, const Scalar<DataVector>& sigma,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& xhat) noexcept;
// @}
}  // namespace ScalarWave
