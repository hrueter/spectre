// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarWave/Conversion.hpp"

#include <algorithm>
#include <array>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/ScalarWave/RadialVPlus/System.hpp"
#include "Evolution/Systems/ScalarWave/System.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"  // IWYU pragma: keep

namespace ScalarWave {
/* Convert between ScalarWave and ScalarWave::RadialVPlus
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
 */
template <size_t Dim>
void get_radialvplus_from_standard_variables(
    const gsl::not_null<Scalar<DataVector>*> radialvplus_phi,
    const gsl::not_null<Scalar<DataVector>*> radialvplus_phiplus,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        radialvplus_lambda,
    const Scalar<DataVector>& standard_psi,
    const Scalar<DataVector>& standard_pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& standard_phi,
    const Scalar<DataVector>& chi, const Scalar<DataVector>& sigma,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& xhat) noexcept {
  /*
    if (UNLIKELY(get_size(get(*radialvplus_phi)) !=
                 get_size(get(standard_psi)))) {
      *radialvplus_phi = Scalar<DataVector>(get_size(get(standard_psi)));
      *radialvplus_phiplus = Scalar<DataVector>(get_size(get(standard_psi)));
      *radialvplus_lambda =
          tnsr::i<DataVector, Dim,
    Frame::Inertial>(get_size(get(standard_psi)));
    }
  */
  radialvplus_phi->get() = chi.get() * standard_psi.get();

  radialvplus_phiplus->get() =
      pow<2>(chi.get()) *
      (-standard_pi.get() + dot_product(xhat, standard_phi).get());
  radialvplus_phiplus->get() += sigma.get() * chi.get() * standard_psi.get();

  for (size_t d = 0; d < Dim; d++) {
    radialvplus_lambda->get(d) = chi.get() * standard_phi.get(d);
  }
}

template <size_t Dim>
void get_standard_from_radialvplus_variables(
    const gsl::not_null<Scalar<DataVector>*> standard_psi,
    const gsl::not_null<Scalar<DataVector>*> standard_pi,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        standard_phi,
    const Scalar<DataVector>& radialvplus_phi,
    const Scalar<DataVector>& radialvplus_phiplus,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& radialvplus_lambda,
    const Scalar<DataVector>& chi, const Scalar<DataVector>& sigma,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& xhat) noexcept {
  /*
    if (UNLIKELY(get_size(get(*standard_psi)) !=
                 get_size(get(radialvplus_phi)))) {
      *standard_psi = Scalar<DataVector>(get_size(get(radialvplus_phi)));
      *standard_pi = Scalar<DataVector>(get_size(get(radialvplus_phi)));
      *standard_phi = tnsr::i<DataVector, Dim, Frame::Inertial>(
          get_size(get(radialvplus_phi)));
    }
  */
  standard_psi->get() = 1.0 / chi.get() * radialvplus_phi.get();

  standard_pi->get() =
      -(radialvplus_phiplus.get() - sigma.get() * radialvplus_phi.get()) /
      pow<2>(chi.get());
  standard_pi->get() += dot_product(xhat, radialvplus_lambda).get() / chi.get();

  for (size_t d = 0; d < Dim; d++) {
    standard_phi->get(d) = 1.0 / chi.get() * radialvplus_lambda.get(d);
  }
}
}  // namespace ScalarWave

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(_, data)                                               \
  template void ScalarWave::get_radialvplus_from_standard_variables(         \
      const gsl::not_null<Scalar<DataVector>*> radialvplus_phi,              \
      const gsl::not_null<Scalar<DataVector>*> radialvplus_phiplus,          \
      const gsl::not_null<tnsr::i<DataVector, DIM(data), Frame::Inertial>*>  \
          radialvplus_lambda,                                                \
      const Scalar<DataVector>& standard_psi,                                \
      const Scalar<DataVector>& standard_pi,                                 \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>& standard_phi,   \
      const Scalar<DataVector>& chi, const Scalar<DataVector>& sigma,        \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& xhat) noexcept; \
  template void ScalarWave::get_standard_from_radialvplus_variables(         \
      const gsl::not_null<Scalar<DataVector>*> standard_psi,                 \
      const gsl::not_null<Scalar<DataVector>*> standard_pi,                  \
      const gsl::not_null<tnsr::i<DataVector, DIM(data), Frame::Inertial>*>  \
          standard_phi,                                                      \
      const Scalar<DataVector>& radialvplus_phi,                             \
      const Scalar<DataVector>& radialvplus_phiplus,                         \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                 \
          radialvplus_lambda,                                                \
      const Scalar<DataVector>& chi, const Scalar<DataVector>& sigma,        \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& xhat) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
