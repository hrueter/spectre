// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarWave/RadialVPlus/Equations.hpp"

#include <algorithm>
#include <array>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
#include "Evolution/Systems/ScalarWave/RadialVPlus/System.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"   // IWYU pragma: keep
#include "Utilities/TMPL.hpp"  // IWYU pragma: keep

// IWYU pragma: no_forward_declare Tensor

template <size_t Dim>
DataVector f(const Scalar<DataVector>& phi,
             const Scalar<DataVector>& /*phiplus*/,
             const tnsr::i<DataVector, Dim, Frame::Inertial>& /*lambda*/) {
  return 0.0 * phi.get();
}

namespace ScalarWave {
namespace RadialVPlus {
// Doxygen is not good at templates and so we have to hide the definition.
/// \cond
template <size_t Dim>
void ComputeDuDt<Dim>::apply(
    const gsl::not_null<Scalar<DataVector>*> dt_phi,
    const gsl::not_null<Scalar<DataVector>*> dt_phiplus,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> dt_lambda,
    const Scalar<DataVector>& phi, const Scalar<DataVector>& phiplus,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& lambda,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& d_phi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& d_phiplus,
    const tnsr::ij<DataVector, Dim, Frame::Inertial>& d_lambda,
    const Scalar<DataVector>& gamma2, const Scalar<DataVector>& chi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& d_chi,
    const Scalar<DataVector>& sigma,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& d_sigma,
    const Scalar<DataVector>& radius,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& xhat) noexcept {
  const DataVector chi_squared = pow<2>(chi.get());
  const Scalar<DataVector> xhat_dot_d_chi = dot_product(xhat, d_chi);
  const Scalar<DataVector> xhat_dot_lambda = dot_product(xhat, lambda);

  // magnitude in StdArrayHelpers
  // DataVector r = magnitude(x);

  // dt phi
  dt_phi->get() = 1.0 / chi.get() * phiplus.get() - sigma.get() / chi.get();
  dt_phi->get() -= xhat_dot_lambda.get();

  // dt phiplus
  dt_phiplus->get() = chi_squared * f(phi, phiplus, lambda);
  for (size_t i = 0; i < Dim; i++) {
    dt_phiplus->get() -= sigma.get() * xhat.get(i) * d_phi.get(i);
    dt_phiplus->get() += xhat.get(i) * d_phiplus.get(i);
    dt_phiplus->get() += chi.get() * d_lambda.get(i, i);
    for (size_t j = 0; j < Dim; j++) {
      dt_phiplus->get() -=
          chi.get() * xhat.get(i) * xhat.get(j) * d_lambda.get(i, j);
    }
    dt_phiplus->get() -= phi.get() * xhat.get(i) * d_sigma.get(i);
  }
  dt_phiplus->get() +=
      (phiplus.get() - sigma.get() * phi.get()) *
      (sigma.get() / chi.get() + 2 * xhat_dot_d_chi.get() / chi.get());

  // dt lambda
  for (size_t i = 0; i < Dim; i++) {
    dt_lambda->get(i) = -sigma.get() / chi.get() * d_phi.get(i);
    dt_lambda->get(i) += 1.0 / chi.get() * d_phiplus.get(i);
    for (size_t j = 0; j < Dim; j++) {
      dt_lambda->get(i) -= xhat.get(j) * d_lambda.get(i, j);
    }
    dt_lambda->get(i) -= 2 * d_chi.get(i) / chi_squared *
                         (phiplus.get() - sigma.get() * phi.get());
    dt_lambda->get(i) -= phi.get() * d_sigma.get(i) / chi.get();
    dt_lambda->get(i) -= lambda.get(i) / radius.get();
    for (size_t j = 0; j < Dim; j++) {
      dt_lambda->get(i) +=
          xhat.get(i) * xhat.get(j) * lambda.get(j) / radius.get();
    }
    dt_lambda->get(i) += xhat_dot_lambda.get() * d_chi.get(i) / chi.get();
    dt_lambda->get(i) +=
        gamma2.get() *
        (d_phi.get(i) - phi.get() * d_chi.get(i) / chi.get() - lambda.get(i));
  }
}

template <size_t Dim>
void ComputeNormalDotFluxes<Dim>::apply(
    const gsl::not_null<Scalar<DataVector>*> phi_normal_dot_flux,
    const gsl::not_null<Scalar<DataVector>*> phiplus_normal_dot_flux,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        lambda_normal_dot_flux,
    const Scalar<DataVector>& phi, const Scalar<DataVector>& phiplus,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& lambda,
    const Scalar<DataVector>& gamma2, const Scalar<DataVector>& chi,
    const Scalar<DataVector>& sigma,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& xhat,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        interface_unit_normal) noexcept {
  const Scalar<DataVector> xhat_dot_normal =
      dot_product(xhat, interface_unit_normal);
  const Scalar<DataVector> xhat_dot_lambda = dot_product(xhat, lambda);

  // phi_normal_dot_flux
  // We assume that all values of phi_normal_dot_flux are the same. The
  // reason is that std::fill is actually surprisingly/disappointingly slow.
  if (phi_normal_dot_flux->get()[0] != 0.0) {
    std::fill(phi_normal_dot_flux->get().begin(),
              phi_normal_dot_flux->get().end(), 0.0);
  }

  // phiplus_normal_dot_flux
  phiplus_normal_dot_flux->get() =
      -sigma.get() * xhat_dot_normal.get() * phi.get();
  phiplus_normal_dot_flux->get() += xhat_dot_normal.get() * phiplus.get();
  phiplus_normal_dot_flux->get() +=
      chi.get() * dot_product(interface_unit_normal, lambda).get();
  phiplus_normal_dot_flux->get() -=
      chi.get() * xhat_dot_normal.get() * xhat_dot_lambda.get();

  // lambda_normal_dot_flux
  for (size_t i = 0; i < Dim; i++) {
    lambda_normal_dot_flux->get(i) =
        -sigma.get() / chi.get() * interface_unit_normal.get(i) * phi.get();
    lambda_normal_dot_flux->get(i) +=
        1.0 / chi.get() * interface_unit_normal.get(i) * phiplus.get();
    lambda_normal_dot_flux->get(i) -=
        interface_unit_normal.get(i) * xhat_dot_lambda.get();
    lambda_normal_dot_flux->get(i) +=
        gamma2.get() * interface_unit_normal.get(i) * phi.get();
  }
}

template <size_t Dim>
void PenaltyFlux<Dim>::package_data(
    const gsl::not_null<Variables<package_tags>*> packaged_data,
    const Scalar<DataVector>& normal_dot_flux_phiplus,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_dot_flux_lambda,
    const Scalar<DataVector>& v_plus, const Scalar<DataVector>& v_minus,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& interface_unit_normal)
    const noexcept {
  // Computes the contribution to the numerical flux from one side of the
  // interface.
  //
  // Note: when PenaltyFlux::operator() is called, an Element passes in its
  // own packaged data to fill the interior fields, and its neighbor's
  // packaged data to fill the exterior fields. This introduces a sign flip
  // for each normal used in computing the exterior fields.
  get<::Tags::NormalDotFlux<Tags::PhiPlus>>(*packaged_data) =
      normal_dot_flux_phiplus;
  get<::Tags::NormalDotFlux<Tags::Lambda<Dim>>>(*packaged_data) =
      normal_dot_flux_lambda;
  get<Tags::VPlus>(*packaged_data) = v_plus;
  get<Tags::VMinus>(*packaged_data) = v_minus;
  auto& normal_times_v_plus = get<NormalTimesVPlus>(*packaged_data);
  auto& normal_times_v_minus = get<NormalTimesVMinus>(*packaged_data);
  for (size_t d = 0; d < Dim; ++d) {
    normal_times_v_plus.get(d) = get(v_plus) * interface_unit_normal.get(d);
    normal_times_v_minus.get(d) = get(v_minus) * interface_unit_normal.get(d);
  }
}

template <size_t Dim>
void PenaltyFlux<Dim>::operator()(
    const gsl::not_null<Scalar<DataVector>*> phi_normal_dot_numerical_flux,
    const gsl::not_null<Scalar<DataVector>*> phiplus_normal_dot_numerical_flux,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        lambda_normal_dot_numerical_flux,
    const Scalar<DataVector>& normal_dot_flux_phiplus_interior,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        normal_dot_flux_lambda_interior,
    const Scalar<DataVector>& /* v_plus_interior */,
    const Scalar<DataVector>& v_minus_interior,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
    /* normal_times_v_plus_interior */,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        normal_times_v_minus_interior,
    const Scalar<DataVector>& /* minus_normal_dot_flux_phiplus_exterior */,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
    /* minus_normal_dot_flux_lambda_exterior */,
    const Scalar<DataVector>& v_plus_exterior,
    const Scalar<DataVector>& /* v_minus_exterior */,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        minus_normal_times_v_plus_exterior,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
    /* minus_normal_times_v_minus_exterior */) const noexcept {
  constexpr double penalty_factor = 1.;

  // NormalDotNumericalFlux<Phi>
  std::fill(phi_normal_dot_numerical_flux->get().begin(),
            phi_normal_dot_numerical_flux->get().end(), 0.);

  // NormalDotNumericalFlux<PhiPlus>
  phiplus_normal_dot_numerical_flux->get() =
      normal_dot_flux_phiplus_interior.get() +
      0.5 * penalty_factor * (v_minus_interior.get() - v_plus_exterior.get());

  // NormalDotNumericalFlux<lambda>
  for (size_t d = 0; d < Dim; ++d) {
    lambda_normal_dot_numerical_flux->get(d) =
        normal_dot_flux_lambda_interior.get(d) -
        0.5 * penalty_factor *
            (normal_times_v_minus_interior.get(d) +
             minus_normal_times_v_plus_exterior.get(d));
  }
}

template <size_t Dim>
void UpwindFlux<Dim>::package_data(
    const gsl::not_null<Variables<package_tags>*> packaged_data,
    const Scalar<DataVector>& normal_dot_flux_phiplus,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_dot_flux_lambda,
    const Scalar<DataVector>& phiplus,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& interface_unit_normal)
    const noexcept {
  // Computes the contribution to the numerical flux from one side of the
  // interface.
  //
  // The packaged_data stores:
  // <PhiPlus> = phiplus
  // <NormalDotFlux<Phiplus>> = normal_dot_flux_phiplus
  // <NormalTimesFluxPhiplus_i> = normal_dot_flux_phiplus * n_i
  // <NormalDotFlux<Lambda>_i> = normal_dot_flux_lambda_i
  //
  // Note: when Upwind::operator() is called, an Element passes in its own
  // packaged data to fill the interior fields, and its neighbors packaged
  // data to fill the exterior fields. This introduces a sign flip for each
  // normal used in computing the exterior fields.
  get<Tags::PhiPlus>(*packaged_data) = phiplus;
  get<::Tags::NormalDotFlux<Tags::PhiPlus>>(*packaged_data) =
      normal_dot_flux_phiplus;
  get<::Tags::NormalDotFlux<Tags::Lambda<Dim>>>(*packaged_data) =
      normal_dot_flux_lambda;
  auto& normal_times_flux_phiplus = get<NormalTimesFluxPhiPlus>(*packaged_data);
  for (size_t d = 0; d < Dim; ++d) {
    normal_times_flux_phiplus.get(d) =
        interface_unit_normal.get(d) * get(normal_dot_flux_phiplus);
  }
}

template <size_t Dim>
void UpwindFlux<Dim>::operator()(
    const gsl::not_null<Scalar<DataVector>*> phi_normal_dot_numerical_flux,
    const gsl::not_null<Scalar<DataVector>*> phiplus_normal_dot_numerical_flux,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        lambda_normal_dot_numerical_flux,
    const Scalar<DataVector>& normal_dot_flux_phiplus_interior,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        normal_dot_flux_lambda_interior,
    const Scalar<DataVector>& phiplus_interior,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        normal_times_flux_phiplus_interior,
    const Scalar<DataVector>& minus_normal_dot_flux_phiplus_exterior,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        minus_normal_dot_flux_lambda_exterior,
    const Scalar<DataVector>& phiplus_exterior,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        normal_times_flux_phiplus_exterior) const noexcept {
  std::fill(phi_normal_dot_numerical_flux->get().begin(),
            phi_normal_dot_numerical_flux->get().end(), 0.);

  phiplus_normal_dot_numerical_flux->get() =
      0.5 * (phiplus_interior.get() - phiplus_exterior.get() +
             normal_dot_flux_phiplus_interior.get() -
             minus_normal_dot_flux_phiplus_exterior.get());

  for (size_t d = 0; d < Dim; ++d) {
    lambda_normal_dot_numerical_flux->get(d) =
        0.5 * (normal_dot_flux_lambda_interior.get(d) -
               minus_normal_dot_flux_lambda_exterior.get(d) +
               normal_times_flux_phiplus_interior.get(d) -
               normal_times_flux_phiplus_exterior.get(d));
  }
}
/// \endcond
}  // namespace RadialVPlus
}  // namespace ScalarWave

// Generate explicit instantiations of partial_derivatives function as well as
// all other functions in Equations.cpp

#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"  // IWYU pragma: keep

template <size_t Dim>
using derivative_tags =
    typename ScalarWave::RadialVPlus::System<Dim>::gradients_tags;

template <size_t Dim>
using variables_tags =
    typename ScalarWave::RadialVPlus::System<Dim>::variables_tag::tags_list;

using derivative_frame = Frame::Inertial;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(_, data)                                                \
  template struct ScalarWave::RadialVPlus::ComputeDuDt<DIM(data)>;            \
  template struct ScalarWave::RadialVPlus::ComputeNormalDotFluxes<DIM(data)>; \
  template struct ScalarWave::RadialVPlus::UpwindFlux<DIM(data)>;             \
  template struct ScalarWave::RadialVPlus::PenaltyFlux<DIM(data)>;            \
  template Variables<                                                         \
      db::wrap_tags_in<::Tags::deriv, derivative_tags<DIM(data)>,             \
                       tmpl::size_t<DIM(data)>, derivative_frame>>            \
  partial_derivatives<derivative_tags<DIM(data)>, variables_tags<DIM(data)>,  \
                      DIM(data), derivative_frame>(                           \
      const Variables<variables_tags<DIM(data)>>& u,                          \
      const Mesh<DIM(data)>& mesh,                                            \
      const InverseJacobian<DataVector, DIM(data), Frame::Logical,            \
                            derivative_frame>& inverse_jacobian) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
