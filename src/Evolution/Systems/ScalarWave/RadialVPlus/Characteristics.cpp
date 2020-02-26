// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarWave/RadialVPlus/Characteristics.hpp"

#include <array>

#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarWave {
namespace RadialVPlus {
template <size_t Dim>
void characteristic_speeds(
    const gsl::not_null<std::array<DataVector, 4>*> char_speeds,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        unit_normal_one_form) noexcept {
  destructive_resize_components(char_speeds,
                                get<0>(unit_normal_one_form).size());
  (*char_speeds)[0] = 0.;   // v(VPhi)
  (*char_speeds)[1] = 0.;   // v(VZero)
  (*char_speeds)[2] = 1.;   // v(VPlus)
  (*char_speeds)[3] = -1.;  // v(VMinus)
}

template <size_t Dim>
std::array<DataVector, 4> characteristic_speeds(
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        unit_normal_one_form) noexcept {
  auto char_speeds = make_with_value<std::array<DataVector, 4>>(
      get<0>(unit_normal_one_form), 0.);
  characteristic_speeds(make_not_null(&char_speeds), unit_normal_one_form);
  return char_speeds;
}

template <size_t Dim>
void characteristic_fields(
    const gsl::not_null<Variables<
        tmpl::list<Tags::VPhi, Tags::VZero<Dim>, Tags::VPlus, Tags::VMinus>>*>
        char_fields,
    const Scalar<DataVector>& phi, const Scalar<DataVector>& phiplus,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& lambda,
    const Scalar<DataVector>& gamma_2, const Scalar<DataVector>& chi,
    const Scalar<DataVector>& sigma,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& xhat,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        unit_normal_one_form) noexcept {
  if (UNLIKELY(char_fields->number_of_grid_points() != get(phi).size())) {
    char_fields->initialize(get(phi).size());
  }

  // Compute lambda_dot_normal = n^i \Lambda_{i} = \sum_i n_i \Lambda_{i}
  // (we use normal_one_form and normal_vector interchangeably in flat space)
  const Scalar<DataVector> lambda_dot_normal =
      dot_product(unit_normal_one_form, lambda);
  const Scalar<DataVector> lambda_dot_xhat = dot_product(xhat, lambda);
  const Scalar<DataVector> normal_dot_xhat =
      dot_product(unit_normal_one_form, xhat);

  // VPhi
  get<Tags::VPhi>(*char_fields) = phi;

  // VZero
  for (size_t i = 0; i < Dim; ++i) {
    get<Tags::VZero<Dim>>(*char_fields).get(i) =
        lambda.get(i) - unit_normal_one_form.get(i) * get(lambda_dot_normal);
  }

  // VPlus
  get<Tags::VPlus>(*char_fields).get() =
      (chi.get() * gamma_2.get() * (1.0 - normal_dot_xhat.get()) -
       sigma.get()) *
      phi.get();
  get<Tags::VPlus>(*char_fields).get() += phiplus.get();
  get<Tags::VPlus>(*char_fields).get() +=
      chi.get() * (lambda_dot_normal.get() - lambda_dot_xhat.get());

  // VMinus
  get<Tags::VMinus>(*char_fields).get() =
      (chi.get() * gamma_2.get() * (1.0 + normal_dot_xhat.get()) -
       sigma.get()) *
      phi.get();
  get<Tags::VMinus>(*char_fields).get() += phiplus.get();
  get<Tags::VMinus>(*char_fields).get() +=
      chi.get() * (-lambda_dot_normal.get() - lambda_dot_xhat.get());
}

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
                          unit_normal_one_form) noexcept {
  Variables<tmpl::list<Tags::VPhi, Tags::VZero<Dim>, Tags::VPlus, Tags::VMinus>>
      char_fields(get_size(get(gamma_2)));
  characteristic_fields(make_not_null(&char_fields), phi, phiplus, lambda,
                        gamma_2, chi, sigma, xhat, unit_normal_one_form);
  return char_fields;
}

template <size_t Dim>
void evolved_fields_from_characteristic_fields(
    const gsl::not_null<
        Variables<tmpl::list<Tags::Phi, Tags::PhiPlus, Tags::Lambda<Dim>>>*>
        evolved_fields,
    const Scalar<DataVector>& v_phi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& v_zero,
    const Scalar<DataVector>& v_plus, const Scalar<DataVector>& v_minus,
    const Scalar<DataVector>& gamma_2, const Scalar<DataVector>& chi,
    const Scalar<DataVector>& sigma,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& xhat,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        unit_normal_one_form) noexcept {
  if (UNLIKELY(evolved_fields->number_of_grid_points() != get(v_phi).size())) {
    evolved_fields->initialize(get(v_phi).size());
  }

  const Scalar<DataVector> normal_dot_xhat =
      dot_product(unit_normal_one_form, xhat);

  // Phi
  get<Tags::Phi>(*evolved_fields) = v_phi;

  // PhiPlus
  get<Tags::PhiPlus>(*evolved_fields).get() =
      (sigma.get() +
       chi.get() * gamma_2.get() * (pow<2>(normal_dot_xhat.get()) - 1.0)) *
      v_phi.get();
  get<Tags::PhiPlus>(*evolved_fields).get() +=
      chi.get() * dot_product(xhat, v_zero).get();
  get<Tags::PhiPlus>(*evolved_fields).get() +=
      0.5 * (1.0 + normal_dot_xhat.get()) * v_plus.get();
  get<Tags::PhiPlus>(*evolved_fields).get() +=
      0.5 * (1.0 - normal_dot_xhat.get()) * v_minus.get();

  // Lambda
  for (size_t i = 0; i < Dim; ++i) {
    get<Tags::Lambda<Dim>>(*evolved_fields).get(i) =
        unit_normal_one_form.get(i) * gamma_2.get() * normal_dot_xhat.get() *
        v_phi.get();
    get<Tags::Lambda<Dim>>(*evolved_fields).get(i) += v_zero.get(i);
    get<Tags::Lambda<Dim>>(*evolved_fields).get(i) +=
        unit_normal_one_form.get(i) * 0.5 * (v_plus.get() - v_minus.get()) /
        chi.get();
  }
}

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
        unit_normal_one_form) noexcept {
  Variables<tmpl::list<Tags::Phi, Tags::PhiPlus, Tags::Lambda<Dim>>>
      evolved_fields(get_size(get(gamma_2)));
  evolved_fields_from_characteristic_fields(
      make_not_null(&evolved_fields), v_phi, v_zero, v_plus, v_minus, gamma_2,
      chi, sigma, xhat, unit_normal_one_form);
  return evolved_fields;
}
}  // namespace RadialVPlus
}  // namespace ScalarWave

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                  \
  template void ScalarWave::RadialVPlus::characteristic_speeds(               \
      const gsl::not_null<std::array<DataVector, 4>*> char_speeds,            \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                  \
          unit_normal_one_form) noexcept;                                     \
  template std::array<DataVector, 4>                                          \
  ScalarWave::RadialVPlus::characteristic_speeds(                             \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                  \
          unit_normal_one_form) noexcept;                                     \
  template struct ScalarWave::RadialVPlus::Tags::CharacteristicSpeedsCompute< \
      DIM(data)>;                                                             \
  template void ScalarWave::RadialVPlus::characteristic_fields(               \
      const gsl::not_null<Variables<                                          \
          tmpl::list<ScalarWave::RadialVPlus::Tags::VPhi,                     \
                     ScalarWave::RadialVPlus::Tags::VZero<DIM(data)>,         \
                     ScalarWave::RadialVPlus::Tags::VPlus,                    \
                     ScalarWave::RadialVPlus::Tags::VMinus>>*>                \
          char_fields,                                                        \
      const Scalar<DataVector>& phi, const Scalar<DataVector>& phiplus,       \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>& lambda,          \
      const Scalar<DataVector>& gamma_2, const Scalar<DataVector>& chi,       \
      const Scalar<DataVector>& sigma,                                        \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& xhat,            \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                  \
          unit_normal_one_form) noexcept;                                     \
  template Variables<                                                         \
      tmpl::list<ScalarWave::RadialVPlus::Tags::VPhi,                         \
                 ScalarWave::RadialVPlus::Tags::VZero<DIM(data)>,             \
                 ScalarWave::RadialVPlus::Tags::VPlus,                        \
                 ScalarWave::RadialVPlus::Tags::VMinus>>                      \
  ScalarWave::RadialVPlus::characteristic_fields(                             \
      const Scalar<DataVector>& phi, const Scalar<DataVector>& phiplus,       \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>& lambda,          \
      const Scalar<DataVector>& gamma_2, const Scalar<DataVector>& chi,       \
      const Scalar<DataVector>& sigma,                                        \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& xhat,            \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                  \
          unit_normal_one_form) noexcept;                                     \
  template struct ScalarWave::RadialVPlus::Tags::CharacteristicFieldsCompute< \
      DIM(data)>;                                                             \
  template void                                                               \
  ScalarWave::RadialVPlus::evolved_fields_from_characteristic_fields(         \
      const gsl::not_null<Variables<                                          \
          tmpl::list<ScalarWave::RadialVPlus::Tags::Phi,                      \
                     ScalarWave::RadialVPlus::Tags::PhiPlus,                  \
                     ScalarWave::RadialVPlus::Tags::Lambda<DIM(data)>>>*>     \
          evolved_fields,                                                     \
      const Scalar<DataVector>& v_phi,                                        \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>& v_zero,          \
      const Scalar<DataVector>& v_plus, const Scalar<DataVector>& v_minus,    \
      const Scalar<DataVector>& gamma_2, const Scalar<DataVector>& chi,       \
      const Scalar<DataVector>& sigma,                                        \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& xhat,            \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                  \
          unit_normal_one_form) noexcept;                                     \
  template Variables<                                                         \
      tmpl::list<ScalarWave::RadialVPlus::Tags::Phi,                          \
                 ScalarWave::RadialVPlus::Tags::PhiPlus,                      \
                 ScalarWave::RadialVPlus::Tags::Lambda<DIM(data)>>>           \
  ScalarWave::RadialVPlus::evolved_fields_from_characteristic_fields(         \
      const Scalar<DataVector>& v_phi,                                        \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>& v_zero,          \
      const Scalar<DataVector>& v_plus, const Scalar<DataVector>& v_minus,    \
      const Scalar<DataVector>& gamma_2, const Scalar<DataVector>& chi,       \
      const Scalar<DataVector>& sigma,                                        \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& xhat,            \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                  \
          unit_normal_one_form) noexcept;                                     \
  template struct ScalarWave::RadialVPlus::Tags::                             \
      EvolvedFieldsFromCharacteristicFieldsCompute<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
