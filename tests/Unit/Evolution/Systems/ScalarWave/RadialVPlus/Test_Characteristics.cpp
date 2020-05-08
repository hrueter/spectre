// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <pup.h>
#include <random>

#include <iostream>

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"        // IWYU pragma: keep
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/Wedge2D.hpp"
#include "Domain/CoordinateMaps/Wedge3D.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Evolution/Systems/ScalarWave/Characteristics.hpp"
#include "Evolution/Systems/ScalarWave/Conversion.hpp"
#include "Evolution/Systems/ScalarWave/RadialVPlus/Characteristics.hpp"
#include "Evolution/Systems/ScalarWave/RadialVPlus/Constraints.hpp"
#include "Evolution/Systems/ScalarWave/RadialVPlus/Tags.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/MathFunctions/Gaussian.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_forward_declare Tags::dt
// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare Variables

// Handy type alias to implement the function for vectors and coverctors at the
// same time
template <typename DataType, size_t Dim, UpLo Ul, typename Fr = Frame::Inertial>
using Vector = Tensor<DataType, tmpl::integral_list<std::int32_t, 1>,
                      index_list<SpatialIndex<Dim, Ul, Fr>>>;

// construct non-zero vector by sampling spherical coordinates
// Note: we use a signed radius and let phi take values in the interval [0,1]
// to account for all possible values in one dimension
template <typename DataType, size_t Dim, UpLo Ul, typename Fr = Frame::Inertial>
const Vector<DataType, Dim, Ul, Fr> make_random_nonzero_vector(
    const gsl::not_null<std::mt19937*> nn_generator, const size_t npts,
    const double min_magnitude, const double max_magnitude) noexcept {
  // check dimension
  static_assert(Dim < 4,
                "Function is only implemented for vectors up to rank 4.");

  constexpr bool is_dim_greater1 = Dim > 1;
  constexpr bool is_dim_greater2 = Dim > 2;

  // generate distributions
  const std::vector<double> signedr_intervals{-max_magnitude, -min_magnitude,
                                              min_magnitude, max_magnitude};
  const std::vector<double> weights{1, 0, 1};
  std::piecewise_constant_distribution<> dist_signedr(
      signedr_intervals.begin(), signedr_intervals.end(), weights.begin());
  std::uniform_real_distribution<> dist_phi(0.0 * is_dim_greater1,
                                            M_PI * is_dim_greater1);
  std::uniform_real_distribution<> dist_sintheta(1.0 - 2. * is_dim_greater2,
                                                 1.0);

  // generate random values
  const Scalar<DataType> sintheta = make_with_random_values<Scalar<DataType>>(
      nn_generator, make_not_null(&dist_sintheta), npts);
  const Scalar<DataType> phi = make_with_random_values<Scalar<DataType>>(
      nn_generator, make_not_null(&dist_phi), npts);
  const Scalar<DataType> signedr = make_with_random_values<Scalar<DataType>>(
      nn_generator, make_not_null(&dist_signedr), npts);

  // construct vector
  Vector<DataType, Dim, Ul, Fr> x(npts);
  x.get(0) = signedr.get() * cos(phi.get()) * sintheta.get();
  if (is_dim_greater1)
    x.get(1) = signedr.get() * sin(phi.get()) * sintheta.get();
  if (is_dim_greater2)
    x.get(2) = signedr.get() * sqrt(1.0 - pow<2>(sintheta.get()));

  return x;
}

template <size_t Dim>
void test_characteristic_speeds() noexcept {
  TestHelpers::db::test_compute_tag<
      ScalarWave::RadialVPlus::Tags::CharacteristicSpeedsCompute<Dim>>(
      "CharacteristicSpeeds");

  // Outward 3-normal to the surface on which characteristic fields are
  // needed
  const tnsr::i<DataVector, Dim, Frame::Inertial> unit_normal_one_form{
      DataVector(1, 1. / sqrt(Dim))};

  std::array<DataVector, 4> char_speeds_standard{};
  std::array<DataVector, 4> char_speeds_radialvplus{};

  ScalarWave::Tags::CharacteristicSpeedsCompute<Dim>::function(
      &char_speeds_standard, unit_normal_one_form);

  ScalarWave::RadialVPlus::Tags::CharacteristicSpeedsCompute<Dim>::function(
      &char_speeds_radialvplus, unit_normal_one_form);

  // The speeds must be the same as in the standard system, so we compare
  CHECK_ITERABLE_APPROX(char_speeds_standard, char_speeds_radialvplus);
}

template <size_t Dim>
void test_characteristic_fields(const size_t npts) noexcept {
  TestHelpers::db::test_compute_tag<
      ScalarWave::RadialVPlus::Tags::CharacteristicFieldsCompute<Dim>>(
      "CharacteristicFields");

  // Get ingredients
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist(-10., 10.);

  const auto nn_generator = make_not_null(&generator);
  const auto nn_dist = make_not_null(&dist);

  const tnsr::I<DataVector, Dim, Frame::Inertial> x =
      make_random_nonzero_vector<DataVector, Dim, UpLo::Up, Frame::Inertial>(
          nn_generator, npts, 0.001, 10.0);
  // Outward 3-normal to the surface on which characteristic fields are needed
  /*
     const tnsr::i<DataVector, Dim, Frame::Inertial> unit_normal_one_form =
      make_random_nonzero_vector<DataVector, Dim, UpLo::Lo, Frame::Inertial>(
          nn_generator, npts, 0.1, 10.0);


  const auto gamma_2 =
      make_with_random_values<Scalar<DataVector>>(nn_generator, nn_dist, npts);
*/
  // SIMPLIFY FOR DEBUGGING
  tnsr::i<DataVector, Dim, Frame::Inertial> unit_normal_one_form(npts);
  for (size_t i = 0; i < Dim; i++) {
    unit_normal_one_form.get(i) = x.get(i) / magnitude(x).get();
  }
  const auto gamma_2 = make_with_value<Scalar<DataVector>>(npts, 0.0);

  // generate standard fields
  const auto psi_standard =
      make_with_random_values<Scalar<DataVector>>(nn_generator, nn_dist, npts);
  const auto pi_standard =
      make_with_random_values<Scalar<DataVector>>(nn_generator, nn_dist, npts);
  const auto phi_standard =
      make_with_random_values<tnsr::i<DataVector, Dim, Frame::Inertial>>(
          nn_generator, nn_dist, npts);

  // get radialvplus extra variables
  std::uniform_real_distribution<> non_negative_dist(0.0, 10.);
  const auto nn_non_negative_dist = make_not_null(&non_negative_dist);

  /*
  const auto chi = make_with_random_values<Scalar<DataVector>>(
      nn_generator, nn_non_negative_dist, npts);
  const auto sigma = make_with_random_values<Scalar<DataVector>>(
      nn_generator, nn_non_negative_dist, npts);
*/
  // SIMPLIFY FOR DEBUGGING
  const auto chi = make_with_value<Scalar<DataVector>>(npts, 1.0);
  const auto sigma = make_with_value<Scalar<DataVector>>(npts, 0.0);

  const Scalar<DataVector> r = magnitude(x);
  tnsr::I<DataVector, Dim, Frame::Inertial> xhat = x;
  for (size_t i = 0; i < Dim; ++i) {
    xhat.get(i) /= r.get();
  }

  // get radialvplus fields
  Scalar<DataVector> phi_radialvplus(npts);
  Scalar<DataVector> phiplus_radialvplus(npts);
  tnsr::i<DataVector, Dim, Frame::Inertial> lambda_radialvplus(npts);

  ScalarWave::get_radialvplus_from_standard_variables(
      make_not_null(&phi_radialvplus), make_not_null(&phiplus_radialvplus),
      make_not_null(&lambda_radialvplus), psi_standard, pi_standard,
      phi_standard, chi, sigma, xhat);

  // get characteristic fields

  Variables<tmpl::list<ScalarWave::Tags::VPsi, ScalarWave::Tags::VZero<Dim>,
                       ScalarWave::Tags::VPlus, ScalarWave::Tags::VMinus>>
      characteristic_fields_standard{};

  ScalarWave::Tags::CharacteristicFieldsCompute<Dim>::function(
      make_not_null(&characteristic_fields_standard), gamma_2, psi_standard,
      pi_standard, phi_standard, unit_normal_one_form);

  const auto& v_psi_standard =
      get<ScalarWave::Tags::VPsi>(characteristic_fields_standard);
  const auto& v_zero_standard =
      get<ScalarWave::Tags::VZero<Dim>>(characteristic_fields_standard);
  const auto& v_plus_standard =
      get<ScalarWave::Tags::VPlus>(characteristic_fields_standard);
  const auto& v_minus_standard =
      get<ScalarWave::Tags::VMinus>(characteristic_fields_standard);

  Variables<tmpl::list<ScalarWave::RadialVPlus::Tags::VPhi,
                       ScalarWave::RadialVPlus::Tags::VZero<Dim>,
                       ScalarWave::RadialVPlus::Tags::VPlus,
                       ScalarWave::RadialVPlus::Tags::VMinus>>
      characteristic_fields_radialvplus{};

  ScalarWave::RadialVPlus::Tags::CharacteristicFieldsCompute<Dim>::function(
      make_not_null(&characteristic_fields_radialvplus), phi_radialvplus,
      phiplus_radialvplus, lambda_radialvplus, gamma_2, chi, sigma, xhat,
      unit_normal_one_form);

  auto v_phi_radialvplus = get<ScalarWave::RadialVPlus::Tags::VPhi>(
      characteristic_fields_radialvplus);
  auto v_zero_radialvplus = get<ScalarWave::RadialVPlus::Tags::VZero<Dim>>(
      characteristic_fields_radialvplus);
  auto v_plus_radialvplus = get<ScalarWave::RadialVPlus::Tags::VPlus>(
      characteristic_fields_radialvplus);
  auto v_minus_radialvplus = get<ScalarWave::RadialVPlus::Tags::VMinus>(
      characteristic_fields_radialvplus);

  // rescale radialvplus characteristic fields to compare
  v_phi_radialvplus.get() /= chi.get();
  for (size_t i = 0; i < Dim; ++i) {
    v_zero_radialvplus.get(i) /= chi.get();
  }
  v_plus_radialvplus.get() /= pow<2>(chi.get());
  v_minus_radialvplus.get() /= pow<2>(chi.get());

  std::cout << "chi: " << chi.get() << std::endl;

  std::cout << "phiplus_radialvplus: " << phiplus_radialvplus.get()
            << std::endl;

  //***** THERE IS A CONFUSION WITH V+ and V- between the too systems!!! ****//

  // Compare characteristic fields
  CHECK_ITERABLE_APPROX(v_psi_standard, v_phi_radialvplus);
  CHECK_ITERABLE_APPROX(v_zero_standard, v_zero_radialvplus);
  CHECK_ITERABLE_APPROX(v_minus_standard, v_plus_radialvplus);
  // CHECK_ITERABLE_APPROX(v_minus_standard, v_minus_radialvplus);

  // get radialvplus evolved fields back from characteristic fields
  TestHelpers::db::test_compute_tag<
      ScalarWave::RadialVPlus::Tags::
          EvolvedFieldsFromCharacteristicFieldsCompute<Dim>>(
      "EvolvedFieldsFromCharacteristicFields");

  Variables<tmpl::list<ScalarWave::RadialVPlus::Tags::Phi,
                       ScalarWave::RadialVPlus::Tags::PhiPlus,
                       ScalarWave::RadialVPlus::Tags::Lambda<Dim>>>
      evolved_radialvplus_fields_from_characteristics{};
  ScalarWave::RadialVPlus::Tags::
      EvolvedFieldsFromCharacteristicFieldsCompute<Dim>::function(
          make_not_null(&evolved_radialvplus_fields_from_characteristics),
          v_phi_radialvplus, v_zero_radialvplus, v_plus_radialvplus,
          v_minus_radialvplus, gamma_2, chi, sigma, xhat, unit_normal_one_form);

  const auto& phi_radialvplus_from_characteristic =
      get<ScalarWave::RadialVPlus::Tags::Phi>(
          evolved_radialvplus_fields_from_characteristics);
  const auto& phiplus_radialvplus_from_characteristic =
      get<ScalarWave::RadialVPlus::Tags::PhiPlus>(
          evolved_radialvplus_fields_from_characteristics);
  const auto& lambda_radialvplus_from_characteristic =
      get<ScalarWave::RadialVPlus::Tags::Lambda<Dim>>(
          evolved_radialvplus_fields_from_characteristics);

  // compare evolved fields
  /*
  CHECK_ITERABLE_APPROX(phi_radialvplus, phi_radialvplus_from_characteristic);
  CHECK_ITERABLE_APPROX(phiplus_radialvplus,
                        phiplus_radialvplus_from_characteristic);
  CHECK_ITERABLE_APPROX(lambda_radialvplus,
                        lambda_radialvplus_from_characteristic);
*/
}

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.ScalarWave.RadialVPlus.Characteristics",
    "[Unit][Evolution]") {
  TestHelpers::db::test_compute_tag<
      ScalarWave::RadialVPlus::ConstraintGamma2Compute>("ConstraintGamma2");

  test_characteristic_speeds<1>();
  test_characteristic_speeds<2>();
  test_characteristic_speeds<3>();

  test_characteristic_fields<1>(5);
  // test_characteristic_fields<2>(5);
  // test_characteristic_fields<3>(5);

  // TODO: Use this grid also for other tests
  // TODO: test failure
  // Divide/Strucutre characteristic fields test case
}
