// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/ScalarWave/Conversion.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Gsl.hpp"

template <size_t Dim>
void test_conversion() noexcept {
  const size_t npts = 3;

  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist(-10., 10.);

  const auto nn_generator = make_not_null(&generator);
  const auto nn_dist = make_not_null(&dist);

  // Get characteristic speeds locally
  const auto psi_in =
      make_with_random_values<Scalar<DataVector>>(nn_generator, nn_dist, npts);
  const auto pi_in =
      make_with_random_values<Scalar<DataVector>>(nn_generator, nn_dist, npts);
  const auto phi_in =
      make_with_random_values<tnsr::i<DataVector, Dim, Frame::Inertial>>(
          nn_generator, nn_dist, npts);

  Scalar<DataVector> psi_out(npts);
  Scalar<DataVector> pi_out(npts);
  tnsr::i<DataVector, Dim, Frame::Inertial> phi_out(npts);

  // Test conversion to ScalarWave::RadialVPlus system
  std::uniform_real_distribution<> non_negative_dist(0.0, 10.);
  const auto nn_non_negative_dist = make_not_null(&non_negative_dist);

  const auto chi = make_with_random_values<Scalar<DataVector>>(
      nn_generator, nn_non_negative_dist, npts);
  const auto sigma = make_with_random_values<Scalar<DataVector>>(
      nn_generator, nn_non_negative_dist, npts);
  auto coords =
      make_with_random_values<tnsr::I<DataVector, Dim, Frame::Inertial>>(
          nn_generator, nn_dist, npts);

  // avoid rare case, where r = 0
  Scalar<DataVector> r = magnitude(coords);
  for (size_t idx = 0; idx < get_size(r.get()); idx++) {
    if (get(r)[idx] == 0.0) {
      coords.get(0)[idx] = 1.0;
      r.get()[idx] = 1.0;
    }
  }

  tnsr::I<DataVector, Dim, Frame::Inertial> xhat = coords;
  for (size_t i = 0; i < Dim; ++i) {
    xhat.get(i) /= r.get();
  }

  Scalar<DataVector> radialvplus_phi(npts);
  Scalar<DataVector> radialvplus_phiplus(npts);
  tnsr::i<DataVector, Dim, Frame::Inertial> radialvplus_lambda(npts);

  ScalarWave::get_radialvplus_from_standard_variables(
      make_not_null(&radialvplus_phi), make_not_null(&radialvplus_phiplus),
      make_not_null(&radialvplus_lambda), psi_in, pi_in, phi_in, chi, sigma,
      xhat);

  ScalarWave::get_standard_from_radialvplus_variables(
      make_not_null(&psi_out), make_not_null(&pi_out), make_not_null(&phi_out),
      radialvplus_phi, radialvplus_phiplus, radialvplus_lambda, chi, sigma,
      xhat);

  // check that after converting back and forth the result is the same
  CHECK_ITERABLE_APPROX(psi_in, psi_out);
  CHECK_ITERABLE_APPROX(pi_in, pi_out);
  CHECK_ITERABLE_APPROX(phi_in, phi_out);
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ScalarWave.Conversion",
                  "[Unit][Evolution]") {
  test_conversion<1>();
  test_conversion<2>();
  test_conversion<3>();
}
