// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cmath>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/ModalVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Mesh.hpp"
#include "NumericalAlgorithms/LinearOperators/CoefficientTransforms.hpp"
#include "NumericalAlgorithms/LinearOperators/Filter.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <size_t Dim>
struct Var1 {
  using type = tnsr::i<DataVector, Dim, Frame::Grid>;
};

struct Var2 {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
using two_vars = tmpl::list<Var1<Dim>, Var2>;

template <size_t Dim>
using one_var = tmpl::list<Var1<Dim>>;

// test that filter works on Variables
// test that all variables are filtered
// test that order 0 mode is unfiltered
// test that highest mode is almost completely filtered out
template <Spectral::Basis Basis, Spectral::Quadrature Quadrature>
void test_filter_variables() noexcept {
  CAPTURE(Basis);
  CAPTURE(Quadrature);
  const size_t n_grid_pts = 7;
  const double alpha = 30.0;
  const size_t s = 20;
  const Mesh<3> mesh{
      {{n_grid_pts + 2, n_grid_pts + 1, n_grid_pts}}, Basis, Quadrature};
  Variables<two_vars<3>> variables{mesh.number_of_grid_points()};

  Var1<3>::type& var1_tensor = get<Var1<3>>(variables);
  Var2::type& var2_tensor = get<Var2>(variables);

  DataVector& var11_data = get<0>(var1_tensor);
  DataVector& var12_data = get<1>(var1_tensor);
  DataVector& var13_data = get<2>(var1_tensor);
  DataVector& var2_data = get(var2_tensor);
  ModalVector modal_vector_only_highest_mode{mesh.number_of_grid_points(), 0.0};
  const ModalVector modal_vector_coefficients_all_one{
      mesh.number_of_grid_points(), 1.0};

  // constant data has only order 0 mode
  var11_data = 1.0;

  // create vectors with only the highest mode
  modal_vector_only_highest_mode.at(modal_vector_only_highest_mode.size() - 1) =
      1;
  to_nodal_coefficients(make_not_null(&var12_data),
                        modal_vector_only_highest_mode, mesh);

  // create vector that contains all coefficients
  to_nodal_coefficients(make_not_null(&var13_data),
                        modal_vector_coefficients_all_one, mesh);
  // same for var2, just to test that all variables are filtered
  to_nodal_coefficients(make_not_null(&var2_data),
                        modal_vector_coefficients_all_one, mesh);

  // CREATE EXPECTED DATAVECTORS //
  // var11 had only order 0 mode, which is never filtered
  DataVector expected_var11_data{mesh.number_of_grid_points(), 1.0};
  // var12 had only highst mode, which is extremely damped
  DataVector expected_var12_data{mesh.number_of_grid_points(), 0.0};

  // make filtered vector for comparison
  ModalVector modal_vector_filtered{mesh.number_of_grid_points()};
  for (size_t i = 0; i < mesh.extents(0); i++) {
    for (size_t j = 0; j < mesh.extents(1); j++) {
      for (size_t k = 0; k < mesh.extents(2); k++) {
        modal_vector_filtered.at(mesh.storage_index(Index<3>{i, j, k})) =
            exp(-alpha *
                pow(static_cast<double>(i) / (mesh.extents(0) - 1), s)) *
            exp(-alpha *
                pow(static_cast<double>(j) / (mesh.extents(1) - 1), s)) *
            exp(-alpha *
                pow(static_cast<double>(k) / (mesh.extents(2) - 1), s));
      }
    }
  }
  DataVector expected_var13_data =
      to_nodal_coefficients(modal_vector_filtered, mesh);
  DataVector expected_var2_data =
      to_nodal_coefficients(modal_vector_filtered, mesh);

  // APPLY FILTER //
  Variables<two_vars<3>> filtered_variables = filter(variables, mesh, alpha, s);

  // CHECK
  Var1<3>::type& filtered_var1_tensor = get<Var1<3>>(filtered_variables);
  Var2::type& filtered_var2_tensor = get<Var2>(filtered_variables);

  DataVector& filtered_var11_data = get<0>(filtered_var1_tensor);
  DataVector& filtered_var12_data = get<1>(filtered_var1_tensor);
  DataVector& filtered_var13_data = get<2>(filtered_var1_tensor);
  DataVector& filtered_var2_data = get(filtered_var2_tensor);

  CHECK_ITERABLE_APPROX(filtered_var11_data, expected_var11_data);
  CHECK_ITERABLE_APPROX(filtered_var12_data, expected_var12_data);

  if (Quadrature == Spectral::Quadrature::Gauss) {
    // for Gauss quadrature we need a looser error bound
    // (because(?) we use the analytic inverse Vandermonde matrix instead of the
    // numerical)
    Approx custom_approx = Approx::custom().epsilon(1.e-8).scale(1.0);
    CHECK_ITERABLE_CUSTOM_APPROX(filtered_var13_data, expected_var13_data,
                                 custom_approx);
    CHECK_ITERABLE_CUSTOM_APPROX(filtered_var2_data, expected_var2_data,
                                 custom_approx);
  } else {
    CHECK_ITERABLE_APPROX(filtered_var13_data, expected_var13_data);
    CHECK_ITERABLE_APPROX(filtered_var2_data, expected_var2_data);
  }
}

// test that filter works with DataVectors
// test that filter cache works
template <Spectral::Basis Basis, Spectral::Quadrature Quadrature>
void test_filter_cache() noexcept {
  CAPTURE(Basis);
  CAPTURE(Quadrature);
  const size_t n_grid_pts = 9;
  const double alpha = 20.0;
  const size_t s = 5;
  const Mesh<2> mesh({{n_grid_pts + 1, n_grid_pts}}, Basis, Quadrature);

  DataVector data{mesh.number_of_grid_points()};
  const ModalVector modal_vector_coefficients_all_one{
      mesh.number_of_grid_points(), 1.0};

  // create vector that contains all coefficients
  to_nodal_coefficients(make_not_null(&data), modal_vector_coefficients_all_one,
                        mesh);

  // CREATE EXPECTED DATAVECTOR //
  ModalVector modal_vector_filtered{mesh.number_of_grid_points()};
  for (size_t i = 0; i < mesh.extents(0); i++) {
    for (size_t j = 0; j < mesh.extents(1); j++) {
      modal_vector_filtered.at(mesh.storage_index(Index<2>{i, j})) =
          exp(-alpha * pow(static_cast<double>(i) / (mesh.extents(0) - 1), s)) *
          exp(-alpha * pow(static_cast<double>(j) / (mesh.extents(1) - 1), s));
    }
  }
  DataVector expected_data = to_nodal_coefficients(modal_vector_filtered, mesh);

  // APPLY FILTER //
  filter_cache_initialize(mesh, alpha, s);
  DataVector filtered_with_cache_data = filter_with_cached_matrix(data, mesh);
  DataVector filtered_without_cache_data = filter(data, mesh, alpha, s);

  // CHECK
  CHECK_ITERABLE_APPROX(filtered_with_cache_data, expected_data);
  CHECK_ITERABLE_APPROX(filtered_with_cache_data, filtered_without_cache_data);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.LinearOperators.Filter",
                  "[NumericalAlgorithms][LinearOperators][Unit]") {
  test_filter_variables<Spectral::Basis::Chebyshev,
                        Spectral::Quadrature::GaussLobatto>();
  test_filter_variables<Spectral::Basis::Chebyshev,
                        Spectral::Quadrature::Gauss>();
  test_filter_variables<Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto>();
  test_filter_variables<Spectral::Basis::Legendre,
                        Spectral::Quadrature::Gauss>();

  test_filter_cache<Spectral::Basis::Chebyshev,
                    Spectral::Quadrature::GaussLobatto>();
  test_filter_cache<Spectral::Basis::Chebyshev, Spectral::Quadrature::Gauss>();
  test_filter_cache<Spectral::Basis::Legendre,
                    Spectral::Quadrature::GaussLobatto>();
  test_filter_cache<Spectral::Basis::Legendre, Spectral::Quadrature::Gauss>();
}
