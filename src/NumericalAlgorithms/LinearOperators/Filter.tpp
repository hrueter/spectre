// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <functional>

#include "DataStructures/Matrix.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "ErrorHandling/Assert.hpp"
#include "NumericalAlgorithms/LinearOperators/ApplyMatrices.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace detail {
template <size_t Dim>
auto filter_matrices_from_cache_impl(const Mesh<Dim>& mesh) {
  const Matrix empty{};
  auto matrices = make_array<Dim>(std::cref(empty));
  for (size_t d = 0; d < Dim; d++) {
    gsl::at(matrices, d) = std::cref(
        Spectral::exponential_filter_matrix_get_cached(mesh.slice_through(d)));
  }

  return matrices;
}
}  // namespace detail

// Note: ASSERT is done inside `apply_matrices`
template <typename VarType, size_t Dim>
void filter_with_cached_matrix(const gsl::not_null<VarType*> result,
                               const VarType& input,
                               const Mesh<Dim>& mesh) noexcept {
  // FIX: Should we make sure, that this function is not used if input and
  // output vector are the same?
  // we are not using inplace versions here

  apply_matrices(result, detail::filter_matrices_from_cache_impl(mesh), input,
                 mesh.extents());
}

// VarType can be either DataVector or Variables<TagList>
template <typename VarType, size_t Dim>
VarType filter_with_cached_matrix(const VarType& input,
                                  const Mesh<Dim>& mesh) noexcept {
  // FIX: do result->initialize in returnbyref version? result->initialize(vars)
  // or result->initialize(npoints)
  // FIX: avoid this allocation completely?
  VarType result{input};
  filter_with_cached_matrix(make_not_null(&result), input, mesh);
  return result;
}

namespace detail {
template <size_t Dim>
auto filter_matrices_impl(const Mesh<Dim>& mesh, const double alpha,
                          const size_t s) {
  const Matrix empty{};
  auto matrices = make_array<Dim>(empty);
  for (size_t d = 0; d < Dim; d++) {
    gsl::at(matrices, d) =
        Spectral::exponential_filter_matrix(mesh.slice_through(d), alpha, s);
  }
  return matrices;
}
}  // namespace detail

// Note: ASSERT is done inside `apply_matrices`
template <typename VarType, size_t Dim>
void filter(const gsl::not_null<VarType*> result, const VarType& input,
            const Mesh<Dim>& mesh, const double alpha,
            const size_t s) noexcept {
  // FIX: Should we make sure, that this function is not used if input and
  // output vector are the same?
  // we are not using inplace versions here

  apply_matrices(result, detail::filter_matrices_impl(mesh, alpha, s), input,
                 mesh.extents());
}

// VarType can be either DataVector or Variables<TagList>
template <typename VarType, size_t Dim>
VarType filter(const VarType& input, const Mesh<Dim>& mesh, const double alpha,
               const size_t s) noexcept {
  // FIX: do result->initialize in returnbyref version? result->initialize(vars)
  // or result->initialize(npoints)
  // FIX: avoid this allocation completely?
  VarType result{input};
  filter(make_not_null(&result), input, mesh, alpha, s);
  return result;
}
