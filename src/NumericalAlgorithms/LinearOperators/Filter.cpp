// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/LinearOperators/Filter.hpp"

#include "DataStructures/DataVector.hpp"
#include "Domain/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/GenerateInstantiations.hpp"

template <size_t Dim>
void filter_cache_initialize(const Mesh<Dim>& mesh, const double alpha,
                             const size_t s) noexcept {
  for (size_t d = 0; d < Dim; d++) {
    Spectral::exponential_filter_matrix_make_cache(mesh.slice_through(d), alpha,
                                                   s);
  }
}

// FIX: Should we also allow filtering in only one direction?
// FIX: Do we need instantiations for different Variables?

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATION(r, data)                                              \
  template void filter<DataVector, GET_DIM(data)>(                          \
      const gsl::not_null<DataVector*>, const DataVector&,                  \
      const Mesh<GET_DIM(data)>&, const double, const size_t) noexcept;     \
  template DataVector filter<DataVector, GET_DIM(data)>(                    \
      const DataVector&, const Mesh<GET_DIM(data)>&, const double,          \
      const size_t) noexcept;                                               \
  template void filter_cache_initialize<GET_DIM(data)>(                     \
      const Mesh<GET_DIM(data)>&, const double, const size_t) noexcept;     \
  template void filter_with_cached_matrix<DataVector, GET_DIM(data)>(       \
      const gsl::not_null<DataVector*>, const DataVector&,                  \
      const Mesh<GET_DIM(data)>&) noexcept;                                 \
  template DataVector filter_with_cached_matrix<DataVector, GET_DIM(data)>( \
      const DataVector&, const Mesh<GET_DIM(data)>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef GET_DIM
#undef INSTANTIATION
