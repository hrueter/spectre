// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

/// \cond
template <size_t>
class Mesh;

namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

/*!
 * \ingroup NumericalAlgorithmsGroup
 * \ingroup SpectralGroup
 * \brief Apply filter for filter stabilization to data
 *
 * \f$u^{result} = F^{nodal} u^{input}\f$
 *
 * Here \f$F^{nodal}\f$ is the exponential filter matrix
 * \see Spectral::exponential_filter_matrix
 *
 * The filter can be applied to any type VarType that is accepted as input for
 * `apply_matrices`, i.e. `DataVector`s and Variables<VariableTags>.
 * Internally apply_matrices will assert that the number of grid points matches
 * between `result` and `input`.
 * \see apply_matrices
 *
 * Since this function is templated on VarType, which can be of type
 * Variables<VariableTags>, there are several possible instantiations for the
 * different VariableTags, which can not be explicitely instantiated in
 * Filter.cpp. Hence the definition of these functions are instead outsourced to
 * Filter.tpp.
 *
 * \tparam VarType either DataVector or Variables<TagsList>
 * \tparam Dim total number of dimensions
 *
 * \param result filtered output nodal coefficients
 * \param input values on the collocation points to filter
 * \param mesh the mesh on which the data is located
 * \param alpha damping factor in the exponent for exponential filter
 * \param s damping exponent for exponential filter
 *
 * \warning the `gsl::not_null` variant assumes `*result` is of the correct
 * size.
 * FIX: remove this warning, apply_matrices will assert
 * FIX: do we need an assert to check that the number of variables is correct?
 * FIX: Assert VarType is of allowed type (see TypeTraits)
 * FIX: delete/replace cache when alpha or s changed?
 */
template <typename VarType, size_t Dim>
void filter(gsl::not_null<VarType*> result, const VarType& input,
            const Mesh<Dim>& mesh, double alpha, size_t s) noexcept;

/**
 * \ingroup NumericalAlgorithmsGroup
 * \ingroup SpectralGroup
 * \brief Apply filter for filter stabilization to data
 *
 * \param input values on the collocation points to filter
 * \param mesh the mesh on which the data is located
 * \param alpha damping factor in the exponent for exponential filter
 * \param s damping exponent for exponential filter
 *
 * \return filtered output nodal coefficients
 */
template <typename VarType, size_t Dim>
VarType filter(const VarType& input, const Mesh<Dim>& mesh, double alpha,
               size_t s) noexcept;

/**
 * \ingroup NumericalAlgorithmsGroup
 * \ingroup SpectralGroup
 * \brief Apply filter for filter stabilization to data
 * using the cached matrix initialized by filter_cache_initialize
 *
 * \see filter_cache_initialize
 * \see exponential_filter_matrix_get_cached
 * \see filter
 *
 * \param result filtered output nodal coefficients
 * \param input values on the collocation points to filter
 * \param mesh the mesh on which the data is located
 */
template <typename VarType, size_t Dim>
void filter_with_cached_matrix(gsl::not_null<VarType*> result,
                               const VarType& input,
                               const Mesh<Dim>& mesh) noexcept;

/**
 * \ingroup NumericalAlgorithmsGroup
 * \ingroup SpectralGroup
 * \brief Apply filter for filter stabilization to data
 * using the cached matrix initialized by filter_cache_initialize
 *
 * \see filter_cache_initialize
 * \see exponential_filter_matrix_get_cached
 * \see filter
 *
 * \param input values on the collocation points to filter
 * \param mesh the mesh on which the data is located
 *
 * \return filtered output nodal coefficients
 */
template <typename VarType, size_t Dim>
VarType filter_with_cached_matrix(const VarType& input,
                                  const Mesh<Dim>& mesh) noexcept;

/*!
 * \ingroup NumericalAlgorithmsGroup
 * \ingroup SpectralGroup
 * \brief Generate cache containing the filter for filter stabilization.
 * This function uses an exponential filter matrix.
 * \see Spectral::exponential_filter_matrix
 *
 * \tparam Dim total number of dimensions
 *
 * \param mesh the mesh on which the data is located
 * \param alpha damping factor in the exponent for exponential filter
 * \param s damping exponent for exponential filter
 *
 * \warning Once the cache is generated it can not be changed, even if the
 * function is invoked with different parameters alpha or s. An invokation with
 * different basis (or quadrature) however will generate a second cache.
 */
template <size_t Dim>
void filter_cache_initialize(const Mesh<Dim>& mesh, double alpha,
                             size_t s) noexcept;

#include "NumericalAlgorithms/LinearOperators/Filter.tpp"  // IWYU pragma: keep
