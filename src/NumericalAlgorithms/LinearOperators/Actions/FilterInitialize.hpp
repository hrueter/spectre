// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "AlgorithmSingleton.hpp"
#include "Parallel/Invoke.hpp"

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Filter.hpp"
#include "NumericalAlgorithms/Spectral/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Printf.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t>
class Mesh;

namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace Actions {
/*!
 * \ingroup ActionsGroup
 * \ingroup NumericalAlgorithmsGroup
 * \ingroup SpectralGroup
 * \brief Initialize the filter matrices in the StaticCache
 *
 * Uses:
 * - ConstGlobalCache:
 *   - OptionTags::ExpFilterAlpha
 *   - OptionTags::ExpFilterS
 * - DataBox:
 *   - Tags::Mesh<volume_dim>
 * - System:
 *   - volume_dim
 *
 * DataBox changes:
 * - Adds: nothing
 * - Removes: nothing
 * - Modifies: nothing
 */
struct FilterInitialize {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    using system = typename Metavariables::system;
    constexpr size_t volume_dim = system::volume_dim;

    const Mesh<volume_dim>& mesh = db::get<Tags::Mesh<volume_dim>>(box);
    const double& alpha = Parallel::get<OptionTags::ExpFilterAlpha>(cache);
    const size_t& s = Parallel::get<OptionTags::ExpFilterS>(cache);

    filter_cache_initialize(mesh, alpha, s);

    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
