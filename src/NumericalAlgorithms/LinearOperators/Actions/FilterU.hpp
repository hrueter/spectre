// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

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
 * \brief Apply the filter that has been cached in Actions::FilterInitialize to
 * the state vector
 *
 * With:
 * - variables_tag = system::variables_tag
 *
 * Uses:
 * - DataBox:
 *   - variables_tag
 *   - Tags::Mesh<volume_dim>
 * - System:
 *   - volume_dim
 *
 * FIX: use extra filtered_variables_tag for variables to filter(?)
 * argument_tags?
 * At the moment all vars in variables_tag are filtered
 *
 * DataBox changes:
 * - Adds: nothing
 * - Removes: nothing
 * - Modifies:
 *   - variables_tag
 */
struct FilterU {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    using system = typename Metavariables::system;
    using variables_tag = typename system::variables_tag;
    constexpr size_t volume_dim = system::volume_dim;

    db::mutate<variables_tag>(
        make_not_null(&box),
        [](const gsl::not_null<db::item_type<variables_tag>*> filtered_u,
           const db::item_type<variables_tag>& u,
           const db::item_type<Tags::Mesh<volume_dim>>& mesh) noexcept {
          *filtered_u = filter_with_cached_matrix(u, mesh);
          // FIX: check whether this inplace version can be used
          // filter_with_cached_matrix(filtered_u, u, mesh, alpha, s);
        },
        db::get<variables_tag>(box), db::get<Tags::Mesh<volume_dim>>(box));

    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
