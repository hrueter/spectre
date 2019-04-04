// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>

#include "NumericalAlgorithms/LinearOperators/Actions/FilterInitialize.hpp"
#include "NumericalAlgorithms/Spectral/Tags.hpp"
#include "tests/Unit/ActionTesting.hpp"

/*
namespace {
struct Var : db::SimpleTag {
  static std::string name() noexcept { return "Var"; }
  using type = double;
};

struct System {
  using variables_tag = Var;
};

using variables_tag = Var;

struct Metavariables;
struct component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tag_list =
      tmpl::list<OptionTags::ExpFilterAlpha, OptionTags::ExpFilterS>;
  using action_list = tmpl::list<Actions::FilterInitialize>;
  using action_list = tmpl::list<
      DgElementArray<EvolutionMetavars, dg::Actions::InitializeElement<Dim>,
                     tmpl::flatten<tmpl::list<Actions::FilterInitialize>>>>;
  using simple_tags = tmpl::list<>;
  using initial_databox = db::compute_databox_type<simple_tags>;
};

struct Metavariables {
  using system = System;
  using component_list = tmpl::list<component>;
  using const_global_cache_tag_list = tmpl::list<>;
};
}  // namespace
*/
