// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>

#include "Domain/Mesh.hpp"
#include "NumericalAlgorithms/LinearOperators/Actions/FilterInitialize.hpp"
#include "NumericalAlgorithms/Spectral/Tags.hpp"
#include "tests/Unit/ActionTesting.hpp"

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
  using simple_tags = tmpl::list<>;
  using initial_databox = db::compute_databox_type<simple_tags>;
};

struct Metavariables {
  using system = System;
  using component_list = tmpl::list<component>;
  using const_global_cache_tag_list = tmpl::list<>;
};
}  // namespace

void test_filter_u() {
  constexpr double alpha = 15.0;
  constexpr size_t s = 40;

  constexpr size_t dim = 2;
  const size_t order = 5;
  using my_component = component<Metavariables>;
  const Mesh<dim> mesh{order, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};

  const ElementId<dim> self_id{0};
  const Element<dim> element(self_id, {});

  auto start_box = [&mesh, &element ]() noexcept {
    // auto map = ElementMap<2, Frame::Inertial>(self_id,
    // coordmap->get_clone());
    auto var = Scalar<DataVector>(mesh.number_of_grid_points(), 1.);
    return db::create<my_component::simple_tags>(0, mesh, element,
                                                 std::move(var));
  }
  ();

  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  using MockDistributedObjectsTag =
      MockRuntimeSystem::MockDistributedObjectsTag<my_component>;
  MockRuntimeSystem::TupleOfMockDistributedObjects dist_objects{};
  tuples::get<MockDistributedObjectsTag>(dist_objects)
      .emplace(self_id, std::move(start_box));

  ActionTesting::MockRuntimeSystem<Metavariables> runner{
      {}, std::move(dist_objects)};

  const auto& var_to_filter =
      db::get<Var>(runner.algorithms<my_component>()
                       .at(self_id)
                       .get_databox<my_component::initial_databox>());

  filter_cache_initialize(mesh, alpha, s);

  CHECK_ITERABLE_APPROX(var_to_filter,
                        Scalar<DataVector>(mesh.number_of_grid_points(), 1.));

  runner.template next_action<my_component>(self_id);

  CHECK_ITERABLE_APPROX(var_to_filter,
                        Scalar<DataVector>(mesh.number_of_grid_points(), 1.));
}

SPECTRE_TEST_CASE("Unit.Numerical.LinearOperators.Filter.Actions.FilterU",
                  "[Unit][NumericalAlgorithms][LinearOperators][Actions]") {
  test_filter_u();
}
