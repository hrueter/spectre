// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Domain/Mesh.hpp"
#include "NumericalAlgorithms/LinearOperators/Actions/FilterU.hpp"
#include "NumericalAlgorithms/LinearOperators/CoefficientTransforms.hpp"
#include "NumericalAlgorithms/LinearOperators/Filter.hpp"
#include "NumericalAlgorithms/Spectral/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"

namespace {
struct Var : db::SimpleTag {
  static std::string name() noexcept { return "Var"; }
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct System {
  using variables_tag = Tags::Variables<tmpl::list<Var>>;
  static constexpr size_t volume_dim = Dim;
};

using variables_tag = Tags::Variables<tmpl::list<Var>>;

template <size_t Dim, typename Metavariables>
struct component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementIndex<Dim>;
  using const_global_cache_tag_list = tmpl::list<>;
  using action_list = tmpl::list<Actions::FilterU>;
  using simple_tags =
      db::AddSimpleTags<Tags::Mesh<Dim>, Tags::Element<Dim>, variables_tag>;
  using initial_databox = db::compute_databox_type<simple_tags>;
};

template <size_t Dim>
struct Metavariables {
  using system = System<Dim>;
  using component_list = tmpl::list<component<Dim, Metavariables>>;
  using const_global_cache_tag_list = tmpl::list<>;
};
}  // namespace

void test_filter_u() {
  constexpr double alpha = 15.0;
  constexpr size_t s = 40;

  constexpr size_t dim = 2;
  constexpr size_t order = 5;

  using VarType = db::item_type<variables_tag>;

  using metavariables = Metavariables<dim>;
  using my_component = component<dim, metavariables>;
  const Mesh<dim> mesh{order, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};

  const ElementId<dim> self_id{0};
  const Element<dim> element(self_id, {});

  auto data_before = DataVector(mesh.number_of_grid_points(), 1.);
  const ModalVector modal_vector_coefficients_all_one{
      mesh.number_of_grid_points(), 1.0};
  data_before = to_nodal_coefficients(modal_vector_coefficients_all_one, mesh);
  const VarType var_before(data_before);

  auto start_box = [&mesh, &element, &var_before ]() noexcept {
    // auto map = ElementMap<2, Frame::Inertial>(self_id,
    // coordmap->get_clone());
    auto var = var_before;
    return db::create<my_component::simple_tags>(mesh, element, std::move(var));
  }
  ();

  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavariables>;
  using MockDistributedObjectsTag =
      MockRuntimeSystem::MockDistributedObjectsTag<my_component>;
  MockRuntimeSystem::TupleOfMockDistributedObjects dist_objects{};
  tuples::get<MockDistributedObjectsTag>(dist_objects)
      .emplace(self_id, std::move(start_box));

  ActionTesting::MockRuntimeSystem<metavariables> runner{
      {}, std::move(dist_objects)};

  const VarType& var_to_filter = db::get<variables_tag>(start_box);

  filter_cache_initialize(mesh, alpha, s);

  CHECK(var_to_filter == var_before);

  runner.template next_action<my_component>(self_id);

  CHECK(var_to_filter == filter(var_before, mesh, alpha, s));
}

SPECTRE_TEST_CASE("Unit.Numerical.LinearOperators.Filter.Actions.FilterU",
                  "[Unit][NumericalAlgorithms][LinearOperators][Actions]") {
  test_filter_u();
}
