// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/TypeTraits.hpp"

class DataVector;
class ModalVector;

// IWYU pragma: no_forward_declare Tensor

namespace {
struct ScalarTag : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "Scalar"; }
};

template <size_t Dim>
struct VectorTag : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim>;
  static std::string name() noexcept {
    return "I<" + std::to_string(Dim) + ">";
  }
};

void test_modal_tag() noexcept {
  static_assert(cpp17::is_same_v<typename Tags::Modal<ScalarTag>::type,
                                 Scalar<ModalVector>>,
                "Failed testing tags::ModalTag");
  static_assert(cpp17::is_same_v<typename Tags::Modal<VectorTag<1>>::type,
                                 tnsr::I<ModalVector, 1>>,
                "Failed testing tags::ModalTag");
  static_assert(cpp17::is_same_v<typename Tags::Modal<VectorTag<3>>::type,
                                 tnsr::I<ModalVector, 3>>,
                "Failed testing tags::ModalTag");
  CHECK(Tags::Modal<ScalarTag>::name() == "Modal(Scalar)");
  CHECK(Tags::Modal<VectorTag<1>>::name() == "Modal(I<1>)");
  CHECK(Tags::Modal<VectorTag<3>>::name() == "Modal(I<3>)");
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tags", "[Unit][DataStructures]") {
  test_modal_tag();
}