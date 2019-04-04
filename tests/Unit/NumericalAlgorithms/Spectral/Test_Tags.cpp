// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "NumericalAlgorithms/Spectral/Tags.hpp"

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.Spectral.Tags",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  CHECK(OptionTags::ExpFilterAlpha::name() == "ExpFilterAlpha");
  CHECK(OptionTags::ExpFilterS::name() == "ExpFilterS");
}
