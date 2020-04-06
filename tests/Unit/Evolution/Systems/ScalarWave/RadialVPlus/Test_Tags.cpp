// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/Framework/TestingFramework.hpp"

#include "Evolution/Systems/ScalarWave/RadialVPlus/Tags.hpp"
#include "tests/Unit/Helpers/DataStructures/DataBox/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ScalarWave.RadialVPlus.Tags",
                  "[Unit][Evolution]") {
  TestHelpers::db::test_simple_tag<ScalarWave::RadialVPlus::Tags::Phi>("Phi");
  TestHelpers::db::test_simple_tag<ScalarWave::RadialVPlus::Tags::PhiPlus>(
      "PhiPlus");
  TestHelpers::db::test_simple_tag<ScalarWave::RadialVPlus::Tags::Lambda<3>>(
      "Lambda");
  TestHelpers::db::test_simple_tag<ScalarWave::RadialVPlus::Tags::RescalingChi>(
      "RescalingChi");
  TestHelpers::db::test_simple_tag<
      ScalarWave::RadialVPlus::Tags::RescalingSigma>("RescalingSigma");
  TestHelpers::db::test_simple_tag<ScalarWave::RadialVPlus::Tags::Radius>(
      "Radius");
  TestHelpers::db::test_simple_tag<
      ScalarWave::RadialVPlus::Tags::RadialNormal<3>>("RadialNormal");
  TestHelpers::db::test_simple_tag<
      ScalarWave::RadialVPlus::Tags::ConstraintGamma2>("ConstraintGamma2");
  TestHelpers::db::test_simple_tag<
      ScalarWave::RadialVPlus::Tags::OneIndexConstraint<3>>(
      "OneIndexConstraint");
  TestHelpers::db::test_simple_tag<
      ScalarWave::RadialVPlus::Tags::TwoIndexConstraint<3>>(
      "TwoIndexConstraint");
  TestHelpers::db::test_simple_tag<ScalarWave::RadialVPlus::Tags::VPhi>("VPhi");
  TestHelpers::db::test_simple_tag<ScalarWave::RadialVPlus::Tags::VZero<3>>(
      "VZero");
  TestHelpers::db::test_simple_tag<ScalarWave::RadialVPlus::Tags::VPlus>(
      "VPlus");
  TestHelpers::db::test_simple_tag<ScalarWave::RadialVPlus::Tags::VMinus>(
      "VMinus");
  TestHelpers::db::test_simple_tag<
      ScalarWave::RadialVPlus::Tags::CharacteristicSpeeds<3>>(
      "CharacteristicSpeeds");
  TestHelpers::db::test_simple_tag<
      ScalarWave::RadialVPlus::Tags::CharacteristicFields<3>>(
      "CharacteristicFields");
  TestHelpers::db::test_simple_tag<
      ScalarWave::RadialVPlus::Tags::EvolvedFieldsFromCharacteristicFields<3>>(
      "EvolvedFieldsFromCharacteristicFields");
}
