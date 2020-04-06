# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_ScalarWave_RadialVPlus")

set(LIBRARY_SOURCES
  Test_Characteristics.cpp
#  Test_Constraints.cpp
#  Test_Equations.cpp
#  Test_Tags.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/ScalarWave/RadialVPlus"
  "${LIBRARY_SOURCES}"
  "ScalarWave;MathFunctions;WaveEquation"
  )
