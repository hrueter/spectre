# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
    CoordinateMaps
    DiscontinuousGalerkin
    DomainCreators
    DualFoliationScalarWave
    IO
    Informer
    LinearOperators
    MathFunctions
    ScalarWave
    Time
    Utilities
    WaveEquation
    ${SPECTRE_LIBRARIES}
  )

# Commented out, because RegularSphericalWave solution only exists in 3D
# and cartoon is not yet ready
#add_spectre_parallel_executable(
#  EvolveDualFoliationScalarWave1D
#  EvolveDualFoliationScalarWave
#  Evolution/Executables/DualFoliationScalarWave
#  EvolutionMetavars<1>
#  "${LIBS_TO_LINK}"
#)
#add_spectre_parallel_executable(
#  EvolveDualFoliationScalarWave2D
#  EvolveDualFoliationScalarWave
#  Evolution/Executables/DualFoliationScalarWave
#  EvolutionMetavars<2>
#  "${LIBS_TO_LINK}"
#)
add_spectre_parallel_executable(
  EvolveDualFoliationScalarWave3D
  EvolveDualFoliationScalarWave
  Evolution/Executables/DualFoliationScalarWave
  EvolutionMetavars<3>
  "${LIBS_TO_LINK}"
)
