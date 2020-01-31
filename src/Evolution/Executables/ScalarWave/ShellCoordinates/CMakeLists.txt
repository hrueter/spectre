# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBS_TO_LINK
    CoordinateMaps
    DiscontinuousGalerkin
    DomainCreators
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


add_spectre_parallel_executable(
  EvolveShellCoordinateScalarWave3D
  EvolveShellCoordinateScalarWave
  Evolution/Executables/ScalarWave/ShellCoordinates
  EvolutionMetavars<3>
  "${LIBS_TO_LINK}"
)
