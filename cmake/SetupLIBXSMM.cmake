# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(LIBXSMM 1.16.1)

if (NOT LIBXSMM_FOUND)
  if (NOT SPECTRE_FETCH_MISSING_DEPS)
    message(FATAL_ERROR "Could not find LIBXSMM. If you want to fetch "
      "missing dependencies automatically, set SPECTRE_FETCH_MISSING_DEPS=ON.")
  endif()
  message(STATUS "Fetching LIBXSMM")

  # This FetchContent code is adapted from the libxsmm docs:
  # https://libxsmm.readthedocs.io/en/latest/#rules-for-building-libxsmm
  include(FetchContent)
  # Need an unreleased version to be compatible with newer glibc versions
  FetchContent_Declare(xsmm
    GIT_REPOSITORY https://github.com/libxsmm/libxsmm.git
    GIT_TAG 10b7dc82b3c46157e76eb40e4e959555f895b24d
    ${SPECTRE_FETCHCONTENT_BASE_ARGS}
  )
  FetchContent_MakeAvailable(xsmm)

  set(LIBXSMMROOT ${xsmm_SOURCE_DIR})
  file(GLOB _GLOB_XSMM_SRCS LIST_DIRECTORIES false CONFIGURE_DEPENDS ${LIBXSMMROOT}/src/*.c)
  list(REMOVE_ITEM _GLOB_XSMM_SRCS ${LIBXSMMROOT}/src/libxsmm_generator_gemm_driver.c)
  set(XSMM_INCLUDE_DIRS ${LIBXSMMROOT}/include)

  target_include_directories(xsmm SYSTEM PUBLIC ${XSMM_INCLUDE_DIRS})
  target_compile_definitions(xsmm PUBLIC LIBXSMM_DEFAULT_CONFIG)

  # Link BLAS
  find_package(BLAS REQUIRED)
  target_link_libraries(xsmm PUBLIC BLAS::BLAS)

  # Provide `Libxsmm` target
  add_library(Libxsmm ALIAS xsmm)
endif()

set_property(
  GLOBAL APPEND PROPERTY SPECTRE_THIRD_PARTY_LIBS
  Libxsmm
  )
