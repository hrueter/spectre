// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarWave/RadialVPlus/Constraints.hpp"

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace ScalarWave {
namespace RadialVPlus {
template <size_t Dim>
tnsr::i<DataVector, Dim, Frame::Inertial> one_index_constraint(
    const Scalar<DataVector>& phi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& d_phi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& lambda,
    const Scalar<DataVector>& chi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& d_chi) noexcept {
  tnsr::i<DataVector, Dim, Frame::Inertial> constraint(
      get_size(get<0>(lambda)));
  one_index_constraint(make_not_null(&constraint), phi, d_phi, lambda, chi,
                       d_chi);
  return constraint;
}

template <size_t Dim>
void one_index_constraint(
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> constraint,
    const Scalar<DataVector>& phi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& d_phi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& lambda,
    const Scalar<DataVector>& chi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& d_chi) noexcept {
  destructive_resize_components(constraint, get_size(get<0>(lambda)));

  for (size_t i = 0; i < Dim; i++) {
    constraint->get(i) =
        d_phi.get(i) - lambda.get(i) - phi.get() * d_chi.get(i) / chi.get();
  }
}

template <size_t Dim>
tnsr::ij<DataVector, Dim, Frame::Inertial> two_index_constraint(
    const tnsr::i<DataVector, Dim, Frame::Inertial>& lambda,
    const tnsr::ij<DataVector, Dim, Frame::Inertial>& d_lambda,
    const Scalar<DataVector>& chi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& d_chi) noexcept {
  tnsr::ij<DataVector, Dim, Frame::Inertial> constraint(
      get_size(get<0, 0>(d_lambda)));
  two_index_constraint(make_not_null(&constraint), lambda, d_lambda, chi,
                       d_chi);
  return constraint;
}

template <size_t Dim>
void two_index_constraint(
    const gsl::not_null<tnsr::ij<DataVector, Dim, Frame::Inertial>*> constraint,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& lambda,
    const tnsr::ij<DataVector, Dim, Frame::Inertial>& d_lambda,
    const Scalar<DataVector>& chi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& d_chi) noexcept {
  destructive_resize_components(constraint, get_size(get<0, 0>(d_lambda)));
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j < Dim; ++j) {
      constraint->get(i, j) = d_lambda.get(i, j) - d_lambda.get(j, i) -
                              lambda.get(j) * d_chi.get(i) / chi.get() +
                              lambda.get(i) * d_chi.get(j) / chi.get();
    }
  }
}
}  // namespace RadialVPlus
}  // namespace ScalarWave

// Explicit Instantiations
/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                  \
  template tnsr::i<DataVector, DIM(data), Frame::Inertial>                    \
  ScalarWave::RadialVPlus::one_index_constraint(                              \
      const Scalar<DataVector>&,                                              \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&,                 \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&,                 \
      const Scalar<DataVector>&,                                              \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&) noexcept;       \
  template void ScalarWave::RadialVPlus::one_index_constraint(                \
      const gsl::not_null<tnsr::i<DataVector, DIM(data), Frame::Inertial>*>,  \
      const Scalar<DataVector>&,                                              \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&,                 \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&,                 \
      const Scalar<DataVector>&,                                              \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&) noexcept;       \
  template tnsr::ij<DataVector, DIM(data), Frame::Inertial>                   \
  ScalarWave::RadialVPlus::two_index_constraint(                              \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&,                 \
      const tnsr::ij<DataVector, DIM(data), Frame::Inertial>&,                \
      const Scalar<DataVector>&,                                              \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&) noexcept;       \
  template void ScalarWave::RadialVPlus::two_index_constraint(                \
      const gsl::not_null<tnsr::ij<DataVector, DIM(data), Frame::Inertial>*>, \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&,                 \
      const tnsr::ij<DataVector, DIM(data), Frame::Inertial>&,                \
      const Scalar<DataVector>&,                                              \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
/// \endcond
