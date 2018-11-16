// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/Index.hpp"

#include <pup.h>  // IWYU pragma: keep

#include "ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/StdHelpers.hpp"  // IWYU pragma: keep

/// \cond HIDDEN_SYMBOLS
template <size_t Dim>
void Index<Dim>::pup(PUP::er& p) noexcept {
  p | indices_;
}

template <size_t N>
size_t collapsed_index(const Index<N>& index,
                       const Index<N>& extents) noexcept {
  size_t result = 0;
  // note: size_t(-1) == std::numeric_limits<size_t>::max()
  for (size_t i = N - 1; i < N; i--) {
    ASSERT(index[i] < extents[i], "Index out of range.");
    result = index[i] + extents[i] * result;
  }
  return result;
}

template <size_t N>
std::ostream& operator<<(std::ostream& os, const Index<N>& i) {
  return os << i.indices_;
}

template <size_t Dim>
bool operator==(const Index<Dim>& lhs, const Index<Dim>& rhs) noexcept {
  return std::equal(lhs.begin(), lhs.end(), rhs.begin());
}

template <size_t Dim>
bool operator!=(const Index<Dim>& lhs, const Index<Dim>& rhs) noexcept {
  return not(lhs == rhs);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define GEN_OP(op, dim)                            \
  template bool operator op(const Index<dim>& lhs, \
                            const Index<dim>& rhs) noexcept;
#define INSTANTIATE(_, data)                                                 \
  template class Index<DIM(data)>;                                           \
  GEN_OP(==, DIM(data))                                                      \
  GEN_OP(!=, DIM(data))                                                      \
  template size_t collapsed_index(const Index<DIM(data)>& index,             \
                                  const Index<DIM(data)>& extents) noexcept; \
  template std::ostream& operator<<(std::ostream& os,                        \
                                    const Index<DIM(data)>& i);

GENERATE_INSTANTIATIONS(INSTANTIATE, (0, 1, 2, 3, 4))

#undef DIM
#undef GEN_OP
#undef INSTANTIATE
/// \endcond
