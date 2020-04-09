// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "Utilities/TypeTraits.hpp"

namespace tt {
// @{
/// \ingroup TypeTraitsGroup
/// \brief Check if type `T` has operator== defined.
///
/// \details
/// Inherits from std::true_type if the type `T` has operator== defined,
/// otherwise inherits from std::false_type
///
/// \usage
/// For any type `T`,
/// \code
/// using result = tt::has_equivalence<T>;
/// \endcode
///
/// \metareturns
/// std::bool_constant
///
/// \semantics
/// If the type `T` has operator== defined, then
/// \code
/// typename result::type = std::true_type;
/// \endcode
/// otherwise
/// \code
/// typename result::type = std::false_type;
/// \endcode
///
/// \example
/// \snippet Test_HasEquivalence.cpp has_equivalence_example
/// \see has_inequivalence
/// \tparam T the type we want to know if it has operator==
template <typename T, typename = std::void_t<>>
struct has_equivalence : std::false_type {};

/// \cond HIDDEN_SYMBOLS
template <typename T>
struct has_equivalence<
    T, std::void_t<decltype(std::declval<T>() == std::declval<T>())>>
    : std::true_type {};
/// \endcond

/// \see has_equivalence
template <typename T>
constexpr bool has_equivalence_v = has_equivalence<T>::value;

/// \see has_equivalence
template <typename T>
using has_equivalence_t = typename has_equivalence<T>::type;
// @}
}  // namespace tt
