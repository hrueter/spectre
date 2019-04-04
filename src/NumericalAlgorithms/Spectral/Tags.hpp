// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Options/Options.hpp"

namespace OptionTags {
/// \ingroup OptionTagsGroup
///
/// \brief Paramater for the exponential filter
/// The n-th spectral coefficient is filtered by a factor
/// \f$\exp(-\alpha * (n/(N-1))^s)\f$
/// This option specifies \f$\alpha\f$.
/// The default value is 36.0 and the value must be larger than 0.
struct ExpFilterAlpha {
  using type = double;
  static constexpr OptionString help = {
      "Factor alpha in the exponential filter."};
  static type default_value() noexcept { return 36.0; }
  static type lower_bound() noexcept { return 0.0; }
};

/// \ingroup OptionTagsGroup
///
/// \brief Paramater for the exponential filter
/// The n-th spectral coefficient is filtered by a factor
/// \f$\exp(-\alpha * (n/(N-1))^s)\f$
/// This option specifies the integer \f$s\f$.
/// The default value is 64 and the value must be larger than 1.
struct ExpFilterS {
  using type = size_t;
  static constexpr OptionString help = {
      "Exponent s in the exponential filter."};
  static type default_value() noexcept { return 64; }
  static type lower_bound() noexcept { return 1; }
};
}  // namespace OptionTags
