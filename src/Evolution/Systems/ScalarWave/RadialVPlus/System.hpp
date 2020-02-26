// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class ScalarWaveSystem.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Systems/ScalarWave/RadialVPlus/Characteristics.hpp"
#include "Evolution/Systems/ScalarWave/RadialVPlus/Equations.hpp"
#include "Evolution/Systems/ScalarWave/RadialVPlus/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarWave {
namespace RadialVPlus {
/*!
 * \ingroup EvolutionSystemsGroup
 * \brief Items related to evolving the scalar wave equation.
 *
 * This system evolves the scalar wave equation, \f$\square\Psi=f\f$, in flat
 * space. The evolved variables are adapted for an evolution on an
 * asympotitically hyperboloidal slice. The particular choice of variables is
 * motivated by the reasoning in \cite Gasperin2019rjg.
 *
 * The definition of the evolved variables are given in terms of the usual wave
 * field \f$\Psi\f$ as following:
 * \f{align*}
 * \Phi      =& \chi \Psi \\
 * \Phi^{+}  =& \chi^2 (\partial_t \Psi + \hat x^i \partial_i \Psi)
 *                + \sigma \chi \Psi \\
 * \Lambda_i =& \chi \partial_i \Psi
 * \f}
 * where \f$\hat x^i = x^i/r\f$ is the outward pointing radial normal
 * vector. \f$\chi\f$ is a rescaling factor depending only on the radius \f$r\f$
 * and approaching \f$r\f$ asymptotically, \f$ \lim_{r\rightarrow\infty} |\chi -
 * r| = 0 \f$ . The factor \f$\sigma\f$ likewise only depends on \f$r\f$ and has
 * to approach a constant value asymptotically. If \f$\sigma\f$ asymptotes to
 * \f$1\f$ then the asymptotic system has particularly easy form.
 *
 * The rescaling by \f$\chi\f$ has been chosen such, that all fields have a
 * finite, potentially non-zero, limit at null infinity. The choice of
 * \f$\Phi^{+}\f$ is motivated by \f$(\partial_t \Psi + \hat x^i \partial_i
 * \Psi)\f$ falling off one power of \f$r\f$ faster than other derivatives
 * of \f$\Psi\f$ .
 *
 * The above variable definitions lead to the following evolution system:
 * \f{align*}
 * \partial_t \Phi     =& \frac{1}{\chi} \Phi^{+}
 *                         - \frac{\sigma}{\chi} \Phi
 *                         - \hat x^i \Lambda_i \\
 * \partial_t \Phi^{+} =&  - \sigma \hat x^i \partial_i \Phi
 *                         + \hat x^i \partial_i \Phi^{+}
 *                         + \chi q^{ij} \partial_i \Lambda_j
 *                         - \Phi \hat x^i \partial_i \sigma
 *                         + (\Phi^{+} - \sigma \Phi)
 *                              (\frac{\sigma}{\chi}
 *                               - 2 \hat x^i \frac{\partial_i \chi}{\chi})
 *                         + \chi^2 f    \\
 * \partial_t \Lambda_i =& -\frac{\sigma}{\chi} \partial_i \Phi
 *                         + \frac{1}{\chi} \partial_i \Phi^{+}
 *                         - \hat x^j \partial_i \Lambda_j
 *                         - 2 \frac{\partial_i \chi}{\chi^2}
 *                              (\Phi^{+} - \sigma \Phi)
 *                         - \Phi \frac{\partial_i \sigma}{\chi}
 *                         - \frac{1}{r} q^j_i \Lambda_j
 *                         + \hat x^j \Lambda_j \frac{\partial_i \chi}{\chi}
 *                         + \gamma_2
 *                              (\partial_i \Phi
 *                               - \Phi \frac{\partial_i \chi}{\chi}
 *                               - \Lambda_i)
 * \f}
 * where \f$q^i_j = \delta^i_j - \hat x^i \hat x^j\f$ is the projection
 * orthogonal to \f$\hat x^i\f$ and \f$\gamma_2\f$ is the constraint damping
 * parameter.
 *
 * \details The current variable defintions are undfined at the origin r=0.
 */
template <size_t Dim>
struct System {
  static constexpr bool is_in_flux_conservative_form = false;
  static constexpr bool has_primitive_and_conservative_vars = false;
  static constexpr size_t volume_dim = Dim;

  using variables_tag = ::Tags::Variables<
      tmpl::list<Tags::Phi, Tags::PhiPlus, Tags::Lambda<Dim>>>;
  // Typelist of which subset of the variables to take the gradient of.
  using gradients_tags =
      tmpl::list<Tags::Phi, Tags::PhiPlus, Tags::Lambda<Dim>>;

  using normal_dot_fluxes = ComputeNormalDotFluxes<Dim>;
  using compute_largest_characteristic_speed =
      ComputeLargestCharacteristicSpeed;

  using char_speeds_tag = Tags::CharacteristicSpeedsCompute<Dim>;

  template <typename Tag>
  using magnitude_tag = ::Tags::EuclideanMagnitude<Tag>;
};
}  // namespace RadialVPlus
}  // namespace ScalarWave
