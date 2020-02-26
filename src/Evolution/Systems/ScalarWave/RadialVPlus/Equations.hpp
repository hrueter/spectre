// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/FaceNormal.hpp"
#include "Evolution/Systems/ScalarWave/RadialVPlus/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Options.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <typename>
class Variables;

class DataVector;

namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl

namespace Tags {
template <typename>
struct NormalDotFlux;
template <typename>
struct Normalized;
}  // namespace Tags

namespace PUP {
class er;
}  // namespace PUP
/// \endcond

// IWYU pragma: no_forward_declare Tensor

namespace ScalarWave {
namespace RadialVPlus {
/*!
 * \brief Compute the time derivative of the evolved variables of the
 * first-order ScalarWave::RadialVPlus system.
 *
 * The evolution equations for the first-order scalar wave system are given by:
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
 *
 * where \f$\Phi = \chi \Psi\f$ is the rescaled scalar field,
 * \f$\Phi^{+} = \chi^2 (\partial_t \Psi + \hat x^i \partial_i \Psi)
 * + \sigma \chi \Psi\f$ is the rescaled outgoing characteristic, and
 * \f$\Lambda_i=\chi \partial_i \Psi\f$ is the rescaled reduction variable.
 */
template <size_t Dim>
struct ComputeDuDt {
  template <template <class> class StepPrefix>
  using return_tags =
      tmpl::list<db::add_tag_prefix<StepPrefix, Tags::Phi>,
                 db::add_tag_prefix<StepPrefix, Tags::PhiPlus>,
                 db::add_tag_prefix<StepPrefix, Tags::Lambda<Dim>>>;

  using argument_tags = tmpl::list<
      Tags::Phi, Tags::PhiPlus, Tags::Lambda<Dim>,
      ::Tags::deriv<Tags::Phi, tmpl::size_t<Dim>, Frame::Inertial>,
      ::Tags::deriv<Tags::PhiPlus, tmpl::size_t<Dim>, Frame::Inertial>,
      ::Tags::deriv<Tags::Lambda<Dim>, tmpl::size_t<Dim>, Frame::Inertial>,
      Tags::ConstraintGamma2, Tags::RescalingChi,
      ::Tags::deriv<Tags::RescalingChi, tmpl::size_t<Dim>, Frame::Inertial>,
      Tags::RescalingSigma,
      ::Tags::deriv<Tags::RescalingSigma, tmpl::size_t<Dim>, Frame::Inertial>,
      Tags::Radius, Tags::RadialNormal<Dim>>;
  static void apply(
      gsl::not_null<Scalar<DataVector>*> dt_phi,
      gsl::not_null<Scalar<DataVector>*> dt_phiplus,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*> dt_lambda,
      const Scalar<DataVector>& phi, const Scalar<DataVector>& phiplus,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& lambda,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& d_phi,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& d_phiplus,
      const tnsr::ij<DataVector, Dim, Frame::Inertial>& d_lambda,
      const Scalar<DataVector>& gamma2, const Scalar<DataVector>& chi,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& d_chi,
      const Scalar<DataVector>& sigma,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& d_sigma,
      const Scalar<DataVector>& radius,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& xhat) noexcept;
};

/*!
 * \brief Compute the normal component of the flux on a boundary.
 *
 * \f{align}
 * n_k F^k(\Phi)     =&  0 \\
 * n_k F^k(\Phi^{+}) =&  -\sigma \hat x^i n_i \Phi
 *                       + \hat x^i n_i \Phi^{+}
 *                       + \chi q^{ij} n_i \Lambda_j \\
 * n_k F^k(\Lambda_i) =& - \frac{\sigma}{\chi} n_i \Phi
 *                       + \frac{1}{\chi} n_i \Phi^{+}
 *                       - \hat x^j n_i \Lambda_j
 *                       + \gamma_2 n_i \Phi
 * \f}
 */
template <size_t Dim>
struct ComputeNormalDotFluxes {
  using argument_tags =
      tmpl::list<Tags::Phi, Tags::PhiPlus, Tags::Lambda<Dim>,
                 Tags::ConstraintGamma2, Tags::RescalingChi,
                 Tags::RescalingSigma, Tags::RadialNormal<Dim>,
                 ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>>;
  static void apply(gsl::not_null<Scalar<DataVector>*> phi_normal_dot_flux,
                    gsl::not_null<Scalar<DataVector>*> phiplus_normal_dot_flux,
                    gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
                        lambda_normal_dot_flux,
                    const Scalar<DataVector>& phi,
                    const Scalar<DataVector>& phiplus,
                    const tnsr::i<DataVector, Dim, Frame::Inertial>& lambda,
                    const Scalar<DataVector>& gamma2,
                    const Scalar<DataVector>& chi,
                    const Scalar<DataVector>& sigma,
                    const tnsr::i<DataVector, Dim, Frame::Inertial>& xhat,
                    const tnsr::i<DataVector, Dim, Frame::Inertial>&
                        interface_unit_normal) noexcept;
};

/*!
 * \ingroup NumericalFluxesGroup
 * \brief Compute the penalty flux for the ScalarWave::RadialVPlus system
 *
 * The penalty flux is given by:
 *
 * \f{align}
 * G(\Phi)      &= n_k F^k(\Phi) = 0 \\
 * G(\Phi^{+})  &= (n_k F^k(\Phi^{+}))_{\mathrm{int}}
 *                 + \frac{1}{2} p
 *                     \left( v^{-}_{\mathrm{int}}
 *                          - v^{+}_{\mathrm{ext}} \right) \\
 * G(\Lambda_i) &= (n_k F^k(\Lambda_i))_{\mathrm{int}}
 *                  - \frac{1}{2} p
 *                    \left( (n_i v^{-})_{\mathrm{int}}
 *                         + (n_i v^{+})_{\mathrm{ext}}\right)
 * \f}
 *
 * where \f$G\f$ is the interface normal dotted with numerical flux, the
 * first terms on the RHS for \f$G(\Phi^{+})\f$ and \f$G(\Lambda_i)\f$ are the
 * fluxes dotted with the interface normal (computed in
 * ScalarWave::RadialVPlus::ComputeNormalDotFluxes), and \f$v^{\pm}\f$ are
 * outgoing and incoming characteristic fields of the system (see
 * characteristic_fields() for their definition). The
 * penalty factor is chosen to be \f$p=1\f$.
 *
 * ToCheck: Are minus/plus signs correct for this variable definition?
 */
template <size_t Dim>
struct PenaltyFlux {
 private:
  struct NormalTimesVPlus {
    using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
    static std::string name() noexcept { return "NormalTimesVPlus"; }
  };
  struct NormalTimesVMinus {
    using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
    static std::string name() noexcept { return "NormalTimesVMinus"; }
  };

 public:
  using options = tmpl::list<>;
  static constexpr OptionString help = {
      "Computes the penalty flux for a scalar wave system. It requires no "
      "options."};
  static std::string name() noexcept { return "Penalty"; }

  // clang-tidy: non-const reference
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT

  // This is the data needed to compute the numerical flux.
  // `dg::SendBoundaryFluxes` calls `package_data` to store these tags in a
  // Variables. Local and remote values of this data are then combined inside
  // `operator()`.
  using package_tags =
      tmpl::list<::Tags::NormalDotFlux<Tags::PhiPlus>,
                 ::Tags::NormalDotFlux<Tags::Lambda<Dim>>, Tags::VPlus,
                 Tags::VMinus, NormalTimesVPlus, NormalTimesVMinus>;

  // These tags on the interface of the element are passed to
  // `package_data` to provide the data needed to compute the numerical fluxes.
  using argument_tags =
      tmpl::list<::Tags::NormalDotFlux<Tags::PhiPlus>,
                 ::Tags::NormalDotFlux<Tags::Lambda<Dim>>, Tags::VPlus,
                 Tags::VMinus,
                 ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>>;

  // pseudo-interface: used internally by Algorithm infrastructure, not
  // user-level code
  // Following the not-null pointer to packaged_data, this function expects as
  // arguments the databox types of the `argument_tags`.
  void package_data(
      gsl::not_null<Variables<package_tags>*> packaged_data,
      const Scalar<DataVector>& normal_dot_flux_phiplus,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_dot_flux_lambda,
      const Scalar<DataVector>& v_plus, const Scalar<DataVector>& v_minus,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& interface_unit_normal)
      const noexcept;

  // pseudo-interface: used internally by Algorithm infrastructure, not
  // user-level code
  // The first three arguments are pointers to Tags::NormalDotNumericalFlux<...>
  // for each variable in the system, then the package_tags on the interior side
  // of the mortar followed by the package_tags on the exterior side.
  void operator()(
      gsl::not_null<Scalar<DataVector>*> phi_normal_dot_numerical_flux,
      gsl::not_null<Scalar<DataVector>*> phiplus_normal_dot_numerical_flux,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          lambda_normal_dot_numerical_flux,
      const Scalar<DataVector>& normal_dot_flux_phiplus_interior,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          normal_dot_flux_lambda_interior,
      const Scalar<DataVector>& v_plus_interior,
      const Scalar<DataVector>& v_minus_interior,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          normal_times_v_plus_interior,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          normal_times_v_minus_interior,
      const Scalar<DataVector>& minus_normal_dot_flux_phiplus_exterior,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          minus_normal_dot_flux_lambda_exterior,
      const Scalar<DataVector>& v_plus_exterior,
      const Scalar<DataVector>& v_minus_exterior,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          minus_normal_times_v_plus_exterior,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          minus_normal_times_v_minus_exterior) const noexcept;
};

/*!
 * \ingroup NumericalFluxesGroup
 * \brief Compute the upwind flux
 *
 * The upwind flux is given by:
 * \f{align}
 * G(\Phi)      =& 0 \\
 * G(\Phi^{+})  =& \frac{1}{2}\left( n_k F^k(\Phi^{+})_{\mathrm{int}}
 *                                 + n_k F^k(\Phi^{+})_{\mathrm{ext}}
 *                                 + \Phi^{+}_{\mathrm{int}}
 *                                 - \Phi^{+}_{\mathrm{ext}}
 *                             \right) \\
 * G(\Lambda_i) =& \frac{1}{2}
 *                   \left( n_k F^k(\Lambda_i)_{\mathrm{int}}
 *                        + n_k F^k(\Lambda_i)_{\mathrm{ext}}
 *                        + (n_i)_{\mathrm{int}} F(\Phi^{+})_{\mathrm{int}}
 *                        - (n_i)_{\mathrm{ext}} F(\Phi^{+})_{\mathrm{ext}}
 *                   \right)
 *  \f}
 * where \f$G\f$ is the normal dotted with the numerical flux and \f$F\f$ is the
 * normal dotted with the flux, which is computed in
 * ScalarWave::RadialVPlus::ComputeNormalDotFluxes.
 *
 * ToCheck: Check definition of upwind flux
 */
template <size_t Dim>
struct UpwindFlux {
 private:
  struct NormalTimesFluxPhiPlus {
    using type = tnsr::i<DataVector, Dim, Frame::Inertial>;
    static std::string name() noexcept { return "NormalTimesFluxPhiPlus"; }
  };

 public:
  using options = tmpl::list<>;
  static constexpr OptionString help = {
      "Computes the upwind flux for a scalar wave system. It requires no "
      "options."};
  static std::string name() noexcept { return "Upwind"; }

  // clang-tidy: non-const reference
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT

  // This is the data needed to compute the numerical flux.
  // `dg::SendBoundaryFluxes` calls `package_data` to store these tags in a
  // Variables. Local and remote values of this data are then combined in the
  // `()` operator.
  using package_tags = tmpl::list<::Tags::NormalDotFlux<Tags::PhiPlus>,
                                  ::Tags::NormalDotFlux<Tags::Lambda<Dim>>,
                                  Tags::PhiPlus, NormalTimesFluxPhiPlus>;

  // These tags on the interface of the element are passed to
  // `package_data` to provide the data needed to compute the numerical fluxes.
  using argument_tags =
      tmpl::list<::Tags::NormalDotFlux<Tags::PhiPlus>,
                 ::Tags::NormalDotFlux<Tags::Lambda<Dim>>, Tags::PhiPlus,
                 ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>>;

  // pseudo-interface: used internally by Algorithm infrastructure, not
  // user-level code
  // Following the not-null pointer to packaged_data, this function expects as
  // arguments the databox types of the `argument_tags`.
  void package_data(
      gsl::not_null<Variables<package_tags>*> packaged_data,
      const Scalar<DataVector>& normal_dot_flux_phiplus,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_dot_flux_lambda,
      const Scalar<DataVector>& phiplus,
      const tnsr::i<DataVector, Dim, Frame::Inertial>& interface_unit_normal)
      const noexcept;

  // pseudo-interface: used internally by Algorithm infrastructure, not
  // user-level code
  // The arguments are first the system::variables_tag::tags_list wrapped in
  // Tags::NormalDotNumericalFLux as not-null pointers to write the results
  // into, then the package_tags on the interior side of the mortar followed by
  // the package_tags on the exterior side.
  void operator()(
      gsl::not_null<Scalar<DataVector>*> phi_normal_dot_numerical_flux,
      gsl::not_null<Scalar<DataVector>*> phiplus_normal_dot_numerical_flux,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          lambda_normal_dot_numerical_flux,
      const Scalar<DataVector>& normal_dot_flux_phiplus_interior,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          normal_dot_flux_lambda_interior,
      const Scalar<DataVector>& phiplus_interior,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          normal_times_flux_phiplus_interior,
      const Scalar<DataVector>& minus_normal_dot_flux_phiplus_exterior,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          minus_normal_dot_flux_lambda_exterior,
      const Scalar<DataVector>& phiplus_exterior,
      const tnsr::i<DataVector, Dim, Frame::Inertial>&
          normal_times_flux_phiplus_exterior) const noexcept;
};

/// Compute the maximum magnitude of the characteristic speeds.
struct ComputeLargestCharacteristicSpeed {
  using argument_tags = tmpl::list<>;
  SPECTRE_ALWAYS_INLINE static constexpr double apply() noexcept { return 1.0; }
};
}  // namespace RadialVPlus
}  // namespace ScalarWave
