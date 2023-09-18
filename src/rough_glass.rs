//! [BSDF] that resembles the appearance of glass. Roughness can be adjusted
use crate::{
    ggx::GGX,
    utils::{self, FloatExt},
    RgbD, SampleIncomingResponse, Vec3d, BSDF,
};

struct BsdfPdfResult {
    bsdf: RgbD,
    pdf: f64,
}

/// A [BSDF] that resembles the appearance of Glass
///
/// **NOTE: the `rough_glass` feature must be enabled to use this code**
pub struct RoughGlass {
    /// directional roughness values
    pub ggx: GGX,

    /// index of refraction. Values in the Range `[1.4, 1.7]` are what you usually want.
    pub ior: f64,
}

impl RoughGlass {
    /// return `ior_above` if omega points above the surface (z=0 - Plane), otherwis`ior_below`ow
    fn get_ior_of(omega: Vec3d, ior_above: f64, ior_below: f64) -> f64 {
        if omega.z > 0.0 {
            ior_above
        } else {
            ior_below
        }
    }

    fn eval_get_pdf_reflect(&self, omega_i: Vec3d, omega_o: Vec3d) -> BsdfPdfResult {
        let m_opt = (omega_i + omega_o).try_normalize();

        let m = match m_opt {
            None => {
                return BsdfPdfResult {
                    bsdf: RgbD::ZERO,
                    pdf: 0.,
                };
            }
            Some(m) => m * omega_o.z.signum(),
        };

        let ior_o = Self::get_ior_of(omega_o, 1.0, self.ior);
        let ior_t = Self::get_ior_of(-omega_o, 1.0, self.ior);

        let f = utils::fresnel(omega_o, m, ior_o, ior_t);

        let pdf = f * utils::sample_ggx_vndf_reflection_lobe_pdf(
            self.ggx,
            omega_o * omega_o.z.signum(),
            m,
        );

        if !GGX::g2_d_satisfied(omega_i, omega_o, m) || omega_o.z * omega_i.z < 0.0 {
            return BsdfPdfResult {
                bsdf: RgbD::ZERO,
                pdf,
            };
        }

        let g = self.ggx.geometric(omega_i, omega_o, m);
        let d = self.ggx.ndf(m);

        let result = g * d * f / (4.0 * omega_i.z * omega_o.z);

        BsdfPdfResult {
            bsdf: RgbD::splat(result),
            pdf,
        }
    }

    fn eval_get_pdf_refract(&self, omega_i: Vec3d, omega_o: Vec3d) -> BsdfPdfResult {
        let ior_o = Self::get_ior_of(omega_o, 1.0, self.ior);
        let ior_t = Self::get_ior_of(-omega_o, 1.0, self.ior);

        // can only ever fail if ior = 1
        let m_opt =
            utils::refract_normal(omega_i, omega_o, ior_t, ior_o).and_then(Vec3d::try_normalize);

        let m = match m_opt {
            None => {
                return BsdfPdfResult {
                    bsdf: RgbD::ZERO,
                    pdf: 0.0,
                }
            }
            Some(m) if self.ior > 1.0 => m,
            Some(m) => -m,
        };

        if m.z < 0.0 {
            return BsdfPdfResult {
                bsdf: RgbD::ZERO,
                pdf: 0.0,
            };
        }

        let fresnel = utils::fresnel(omega_o, m, ior_o, ior_t);

        let vndf = self.ggx.vndf(omega_o * omega_o.z.signum(), m);

        #[allow(clippy::suboptimal_flops)]
        let jacobian = ior_t.sq() * omega_i.dot(m).abs()
            / (ior_t * m.dot(omega_i) + ior_o * m.dot(omega_o)).sq();

        let vndf_pdf = (1.0 - fresnel) * vndf * jacobian;

        assert!(vndf_pdf >= 0.0);

        if !GGX::g2_d_satisfied(omega_i, omega_o, m) || omega_i.z * omega_o.z > 0.0 {
            return BsdfPdfResult {
                bsdf: RgbD::ZERO,
                pdf: vndf_pdf,
            };
        }

        let masking_shadowing = self.ggx.geometric(omega_i, omega_o, m);
        let ndf = self.ggx.ndf(m);

        #[allow(clippy::suboptimal_flops)]
        let btdf_p = ior_o.sq() / (ior_t * omega_i.dot(m) + ior_o * omega_o.dot(m)).sq();
        let correction_factors =
            ((omega_i.dot(m) / omega_i.z) * (omega_o.dot(m) / omega_o.z)).abs();

        let radiance_correction = ior_t / ior_o;

        let bsdf = masking_shadowing
            * ndf
            * (1.0 - fresnel)
            * btdf_p
            * correction_factors
            * radiance_correction.sq();

        BsdfPdfResult {
            bsdf: RgbD::splat(bsdf),
            pdf: vndf_pdf,
        }
    }

    fn bsdf_get_pdf(&self, omega_i: Vec3d, omega_o: Vec3d) -> BsdfPdfResult {
        // add the results together because for higher roughness values the pdf lobes overlab
        // (the distributions are'n cut at the xy-plane)
        let reflect = self.eval_get_pdf_reflect(omega_i, omega_o);
        let refract = self.eval_get_pdf_refract(omega_i, omega_o);
        BsdfPdfResult {
            bsdf: reflect.bsdf + refract.bsdf,
            pdf: reflect.pdf + refract.pdf,
        }
    }
}

impl BSDF for RoughGlass {
    fn evaluate(&self, omega_o: Vec3d, omega_i: Vec3d) -> RgbD {
        self.bsdf_get_pdf(omega_i, omega_o).bsdf
    }

    fn sample_incoming(&self, omega_o: Vec3d, rdf: Vec3d) -> SampleIncomingResponse {
        let ior_o = Self::get_ior_of(omega_o, 1.0, self.ior);
        let ior_t = Self::get_ior_of(-omega_o, 1.0, self.ior);
        let m = self
            .ggx
            .sample_vndf(omega_o * omega_o.z.signum(), rdf.x, rdf.y);
        let fresnel = utils::fresnel(omega_o, m, ior_o, ior_t);
        let omega_i = if rdf.z <= fresnel {
            utils::reflect(m, omega_o)
        } else {
            // unwrap is ok, because fresnel will be 1.0 if total internal refraction happens
            utils::refract_good(omega_o, m, ior_o, ior_t)
                .unwrap()
                .normalize()
        };
        let BsdfPdfResult { bsdf, pdf } = self.bsdf_get_pdf(omega_i, omega_o);

        SampleIncomingResponse {
            omega_i,
            bsdf,
            emission: RgbD::ZERO,
            pdf,
        }
    }

    fn sample_incoming_pdf(&self, omega_o: Vec3d, omega_i: Vec3d) -> f64 {
        self.bsdf_get_pdf(omega_i, omega_o).pdf
    }

    fn base_color(&self, _omega_o: Vec3d) -> RgbD {
        RgbD::ZERO
    }
}

#[cfg(test)]
impl crate::core::TransmissiveBsdf for RoughGlass {
    fn ior(&self) -> f64 {
        self.ior
    }
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod tests {

    use crate::{ggx::GGX, test_utils, utils::FloatExt};

    use super::RoughGlass;

    #[test]
    fn rough_glass() {
        let mat = RoughGlass {
            ior: 1.4,
            ggx: GGX {
                alpha_x: 0.3.sq(),
                alpha_y: 0.4.sq(),
            },
        };
        test_utils::test_bsdf_sample_eval(&mat);
        test_utils::test_bsdf_reciprocity_glass(&mat);
    }

    #[test]
    fn pdf_integral() {
        let mat = RoughGlass {
            ior: 1.4,
            ggx: GGX {
                alpha_x: 0.3.sq(),
                alpha_y: 0.4.sq(),
            },
        };
        test_utils::test_integrate_inverse_pdf(&mat);
    }

    #[test]
    fn energy_conservation() {
        let mat = RoughGlass {
            ior: 1.4,
            ggx: GGX {
                alpha_x: 0.05.sq(),
                alpha_y: 0.05.sq(),
            },
        };
        test_utils::test_energy_conservation(&mat, 0.05);
    }
}
