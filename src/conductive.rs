//! [BSDF] that resembles the appearance of metals
use crate::{
    ggx::GGX,
    utils::{self, SafeCast},
    RgbD, RgbF, SampleIncomingResponse, Vec2d, Vec3d, BSDF,
};

/// [BSDF] that resembles the appearance of metals
#[derive(Clone, Copy, Debug)]
pub struct Conductive {
    /// The `F0` color of the metal
    pub color: RgbF,

    /// The scattering parameters, contains the directional roughness parameters
    pub ggx: GGX,

    /// The Index of refraction. Values in the Range `[1.4, 1.7]` are what you usually want.
    pub ior: f32,
}

impl Conductive {
    const fn ggx(&self) -> GGX {
        self.ggx
    }
    fn bsdf(&self, omega_i: Vec3d, omega_o: Vec3d) -> RgbD {
        let m_opt = utils::positive_reflect_normal(omega_i, omega_o);
        if !GGX::g2_d_satisfied_opt(omega_i, omega_o, m_opt) {
            return RgbD::ZERO;
        }
        let m = m_opt.unwrap();
        let fresnel = self.colored_fresnel(omega_i, m, 1.0, f64::from(self.ior));
        let ggx = self.ggx();
        let masking_shadowing = ggx.geometric(omega_i, omega_o, m);
        let ndf = ggx.ndf(m);

        fresnel * masking_shadowing * ndf / (4.0 * omega_i.z * omega_o.z) // * omega_i.z;//vec3::dot(omega_i, m);
    }

    fn colored_fresnel(&self, omega_i: Vec3d, m: Vec3d, ior_i: f64, ior_t: f64) -> RgbD {
        utils::fresnel_colored(omega_i, m, ior_i, ior_t, self.color.safe_cast(), RgbD::ONE)
    }

    fn pdf_vndf(&self, omega_o: Vec3d, m: Vec3d) -> f64 {
        utils::sample_ggx_vndf_reflection_lobe_pdf(self.ggx(), omega_o, m)
    }

    fn path_trace_sample(&self, omega_o: Vec3d, rand: Vec3d) -> SampleIncomingResponse {
        let (omega_i, pdf) =
            utils::sample_ggx_vndf_reflection_lobe(self.ggx(), omega_o, Vec2d::new(rand.x, rand.y));

        let bsdf = self.bsdf(omega_i, omega_o);
        SampleIncomingResponse {
            omega_i,
            bsdf,
            emission: RgbD::ZERO,
            pdf,
        }
    }

    fn path_trace_bsdf(&self, omega_i: Vec3d, omega_o: Vec3d) -> (RgbD, RgbD, f64) {
        // allow opposite sides, so that we do not lie about the pdf
        let m_opt = utils::positive_reflect_normal_allow_opposite_sides(omega_i, omega_o);
        let Some(m) = m_opt else {
            return (RgbD::ZERO, RgbD::ZERO, 0.0);
        };

        let pdf = self.pdf_vndf(omega_o, m);
        if !GGX::g2_d_satisfied(omega_i, omega_o, m) {
            return (RgbD::ZERO, RgbD::ZERO, pdf);
        }
        let bsdf = self.bsdf(omega_i, omega_o);

        (bsdf, RgbD::ZERO, pdf)
    }
}

impl BSDF for Conductive {
    fn sample_incoming(&self, omega_o: Vec3d, rdf: Vec3d) -> SampleIncomingResponse {
        assert!(omega_o.is_normalized());
        self.path_trace_sample(omega_o, rdf)
    }

    fn evaluate(&self, omega_o: Vec3d, omega_i: Vec3d) -> RgbD {
        assert!(omega_o.is_normalized() && omega_i.is_normalized());
        self.path_trace_bsdf(omega_i, omega_o).0
    }

    fn sample_incoming_pdf(&self, omega_o: Vec3d, omega_i: Vec3d) -> f64 {
        assert!(omega_o.is_normalized() && omega_i.is_normalized());
        self.path_trace_bsdf(omega_i, omega_o).2
    }

    fn base_color(&self, _omega_o: Vec3d) -> RgbD {
        self.color.safe_cast()
    }
}

#[cfg(test)]
mod tests {
    use crate::{ggx::GGX, test_utils, utils::FloatExt, RgbF};

    use super::Conductive;

    const CONDUCTIVE_ROUGH: Conductive = Conductive {
        color: RgbF::ONE,
        ior: 1.4,
        ggx: GGX {
            alpha_x: 0.3 * 0.3,
            alpha_y: 0.4 * 0.4,
        },
    };

    const CONDUCTIVE_SMOOTH: Conductive = Conductive {
        color: RgbF::ONE,
        ior: 1.4,
        ggx: GGX {
            alpha_x: 0.01 * 0.01,
            alpha_y: 0.01 * 0.01,
        },
    };

    #[test]
    fn sample_eval() {
        test_utils::test_bsdf_sample_eval(&CONDUCTIVE_ROUGH);
        test_utils::test_bsdf_sample_eval_adjoint(&CONDUCTIVE_ROUGH);
    }

    #[test]
    fn reciprocity() {
        test_utils::test_bsdf_reciprocity(&CONDUCTIVE_ROUGH);
    }

    #[test]
    fn pdf_integral() {
        test_utils::test_integrate_inverse_pdf(&CONDUCTIVE_ROUGH);
    }

    #[test]
    fn white_furnace() {
        test_utils::test_white_furnace(&CONDUCTIVE_SMOOTH, 0.02);
    }
    #[test]
    fn white_furnace_adjoint() {
        test_utils::test_white_furnace_adjoint(&CONDUCTIVE_SMOOTH, 0.02);
    }
}
