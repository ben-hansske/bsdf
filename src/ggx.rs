//! An implementation of the GGX Distribution

use crate::{utils::FloatExt, Vec3d};
use std::f64::consts;

/// This is a common microsurface model to describe anisotropic rough surfaces. This class is
/// useful for creating own microsurface [`crate::BSDF`]s.
/// It is widely used in [`crate::conductive::Conductive`], [`crate::rough_glass::RoughGlass`] and
/// [`crate::disney::Disney`]
///
/// # Mathematical background
/// * [Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs](https://jcgt.org/published/0003/02/03/)
/// * [Sampling the GGX Distribution of Visible Normals](https://jcgt.org/published/0007/04/01/)
#[derive(Clone, Copy, Debug)]
pub struct GGX {
    /// roughness in direction x
    /// This roughness values are not perceived linearly. The following formula is a good approximation for
    /// choosing roughness values.
    /// alpha_x = perceived_rougness_x^2
    pub alpha_x: f64,

    /// roughness in direction y
    /// This roughness values are not perceived linearly. The following formula is a good approximation for
    /// choosing roughness values.
    /// alpha_y = perceived_rougness_y^2
    pub alpha_y: f64,
}

impl GGX {
    #[must_use]
    pub fn from_remapped(roughness: f64, anisotropic: f64) -> Self {
        let min_alpha: f64 = 0.001;
        let max_aniso: f64 = 0.9;
        let alpha = roughness.sq();
        let alpha = alpha.max(min_alpha);

        #[allow(clippy::suboptimal_flops)]
        let aspect = (1.0 - max_aniso * anisotropic).sqrt();
        Self {
            alpha_x: alpha / aspect,
            alpha_y: alpha * aspect,
        }
    }

    /// Distribution of normals / Normal Distribution Function
    /// This is the $D$ term in typical Cook-Torance / GGX model
    #[must_use]
    pub fn ndf(&self, m: Vec3d) -> f64 {
        if m.z <= 1e-10 {
            return 0.0;
        }
        let denom = consts::PI
            * self.alpha_x
            * self.alpha_y
            * ((m.x / self.alpha_x).sq() + (m.y / self.alpha_y).sq() + (m.z).sq()).sq();
        1.0 / denom
    }

    /// Masking-Shadowing function
    /// This is the $G$ term in typical Cook-Torance / GGX model
    /// Self shadowing on the surface
    #[must_use]
    pub fn geometric(&self, omega_i: Vec3d, omega_o: Vec3d, micro_surface: Vec3d) -> f64 {
        self.shadowing(omega_i, micro_surface) * self.shadowing(omega_o, micro_surface)
    }

    /// shadowing function
    /// This is the `G_1` term in typical Cook-Torance / GGX model
    #[must_use]
    pub fn shadowing(&self, omega: Vec3d, micro_surface: Vec3d) -> f64 {
        Self::g1_local(omega, micro_surface) * self.g1_distant(omega)
    }

    #[must_use]
    fn g1_distant(&self, omega: Vec3d) -> f64 {
        if omega.z.abs() < 1e-10 {
            return 0.0;
        }
        // return 2.0 * dot / (dot + f32::sqrt(alpha * alpha + (1.0 - alpha * alpha) * dot * dot));
        (2.0)
            / ((1.0)
                + f64::sqrt(
                    (1.0)
                        + ((self.alpha_x * omega.x).sq() + (self.alpha_y * omega.y).sq())
                            / (omega.z).sq(),
                ))
    }

    #[must_use]
    fn g1_local(omega: Vec3d, m: Vec3d) -> f64 {
        // return std::max(0.0, dot);
        if omega.dot(m) * omega.z >= (0.0) {
            1.0
        } else {
            0.0
        }
    }

    #[must_use]
    pub fn vndf(&self, omega_o: Vec3d, m: Vec3d) -> f64 {
        self.shadowing(omega_o, m) * omega_o.dot(m).clamp(0.0, 1.0) * self.ndf(m) / omega_o.z.abs()
    }

    #[must_use]
    pub fn sample_vndf(&self, omega_o: Vec3d, r1: f64, r2: f64) -> Vec3d {
        let v_h: Vec3d = Vec3d::new(
            omega_o.x * self.alpha_x,
            omega_o.y * self.alpha_y,
            omega_o.z,
        )
        .normalize();

        #[allow(clippy::suboptimal_flops)]
        let lensq = v_h.x * v_h.x + v_h.y * v_h.y;
        let at1: Vec3d = if lensq > (1.0e-10) {
            Vec3d::new(-v_h.y, v_h.x, 0.0) / f64::sqrt(lensq)
        } else {
            Vec3d::new(1.0, 0.0, 0.0)
        };
        let at2: Vec3d = Vec3d::cross(v_h, at1);

        let r = r1.sqrt();
        let phi: f64 = (2.0) * consts::PI * r2;
        let t1 = r * phi.cos();
        let t2 = r * phi.sin();
        let s: f64 = (0.5) * ((1.0) + v_h.z);

        #[allow(clippy::suboptimal_flops)]
        let t2r = (1.0 - s) * (1.0 - t1 * t1).sqrt() + s * t2;

        #[allow(clippy::suboptimal_flops)]
        let m_h: Vec3d = at1 * t1 + at2 * t2r + v_h * (1.0 - t1 * t1 - t2r * t2r).max(0.0).sqrt();

        Vec3d::new(
            self.alpha_x * m_h.x,
            self.alpha_y * m_h.y,
            f64::max(0.0, m_h.z),
        )
        .normalize()
    }

    #[must_use]
    pub fn g1_local_satisfied(omega: Vec3d, m: Vec3d) -> bool {
        m.dot(omega) * omega.z > 0.0
    }
    fn g2_local_satisfied(omega_i: Vec3d, omega_o: Vec3d, m: Vec3d) -> bool {
        Self::g1_local_satisfied(omega_i, m) && Self::g1_local_satisfied(omega_o, m)
    }
    fn d_satisfied(m: Vec3d) -> bool {
        m.z > 0.0
    }

    #[must_use]
    pub fn g1_d_satisfied(omega: Vec3d, m: Vec3d) -> bool {
        Self::g1_local_satisfied(omega, m) && Self::d_satisfied(m)
    }

    #[must_use]
    pub fn g2_d_satisfied(omega_i: Vec3d, omega_o: Vec3d, m: Vec3d) -> bool {
        Self::g2_local_satisfied(omega_i, omega_o, m) && Self::d_satisfied(m)
    }

    #[must_use]
    pub fn g1_d_satisfied_opt(omega: Vec3d, m_opt: Option<Vec3d>) -> bool {
        m_opt.map_or(false, |m| Self::g1_d_satisfied(omega, m))
    }

    #[must_use]
    pub fn g2_d_satisfied_opt(omega_i: Vec3d, omega_o: Vec3d, m_opt: Option<Vec3d>) -> bool {
        m_opt.map_or(false, |m| Self::g2_d_satisfied(omega_i, omega_o, m))
    }
}
