use crate::{ggx::GGX, RgbD, RgbF, Vec2d, Vec3d};

pub trait FloatExt {
    fn sq(self) -> Self;
    fn lerp(self, other: Self, t: Self) -> Self;
}

impl FloatExt for f64 {
    fn sq(self) -> Self {
        self * self
    }
    fn lerp(self, other: Self, t: Self) -> Self {
        #[allow(clippy::suboptimal_flops)]
        {
            self * (1.0 - t) + other * t
        }
    }
}

impl FloatExt for f32 {
    fn sq(self) -> Self {
        self * self
    }

    fn lerp(self, other: Self, t: Self) -> Self {
        #[allow(clippy::suboptimal_flops)]
        {
            self * (1.0 - t) + other * t
        }
    }
}

pub trait SafeCast<Target> {
    fn safe_cast(self) -> Target;
}

impl SafeCast<RgbD> for RgbF {
    fn safe_cast(self) -> RgbD {
        RgbD {
            x: self.x as f64,
            y: self.y as f64,
            z: self.z as f64,
        }
    }
}

pub trait VecExt {
    type Scalar;
    #[must_use]
    fn luminance(self) -> Self::Scalar;
    #[must_use]
    fn sq(self) -> Self;
    #[must_use]
    fn sqrt(self) -> Self;
}

impl VecExt for Vec3d {
    type Scalar = f64;

    fn sq(self) -> Self {
        self * self
    }

    /// Returns the perceived brightness of the color
    fn luminance(self) -> Self::Scalar {
        let lfac = Self::new(0.2162, 0.7152, 0.0722);
        self.dot(lfac)
        //(self.r * self.r + self.g * self.g + self.b * self.b).sqrt()
    }

    fn sqrt(self) -> Self {
        Self {
            x: self.x.sqrt(),
            y: self.y.sqrt(),
            z: self.z.sqrt(),
        }
    }
}

pub fn reflect(n: Vec3d, vec: Vec3d) -> Vec3d {
    n * (n.dot(vec) * 2.0) - vec
}

pub fn positive_reflect_normal(omega_i: Vec3d, omega_o: Vec3d) -> Option<Vec3d> {
    if omega_i.z * omega_o.z <= 0.0 {
        // This can not come from a reflection
        None
    } else {
        ((omega_i + omega_o) * omega_o.z.signum()).try_normalize()
    }
}

pub fn positive_reflect_normal_allow_opposite_sides(
    omega_i: Vec3d,
    omega_o: Vec3d,
) -> Option<Vec3d> {
    let m = (omega_i + omega_o).try_normalize()?;
    Some(m * m.z.signum())
}

pub fn fresnel(omega_i: Vec3d, m: Vec3d, ior_i: f64, ior_t: f64) -> f64
where
{
    let one_half = 0.5;

    let c = omega_i.dot(m).abs();
    let n_rel = ior_t / ior_i;
    let g2 = n_rel.sq() - 1.0 + c.sq();
    if g2 <= 0.0 {
        // total internal reflection
        return 1.0;
    }
    let g = g2.sqrt();
    if g + c == 0.0 {
        return 1.0;
    }
    let f1 = (g - c) / (g + c);

    #[allow(clippy::suboptimal_flops)]
    let f2 = (c * (g + c) - 1.0) / (c * (g - c) + 1.0);
    one_half * f1.sq() * (1.0 + f2.sq())
}

#[must_use]
pub fn fresnel_colored(
    omega_i: Vec3d,
    m: Vec3d,
    ior_i: f64,
    ior_t: f64,
    fd90: RgbD,
    fd0: RgbD,
) -> RgbD {
    let unpolar = fresnel(omega_i, m, ior_i, ior_t);
    fd90.lerp(fd0, unpolar)
}

pub fn sample_ggx_vndf_reflection_lobe_pdf(ggx: GGX, omega_o: Vec3d, m: Vec3d) -> f64 {
    ggx.shadowing(omega_o, m) * ggx.ndf(m) / (4.0 * omega_o.z.abs())
}

pub fn sample_ggx_vndf_reflection_lobe(ggx: GGX, omega_o: Vec3d, rnf: Vec2d) -> (Vec3d, f64) {
    let m = ggx.sample_vndf(omega_o * omega_o.z.signum(), rnf.x, rnf.y);

    let omega_i = reflect(m, omega_o);

    let pdf = sample_ggx_vndf_reflection_lobe_pdf(ggx, omega_o, m);
    // let bsdf = self.bsdf(omega_i, omega_o);
    (omega_i, pdf)
}

pub fn sample_diffuse_lobe_pdf(omega_o: Vec3d, omega_i: Vec3d) -> f64 {
    if omega_i.z * omega_o.z >= 0.0 {
        omega_i.z.abs() / (std::f64::consts::PI)
    } else {
        0.0
    }
}

pub fn sample_diffuse_lobe(omega_o: Vec3d, eps1: f64, eps2: f64) -> (Vec3d, f64) {
    let eps_theta_sample = eps1.clamp(1e-6, 1.0); // prevent division by zero (division by pdf)
    let cos_theta = eps_theta_sample.sqrt() * omega_o.z.signum();
    let sin_theta = (1.0 - eps_theta_sample).sqrt();
    let phi: f64 = (2.0 * std::f64::consts::PI) * eps2;
    let (sin_phi, cos_phi) = phi.sin_cos();
    let omega_i = Vec3d {
        x: sin_theta * sin_phi,
        y: sin_theta * cos_phi,
        z: cos_theta,
    };
    (omega_i, cos_theta.abs() / (std::f64::consts::PI))
}

/* pdf is |cos(theta)| / (2.0 * pi) */
#[must_use]
pub fn spherical_sample_abs_cos_weighted_uv(u: f64, v: f64) -> (Vec3d, f64) {
    #[allow(clippy::suboptimal_flops)]
    let u = 2.0 * u - 1.0;
    let (omega, pdf) = hemispherical_sample_cos_weighted_uv(u.abs(), v);
    if u < 0.0 {
        (
            Vec3d {
                x: omega.x,
                y: omega.y,
                z: -omega.z,
            },
            pdf / 2.0,
        )
    } else {
        (omega, pdf / 2.0)
    }
}

/* pdf is cos(theta) / pi */
#[must_use]
pub fn hemispherical_sample_cos_weighted_uv(u: f64, v: f64) -> (Vec3d, f64) {
    let eps_theta_sample = u.clamp(1e-6, 1.0); // prevent division by zero (division by pdf)
    let cos_theta = eps_theta_sample.sqrt();
    let sin_theta = (1.0 - eps_theta_sample).sqrt();
    let phi = 2.0 * std::f64::consts::PI * v;
    let (sin_phi, cos_phi) = phi.sin_cos();
    let omega_i = Vec3d {
        x: sin_theta * sin_phi,
        y: sin_theta * cos_phi,
        z: cos_theta,
    };
    (omega_i, cos_theta / std::f64::consts::PI)
}

#[must_use]
pub fn refract_normal(omega_i: Vec3d, omega_o: Vec3d, ior_i: f64, ior_o: f64) -> Option<Vec3d> {
    let m = -(omega_i * ior_i + omega_o * ior_o);
    if m.dot(omega_i) * m.dot(omega_o) < 0.0 {
        Some(m)
    } else {
        None
    }
}

#[must_use]
pub fn refract_good(omega_i: Vec3d, m: Vec3d, ior_i: f64, ior_o: f64) -> Option<Vec3d> {
    let c = omega_i.dot(m);
    let ior_rel = ior_i / ior_o;
    #[allow(clippy::suboptimal_flops)]
    let disc = 1.0 + ior_rel.sq() * (c.sq() - 1.0);
    if disc <= 0.0 {
        None
    } else {
        #[allow(clippy::suboptimal_flops)]
        Some(m * (c * ior_rel - omega_i.z.signum() * disc.sqrt()) - omega_i * ior_rel)
    }
}

pub fn rotate_around_z(v: Vec3d, angle: f64) -> Vec3d {
    let (sin, cos) = angle.sin_cos();
    #[allow(clippy::suboptimal_flops)]
    Vec3d {
        x: v.x * cos - v.y * sin,
        y: v.x * sin + v.y * cos,
        z: v.z,
    }
}

pub fn pow5(v: f64) -> f64 {
    let v2 = v * v;
    v2 * v2 * v
}
