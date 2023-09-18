pub trait ApproxEqual: Copy {
    fn equals_approx(self, other: Self, eps: Self, eps_rel: Self) -> bool;
    fn equals_approx_abs(self, other: Self, eps: Self) -> bool;
    fn equals_approx_rel(self, other: Self, eps: Self) -> bool;
}

macro_rules! assert_eq_approx {
    ($lhs:expr, $rhs:expr, $eps_abs:expr, $eps_rel:expr) => {
        assert!(
            $crate::test_utils::ApproxEqual::equals_approx($lhs, $rhs, $eps_abs, $eps_rel),
            r#"assert_eq_abs failed:
    {}: {:?}
    {}: {:?}
    {} (maximum absolute error): {:?}
    {} (maximum relative error): {:?}"#,
            stringify!($lhs),
            $lhs,
            stringify!($rhs),
            $rhs,
            stringify!($eps_abs),
            $eps_abs,
            stringify!($eps_rel),
            $eps_rel,
        );
    };

    ($lhs:expr, $rhs:expr, $eps_abs: expr, $eps_rel:expr, $($arg:tt)+) => {
        assert!($crate::test_utils::ApproxEqual::equals_approx($lhs, $rhs, $eps_abs, $eps_rel), $($arg)*);
    }
}

macro_rules! assert_eq_approx_abs {
    ($lhs:expr, $rhs:expr, $eps_abs:expr) => {
        assert!(
            $crate::test_utils::ApproxEqual::equals_approx_abs($lhs, $rhs, $eps_abs),
            r#"assert_eq_abs failed:
    {}: {:?}
    {}: {:?}
    {} (maximum absolute error): {:?}"#,
            stringify!($lhs),
            $lhs,
            stringify!($rhs),
            $rhs,
            stringify!($eps_abs),
            $eps_abs,
        )
    };

    ($lhs:expr, $rhs:expr, $eps_abs:expr, $($arg:tt)+) => {
        assert!($crate::test_utils::ApproxEqual::equals_approx_abs($lhs, $rhs, $eps_abs),
        $($arg)*);
    };
}

macro_rules! assert_in_range {
    ($value:expr, $lower:expr, $upper:expr) => {
        assert!(
            $lower <= $value && $value <= $upper,
            r#"assert_in_range failed:
    {} (value): {:?}
    {} (lower bound): {:?}
    {} (upper bound): {:?}"#,
            stringify!($value),
            $value,
            stringify!($lower),
            $lower,
            stringify!($upper),
            $upper
        )
    };
}

macro_rules! impl_approx_equal {
    ($scalar:ty, $vector:ty) => {
        impl ApproxEqual for $scalar {
            fn equals_approx(self, other: Self, eps: Self, eps_rel: Self) -> bool {
                #[allow(clippy::float_cmp)]
                if self == other || (self - other).abs() <= eps {
                    true
                } else {
                    let diff = (self - other).abs();
                    let max = self.abs().max(self.abs());
                    diff <= max * eps_rel
                }
            }

            fn equals_approx_abs(self, other: Self, eps: Self) -> bool {
                #[allow(clippy::float_cmp)]
                if self == other {
                    true
                } else {
                    (self - other).abs() <= eps
                }
            }

            fn equals_approx_rel(self, other: Self, eps: Self) -> bool {
                #[allow(clippy::float_cmp)]
                if self == other {
                    return true;
                }
                let diff = (self - other).abs();
                let max = self.abs().max(other.abs());
                diff <= max * eps
            }
        }

        impl ApproxEqual for $vector {
            fn equals_approx_rel(self, other: Self, eps: Self) -> bool {
                $crate::test_utils::ApproxEqual::equals_approx_rel(self.x, other.x, eps.x)
                    && $crate::test_utils::ApproxEqual::equals_approx_rel(self.y, other.y, eps.y)
                    && $crate::test_utils::ApproxEqual::equals_approx_rel(self.z, other.z, eps.z)
            }
            fn equals_approx_abs(self, other: Self, eps: Self) -> bool {
                $crate::test_utils::ApproxEqual::equals_approx_abs(self.x, other.x, eps.x)
                    && $crate::test_utils::ApproxEqual::equals_approx_abs(self.y, other.y, eps.y)
                    && $crate::test_utils::ApproxEqual::equals_approx_abs(self.z, other.z, eps.z)
            }
            fn equals_approx(self, other: Self, eps_abs: Self, eps_rel: Self) -> bool {
                $crate::test_utils::ApproxEqual::equals_approx(
                    self.x, other.x, eps_abs.x, eps_rel.x,
                ) && $crate::test_utils::ApproxEqual::equals_approx(
                    self.y, other.y, eps_abs.y, eps_rel.y,
                ) && $crate::test_utils::ApproxEqual::equals_approx(
                    self.z, other.z, eps_abs.z, eps_rel.z,
                )
            }
        }
    };
}

impl_approx_equal!(f64, Vec3d);

use std::f64::consts;

pub(crate) use assert_eq_approx;
pub(crate) use assert_eq_approx_abs;

use crate::{
    core::TransmissiveBsdf,
    utils::{FloatExt, VecExt},
    RgbD, SampleIncomingResponse, SampleOutgoingResponse, Vec3d, BSDF,
};

pub trait SamplerExt {
    fn vec3d(&mut self) -> Vec3d;
}

impl SamplerExt for fastrand::Rng {
    fn vec3d(&mut self) -> Vec3d {
        Vec3d::new(self.f64(), self.f64(), self.f64())
    }
}

/** sample a direction with density 1 / 4pi */
pub fn spherical_sample(rd: &mut fastrand::Rng) -> Vec3d {
    let u = rd.f64();
    let v = rd.f64();
    spherical_sample_uv(u, v)
}

fn spherical_sample_uv(u: f64, v: f64) -> Vec3d {
    #[allow(clippy::suboptimal_flops)]
    let cos_theta = 2.0 * u - 1.0;
    #[allow(clippy::suboptimal_flops)]
    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
    let phi = v * 2.0 * consts::PI;
    let (sin_phi, cos_phi) = phi.sin_cos();
    Vec3d::new(sin_theta * sin_phi, sin_theta * cos_phi, cos_theta)
}

#[allow(clippy::cast_lossless)]
pub fn test_energy_conservation<T: BSDF>(material: &T, allowed_energy_loss: f64) {
    let mut rd = fastrand::Rng::new();
    let runs = 100;
    let num_samples = 100_000;
    for _i in 0..runs {
        let omega_i = spherical_sample(&mut rd);
        let mut sum = RgbD::ZERO;
        let mut sum2 = RgbD::ZERO;
        for _ in 0..num_samples {
            let SampleOutgoingResponse { omega_o, bsdf, pdf } =
                material.sample_outgoing(omega_i, rd.vec3d());

            if bsdf.luminance() > 0.0 {
                let contrib = bsdf / pdf * omega_o.z.abs();
                sum += contrib;
                sum2 += contrib.sq();
            }
        }
        sum /= num_samples as f64;
        sum2 /= num_samples as f64;

        let variance =
            (sum2 - sum.sq()).luminance() * num_samples as f64 / (num_samples - 1) as f64;

        // assert!(variance >= 0.0);
        let std_error = (variance.abs() / num_samples as f64).sqrt();
        let confidence = (4.0 * std_error).max(1e-3);

        assert_in_range!(
            sum.x,
            1.0 - confidence - allowed_energy_loss,
            1.0 + confidence
        );
        assert_in_range!(
            sum.y,
            1.0 - confidence - allowed_energy_loss,
            1.0 + confidence
        );
        assert_in_range!(
            sum.z,
            1.0 - confidence - allowed_energy_loss,
            1.0 + confidence
        );
        //     assert_eq_approx_abs!(
        //         sum,
        //         RgbD::WHITE,
        //         Rgb::splat(0.001),
        //         r#"
        // sum: {sum:?}
        // i: {i},
        // std_error: {std_error},
        // omega_i: {omega_i:?}"#
        //     );
    }
}

pub fn test_bsdf_sample_eval<T: BSDF>(material: &T) {
    let mut rd = fastrand::Rng::new();
    // rd.seed(0);
    let runs = 10000;
    for _ in 0..runs {
        let omega_o = spherical_sample(&mut rd);
        let SampleIncomingResponse {
            omega_i,
            emission: _,
            bsdf,
            pdf,
        } = material.sample_incoming(omega_o, rd.vec3d());
        let c_bsdf = material.evaluate(omega_o, omega_i);
        let c_pdf = material.sample_incoming_pdf(omega_o, omega_i);
        assert!(
            (c_pdf > 0.0 && pdf > 0.0) || bsdf.luminance() == 0.0,
            r#"
    PDFs must be greater than 0.
    pdf: {pdf},
    c_pdf: {c_pdf},
    bsdf: {bsdf:?},
    omega_o: {omega_o:?},
    omega_i: {omega_i:?}"#
        );
        assert_eq_approx!(
            pdf,
            c_pdf,
            0.01,
            0.003,
            r#"
    PDFs must be equal for sample_incoming and sample_incoming_pdf,
    pdf: {pdf},
    c_pdf: {c_pdf},
    omega_o: {omega_o:?},
    omega_i: {omega_i:?}"#
        );
        assert_eq_approx!(bsdf, c_bsdf, RgbD::splat(0.001), RgbD::splat(0.001));

        assert!(pdf >= 0.0);
        assert!(bsdf.x >= 0.0);
        assert!(bsdf.y >= 0.0);
        assert!(bsdf.z >= 0.0);
    }
}

pub fn test_bsdf_reciprocity<T: BSDF>(material: &T) {
    let mut rd = fastrand::Rng::new();
    let runs = 10000;
    for _i in 0..runs {
        let omega_o = spherical_sample(&mut rd);
        let omega_i = spherical_sample(&mut rd);

        let c_bsdf = material.evaluate(omega_o, omega_i);
        let r_bsdf = material.evaluate(omega_i, omega_o);

        let c_pdf = material.sample_incoming_pdf(omega_o, omega_i);
        let r_pdf = material.sample_incoming_pdf(omega_i, omega_o);

        assert!(c_pdf >= 0.0, "the pdf should always be more than 0");
        assert!(r_pdf >= 0.0, "the pdf should always be more than 0");
        assert!(c_bsdf.x >= 0.0, "the bsdf should always be positive");
        assert!(c_bsdf.y >= 0.0, "the bsdf should always be positive");
        assert!(c_bsdf.z >= 0.0, "the bsdf should always be positive");

        assert_eq_approx!(c_bsdf, r_bsdf, RgbD::splat(0.001), RgbD::splat(0.0001));
    }
}

pub fn test_integrate_inverse_pdf<T: BSDF>(material: &T) {
    const DOMAIN: f64 = 4.0 * std::f64::consts::PI;

    let mut rd = fastrand::Rng::new();
    // rd.seed(0);
    let runs = 100;
    let num_samples = 1_000_000;
    for i in 0..runs {
        let omega_o: Vec3d = spherical_sample(&mut rd);
        // TODO remove this line
        // let omega_o = omega_o * omega_o.z.signum();
        let mut sum = 0.0;
        let mut sum_of_squared = 0.0;
        for _ in 0..num_samples {
            let pdf = if rd.f32() > 0.5 {
                let SampleIncomingResponse {
                    omega_i: _,
                    emission: _,
                    bsdf: _,
                    pdf: pdf_bsdf,
                } = material.sample_incoming(omega_o, rd.vec3d());
                let spheric_pdf = 1.0 / 4.0 / std::f64::consts::PI;
                #[allow(clippy::suboptimal_flops)]
                {
                    0.5 * spheric_pdf + 0.5 * pdf_bsdf
                }
            } else {
                let omega_i = spherical_sample(&mut rd);
                let spheric_pdf = 1.0 / 4.0 / std::f64::consts::PI;
                let pdf_bsdf = material.sample_incoming_pdf(omega_o, omega_i);
                #[allow(clippy::suboptimal_flops)]
                {
                    0.5 * spheric_pdf + 0.5 * pdf_bsdf as f64
                }
            };
            let value = 1.0 / pdf;
            sum += value;
            sum_of_squared += value.sq();
        }
        sum /= DOMAIN * num_samples as f64;
        sum_of_squared /= DOMAIN.sq() * (num_samples) as f64;
        let variance_unscaled = sum_of_squared - sum.sq();

        let sample_standard_deviation =
            ((num_samples as f64) / (num_samples - 1) as f64 * variance_unscaled).sqrt();
        let standard_error = sample_standard_deviation / (num_samples as f64).sqrt();

        let confidence_thres = 3.5 * standard_error;
        // panic!(r#"lets finish things
        // omega_o: {omega_o:?}"#);
        assert_eq_approx_abs!(
            sum,
            1.0,
            confidence_thres,
            r#"
    expected the monte carlo test to approach 1.
    But it approached {sum} after {num_samples} Samples with a standard error of {standard_error}.
    Required Confidence is {}.
    Difference is {}.
    omega_o: {omega_o:?}
    i: {i}"#,
            confidence_thres,
            (sum - 1.0).abs()
        );

        assert!(
            standard_error < 0.005,
            "standard_error: {standard_error} is not below threshold."
        );
    }
}

pub fn test_bsdf_reciprocity_glass<Material: TransmissiveBsdf>(material: &Material) {
    let get_ior = |z: f64| {
        if z > 0.0 {
            1.0
        } else {
            material.ior()
        }
    };
    let mut rd = fastrand::Rng::new();
    let runs = 10000;
    for _ in 0..runs {
        let omega_o = spherical_sample(&mut rd);
        let omega_i = spherical_sample(&mut rd);
        let c_bsdf = material.evaluate(omega_o, omega_i);
        let r_bsdf = material.evaluate(omega_i, omega_o);
        // let c_pdf = material.sample_incoming_pdf(omega_i, omega_o);
        // assert!(utils::flt::equals_approx(pdf, c_pdf, 0.001, 0.0001), "sample_incoming and sample_incoming_pdf should return the same value for pdf:\n\tsample_incoming returns {}\n\tsample_incoming_pdf returns {}\n\thappened in {}'th iteration", pdf, c_pdf, i);

        let c_bsdf_norm = c_bsdf / get_ior(omega_i.z).sq();
        let r_bsdf_norm = r_bsdf / get_ior(omega_o.z).sq();
        assert_eq_approx!(
            c_bsdf_norm,
            r_bsdf_norm,
            RgbD::splat(0.001),
            RgbD::splat(0.0001),
            r#"
    - c_bsdf_norm: {c_bsdf_norm:?},
    - r_bsdf_norm: {r_bsdf_norm:?},
    - max_delta: 0.001,
    - max_delta_rel: 0.0001,
    - omega_o: {omega_o:?},
    - omega_i: {omega_i:?},
    "#
        );
    }
}
