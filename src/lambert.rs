//! [BSDF] that can resemble smooth Diffuse surfaces like plastic
use std::f64::consts;

use crate::{RgbF, BSDF, Vec3d, SampleIncomingResponse, RgbD, utils::SafeCast};

//TODO move sampling code to utils

/// [BSDF] that can resemble smooth Diffuse surfaces like plastic
#[derive(Copy, Clone)]
pub struct Lambert {
    /// The color. Every component should be in \[0,1\] to preserve physical validity.
    pub kd: RgbF,
}

impl BSDF for Lambert {
    fn sample_incoming(
        &self,
        omega_o: Vec3d,
        rdf: Vec3d,
    ) -> SampleIncomingResponse {
        assert!(omega_o.is_normalized());
        let eps_theta_sample = rdf.x.clamp(1e-6, 1.0); // prevent division by zero (division by pdf)
        let cos_theta = eps_theta_sample.sqrt() * omega_o.z.signum();
        let sin_theta = (1.0 - eps_theta_sample).sqrt();
        let phi = 2.0 * std::f64::consts::PI * rdf.y;
        let (sin_phi, cos_phi) = phi.sin_cos();
        let omega_i = Vec3d {
            x: sin_theta * sin_phi,
            y: sin_theta * cos_phi,
            z: cos_theta,
        };

        SampleIncomingResponse {
            omega_i,
            bsdf: self.kd.safe_cast() / consts::PI,
            emission: RgbD::ZERO,
            pdf: cos_theta.abs() / consts::PI,
        }
    }
    fn evaluate(&self, omega_o: Vec3d, omega_i: Vec3d) -> RgbD {
        assert!(omega_o.is_normalized() && omega_i.is_normalized());
        if omega_i.z * omega_o.z >= 0.0 {
            self.kd.safe_cast() / consts::PI
        } else {
            RgbD::ZERO
        }
    }

    fn sample_incoming_pdf(&self, omega_o: Vec3d, omega_i: Vec3d) -> f64 {
        assert!(omega_o.is_normalized() && omega_i.is_normalized());
        if omega_i.z * omega_o.z >= 0.0 {
            omega_i.z.abs() / consts::PI
        } else {
            0.0
        }
    }

    fn base_color(&self, _omega_o: Vec3d) -> RgbD {
        self.kd.safe_cast()
    }
}


#[cfg(test)]
mod tests {
    use crate::{RgbF, test_utils};

    use super::Lambert;



    #[test]
    fn lambert() {
        let mat = Lambert { kd: RgbF::ONE };
        test_utils::test_bsdf_sample_eval(&mat);
        test_utils::test_bsdf_reciprocity(&mat);
    }

    #[test]
    fn pdf_integral() {
        let mat = Lambert { kd: RgbF::ONE };
        test_utils::test_integrate_inverse_pdf(&mat);
    }

    #[test]
    fn energy_conservation() {
        let mat = Lambert { kd: RgbF::ONE };
        test_utils::test_energy_conservation(&mat, 0.0);
    }
}
