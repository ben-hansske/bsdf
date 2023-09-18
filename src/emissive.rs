//! [BSDF] that emits light but reflects none
use std::f64::consts;

use crate::{RgbF, BSDF, Vec3d, SampleIncomingResponse, RgbD, Vec2d, SampleEmissionResponse, utils::{self, SafeCast}};




/// A [BSDF] that emits light but scatters none
/// Can be used for lights
pub struct Emissive {
    /// the amount of emission
    pub emission: RgbF,
}

impl BSDF for Emissive {
    fn sample_incoming(
        &self,
        _omega_o: Vec3d,
        _rd: Vec3d
    ) -> SampleIncomingResponse {
        SampleIncomingResponse {
            omega_i: Vec3d::new(0.0, 0.0, 0.0),
            bsdf: RgbD::ZERO,
            emission: self.emission.safe_cast(),
            pdf: 1.0,
        }
    }

    fn evaluate(&self, omega_o: Vec3d, omega_i: Vec3d) -> RgbD {
        assert!(omega_o.is_normalized() && omega_i.is_normalized());
        RgbD::ZERO
    }

    fn emission(&self, omega_o: Vec3d) -> RgbD {
        assert!(omega_o.is_normalized());
        self.emission.safe_cast()
    }

    fn sample_emission_pdf(&self, omega_o: Vec3d) -> f64 {
        assert!(omega_o.is_normalized());
        omega_o.z.abs() / (2.0 * consts::PI)
    }

    fn sample_emission(&self, rdf: Vec2d) -> SampleEmissionResponse {
        let (omega_o, pdf) = utils::spherical_sample_abs_cos_weighted_uv(rdf.x, rdf.y);
        SampleEmissionResponse {
            omega_o,
            emission: self.emission.safe_cast(),
            pdf,
        }
    }

    fn sample_incoming_pdf(&self, omega_o: Vec3d, omega_i: Vec3d) -> f64 {
        assert!(omega_o.is_normalized() && omega_i.is_normalized());
        1.0
    }

    fn base_color(&self, _omega_o: Vec3d) -> RgbD {
        self.emission.safe_cast()
    }
}


