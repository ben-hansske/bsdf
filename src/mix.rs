//! [BSDF] that can blend two arbitrary BSDF's together
use crate::{
    utils::FloatExt, RgbD, SampleEmissionResponse, SampleIncomingResponse, SampleOutgoingResponse,
    Vec2d, Vec3d, BSDF,
};

/// [BSDF] that can blend two arbitrary BSDF's together
pub struct Mix<B1, B2> {
    pub b1: B1,
    pub b2: B2,
    pub factor: f32,
}

impl<B1, B2> BSDF for Mix<B1, B2>
where
    B1: BSDF,
    B2: BSDF,
{
    fn evaluate(&self, omega_o: Vec3d, omega_i: Vec3d) -> RgbD {
        let b1_result = self.b1.evaluate(omega_o, omega_i);
        let b2_result = self.b2.evaluate(omega_o, omega_i);
        RgbD::lerp(b1_result, b2_result, f64::from(self.factor))
    }

    fn emission(&self, omega_o: Vec3d) -> RgbD {
        let e1 = self.b1.emission(omega_o);
        let e2 = self.b2.emission(omega_o);
        RgbD::lerp(e1, e2, f64::from(self.factor))
    }

    fn sample_incoming(&self, omega_o: Vec3d, rdf: Vec3d) -> SampleIncomingResponse {
        let factor = f64::from(self.factor);
        let draw = rdf.z;
        if draw > factor {
            let r = self.b1.sample_incoming(
                omega_o,
                Vec3d {
                    x: rdf.x,
                    y: rdf.y,
                    z: (draw - factor) / (1.0 - factor),
                },
            );
            let bsdf = self.b2.evaluate(omega_o, r.omega_i);
            let emission = self.b2.emission(omega_o);
            let pdf = self.b2.sample_incoming_pdf(omega_o, r.omega_i);
            SampleIncomingResponse {
                omega_i: r.omega_i,
                bsdf: RgbD::lerp(r.bsdf, bsdf, factor),
                emission: RgbD::lerp(r.emission, emission, factor),
                pdf: f64::lerp(r.pdf, pdf, factor),
            }
        } else {
            let r = self.b2.sample_incoming(
                omega_o,
                Vec3d {
                    x: rdf.x,
                    y: rdf.y,
                    z: draw / factor,
                },
            );
            let bsdf = self.b1.evaluate(omega_o, r.omega_i);
            let emission = self.b1.emission(omega_o);
            let pdf = self.b1.sample_incoming_pdf(omega_o, r.omega_i);
            SampleIncomingResponse {
                omega_i: r.omega_i,
                bsdf: RgbD::lerp(bsdf, r.bsdf, factor),
                emission: RgbD::lerp(emission, r.emission, factor),
                pdf: f64::lerp(pdf, r.pdf, factor),
            }
        }
    }
    fn sample_outgoing(&self, omega_i: Vec3d, rdf: Vec3d) -> SampleOutgoingResponse {
        let factor = f64::from(self.factor);
        if rdf.z > self.factor as f64 {
            let r = self.b1.sample_outgoing(
                omega_i,
                Vec3d {
                    x: rdf.x,
                    y: rdf.y,
                    z: (rdf.z - factor) / (1.0 - factor),
                },
            );
            let bsdf = self.b2.evaluate(r.omega_o, omega_i);
            let adjoint = self.b2.evaluate(omega_i, r.omega_o);
            let pdf = self.b2.sample_outgoing_pdf(r.omega_o, omega_i);
            SampleOutgoingResponse {
                omega_o: r.omega_o,
                bsdf: RgbD::lerp(r.bsdf, bsdf, factor),
                adjoint_bsdf: RgbD::lerp(r.adjoint_bsdf, adjoint, factor),
                pdf: f64::lerp(r.pdf, pdf, factor),
            }
        } else {
            let r = self.b2.sample_outgoing(
                omega_i,
                Vec3d {
                    x: rdf.x,
                    y: rdf.y,
                    z: rdf.z / factor,
                },
            );
            let bsdf = self.b1.evaluate(r.omega_o, omega_i);
            let adjoint = self.b1.evaluate(omega_i, r.omega_o);
            let pdf = self.b1.sample_outgoing_pdf(r.omega_o, omega_i);
            SampleOutgoingResponse {
                omega_o: r.omega_o,
                bsdf: RgbD::lerp(bsdf, r.bsdf, factor),
                adjoint_bsdf: RgbD::lerp(adjoint, r.adjoint_bsdf, factor),
                pdf: f64::lerp(pdf, r.pdf, factor),
            }
        }
    }
    fn sample_emission(&self, rd: Vec2d) -> SampleEmissionResponse {
        let factor = f64::from(self.factor);
        if rd.y > factor {
            let r = self.b1.sample_emission(Vec2d {
                x: rd.x,
                y: (rd.y - factor) / (1.0 - factor),
            });
            let emission = self.b2.emission(r.omega_o);
            let pdf = self.b2.sample_emission_pdf(r.omega_o);
            SampleEmissionResponse {
                omega_o: r.omega_o,
                pdf: f64::lerp(r.pdf, pdf, factor),
                emission: RgbD::lerp(r.emission, emission, factor),
            }
        } else {
            let r = self.b2.sample_emission(Vec2d {
                x: rd.x,
                y: rd.y / factor,
            });
            let emission = self.b1.emission(r.omega_o);
            let pdf = self.b1.sample_emission_pdf(r.omega_o);
            SampleEmissionResponse {
                omega_o: r.omega_o,
                emission: RgbD::lerp(emission, r.emission, factor),
                pdf: f64::lerp(pdf, r.pdf, factor),
            }
        }
    }
    fn sample_incoming_pdf(&self, omega_o: Vec3d, omega_i: Vec3d) -> f64 {
        f64::lerp(
            self.b1.sample_incoming_pdf(omega_o, omega_i),
            self.b2.sample_incoming_pdf(omega_o, omega_i),
            f64::from(self.factor),
        )
    }
    fn sample_outgoing_pdf(&self, omega_o: Vec3d, omega_i: Vec3d) -> f64 {
        f64::lerp(
            self.b1.sample_outgoing_pdf(omega_o, omega_i),
            self.b2.sample_outgoing_pdf(omega_o, omega_i),
            f64::from(self.factor),
        )
    }
    fn sample_emission_pdf(&self, omega_o: Vec3d) -> f64 {
        f64::lerp(
            self.b1.sample_emission_pdf(omega_o),
            self.b2.sample_emission_pdf(omega_o),
            f64::from(self.factor),
        )
    }
    fn base_color(&self, omega_o: Vec3d) -> RgbD {
        let e1 = self.b1.base_color(omega_o);
        let e2 = self.b2.base_color(omega_o);
        RgbD::lerp(e1, e2, f64::from(self.factor))
    }
}
