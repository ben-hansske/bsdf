/// used for colors
pub type RgbD = glam::f64::DVec3;
/// used for colors
pub type RgbF = glam::f32::Vec3;

/// used for direction vectors
pub type Vec3d = glam::f64::DVec3;
/// used for direction vectors
pub type Vec2d = glam::f64::DVec2;

/// Contains the Data that is returned by [`BSDF::sample_incoming`]
pub struct SampleIncomingResponse {
    /// # Incoming Direction
    ///The direction where light could be arriving at the surface
    pub omega_i: Vec3d,

    /// The value at for the BSDF. Indicates how much light is scattered from the incoming
    /// direction to the outgoing direction
    pub bsdf: RgbD,

    /// Indicates how much light is emitted in the outgoing direction
    pub emission: RgbD,

    /// The probability distribution for choosing `omega_i` given `omega_o`
    pub pdf: f64,
}

/// Contains the Data that is returned by [`BSDF::sample_outgoing`]
pub struct SampleOutgoingResponse {
    /// The direction to which light is scattered to
    pub omega_o: Vec3d,

    /// The value at for the BSDF. Indicates how much light is scattered from the incoming
    /// direction to the outgoing direction
    pub bsdf: RgbD,

    /// The probability distribution for choosing `omega_o` given `omega_i`
    pub pdf: f64,
}

/// Contains the Data that is returned by [`BSDF::sample_emission`]
pub struct SampleEmissionResponse {
    /// The direction to which light is emitted to
    pub omega_o: Vec3d,

    /// The amount of light which is emitted
    pub emission: RgbD,

    /// The probability density of choosing that exact direction
    pub pdf: f64,
}

/// Bidirectional Scattering Distribution Functions. A trait that describes the surface properties
/// of a material if you will.
///
/// This trait contains functions to importance sample and evaluate a specific BSDF
pub trait BSDF {
    /// Given a direction where light is scattered to, samples an incident direction, from which the light
    /// may come from
    ///
    /// # Arguments
    /// * `omega_o` - The direction where light is scattered to. Outgoing direction
    /// * `rd` - A random distribution for sampling
    ///
    /// # Return
    /// See [`SampleIncomingResponse`]
    fn sample_incoming(&self, omega_o: Vec3d, rdf: Vec3d) -> SampleIncomingResponse;

    /// Given an incident light direction, samples a direction where light is scattered to
    ///
    /// # Arguments
    /// * `omega_i` - A direction where light is coming from
    /// * `surface` - The properties of the surface at a given point
    /// * `rd` - A random distribution for sampling
    /// # Return
    /// See [`SampleOutgoingResponse`]
    fn sample_outgoing(&self, omega_i: Vec3d, rdf: Vec3d) -> SampleOutgoingResponse {
        assert!(omega_i.is_normalized());
        let response = self.sample_incoming(omega_i, rdf);
        SampleOutgoingResponse {
            omega_o: response.omega_i,
            bsdf: response.bsdf,
            pdf: response.pdf,
        }
    }

    /// Returns the value of the BSDF at the given directions
    ///
    /// # Arguments
    /// * `omega_o` - Exitant light direction
    /// * `omega_i` - Incident light direction
    fn evaluate(&self, omega_o: Vec3d, omega_i: Vec3d) -> RgbD;

    /// Returns the probability density sampling an incoming direction given an outgoing direction
    /// See [`BSDF::sample_incoming`] and [`SampleIncomingResponse`]
    fn sample_incoming_pdf(&self, omega_o: Vec3d, omega_i: Vec3d) -> f64;

    /// Returns the probability density sampling an outgoing direction given an incoming direction
    /// See [`BSDF::sample_outgoing_pdf`] and [`SampleOutgoingResponse`]
    fn sample_outgoing_pdf(&self, omega_o: Vec3d, omega_i: Vec3d) -> f64 {
        assert!(omega_o.is_normalized());
        assert!(omega_i.is_normalized());
        self.sample_incoming_pdf(omega_i, omega_o)
    }

    /// Returns how much light is emitted to the given direction `omega_o`
    fn emission(&self, omega_o: Vec3d) -> RgbD {
        assert!(omega_o.is_normalized());
        RgbD::ZERO
    }

    /// Returns the probability density of choosing the given direction when sampled with
    /// [`BSDF::sample_emission`]
    /// Returns `0.0` if the surface does not emit light
    fn sample_emission_pdf(&self, omega_o: Vec3d) -> f64 {
        assert!(omega_o.is_normalized());
        0.0
    }

    /// Samples a direction in which light is emitted to and also returns the amount of emission
    /// The returned vector is the null vector, if this surface does not emit any light
    /// See also: [`SampleEmissionResponse`]
    fn sample_emission(&self, _rdf: Vec2d) -> SampleEmissionResponse {
        SampleEmissionResponse {
            omega_o: Vec3d::ZERO,
            emission: RgbD::ZERO,
            pdf: 0.0,
        }
    }

    /// Returns the base color of the surface.
    /// This function is used to generate auxiliary images for AI tools such as Open Image Denoise
    fn base_color(&self, omega_o: Vec3d) -> RgbD;
}

/// this is only used for testing
#[cfg(test)]
pub trait TransmissiveBsdf: BSDF {
    fn ior(&self) -> f64;
}
