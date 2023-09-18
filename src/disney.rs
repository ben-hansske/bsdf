//! [BSDF] inspired by the Disney BSDF and Blender's Principled BSDF

use crate::{
    ggx::GGX,
    utils::{self, FloatExt, SafeCast, VecExt},
    RgbD, RgbF, SampleEmissionResponse, SampleIncomingResponse, Vec2d, Vec3d, BSDF,
};

const EPS: f32 = 1e-7;

/// This [BSDF] is inspired by Disney's BSDF and the Principled BSDF of Blender.
/// It has lots of parameters and can resemble a wide spectrum of materials.
/// However, there are still effects that this BSDF can't reproduce:
/// * Subsurface Scattering and other Volumetric Effects: This is a for now a BSDF and not a BSSRDF. Therefore Subsurface
/// Scattering events are impossible to reproduce.
/// * Wavelength effects of any kind, since colors are represented as RGB values. For example different IORs for different wavelengths (like you get in prisms that can break
/// light up into its wavelength components)
///
/// **Note: This code is only available with the `disney` feature**
#[derive(Copy, Clone, Debug)]
pub struct Disney {
    /// Determines the Color of the material. All components should be in \[0,1\]
    pub base_color: RgbF,

    /// How much light is emitted by the surface.
    pub emission: RgbF,

    /// How rough this material appears. Does not affect the emission component. Should be in \[0,1\]
    pub roughness: f32,

    /// Determines how stretched out glossy reflections appear. Should be in \[0, 1\]
    pub anisotropic: f32,

    /// How much light is reflected by diffuse objects. Can be used to e.g. resemble shiny plastic.
    pub specular: f32,

    /// Determines in witch direction glossy reflections are stretched out. 0 means 0 degree, while
    /// 1 is 180 degree.
    pub anisotropic_rotation: f32,

    /// Resembles a little bit the appearance of velvet like materials. Note: This is more a cheap hack than a sophisticated Cloth BSDF. Affects only the diffuse component. This component is very week and might values beyond to be actually visible.
    pub sheen: f32,

    /// Controls how much the sheen component is tinted towards the `base_color`. 0 is completely
    /// white, while 1 is the normalized base color.
    pub sheen_tint: f32,

    /// How metallic an objects appears. Note: Only values of either 0 or 1 are physically
    /// plausible. However interpolating between BSDF in parameter space is computationally much cheaper than evaluating both and then blending the results. Therefore, using values in between 0 and 1 makes sense when using textures for blending materials.
    pub metallic: f32,

    /// Provides a secondary specular reflection lobe that resembles a thin layer of coating on top
    /// of the base material layer.
    pub clearcoat: f32,

    /// How rough or shiny this additional reflective layer is.
    pub clearcoat_roughness: f32,

    /// How much light is refracted through the material in a glass like manner.
    /// Note: Since we do not model volumetric scattering, the transmission should be either 0 or 1 to
    /// physically plausible and constant over the entire surface of an object.
    pub transmission: f32,

    /// Index of Refraction. Controls how strong the light is refracted at the boundary in glass.
    /// Physically plausible parameters lie in the range of 1 up to 2. However there are certain
    /// materials where the ior is beyond 2. Physically, this parameter can be understand as
    ///
    /// ```
    ///        speedOfLightAboveTheSurface
    /// ior = -----------------------------
    ///        speetOfLightBelowTheSurface
    /// ```
    ///
    /// Note: Since volumetric refraction is not modelled, this parameter should be constant over the
    /// entire surface of an object.
    pub ior: f32,
}

struct SampleWeights {
    pub(crate) clear_coat_lobe: f64,
    pub(crate) reflection_refraction_lobe: f64,
    pub(crate) diffuse_lobe: f64,
}

impl Disney {
    /// Allows for blending between two Disney BSDFs. The blending is done linearly in parameter
    /// space. If do not like this, try if [`crate::mix::Mix`] does the job for you!
    #[must_use]
    pub fn lerp(self, other: Self, t: f32) -> Self {
        Self {
            base_color: RgbF::lerp(self.base_color, other.base_color, t),
            emission: RgbF::lerp(self.emission, other.emission, t),
            roughness: self.roughness.lerp(other.roughness, t),
            sheen: self.sheen.lerp(other.sheen, t),
            sheen_tint: self.sheen_tint.lerp(other.sheen_tint, t),
            specular: self.specular.lerp(other.specular, t),
            metallic: self.metallic.lerp(other.metallic, t),
            anisotropic: self.anisotropic.lerp(other.anisotropic, t),
            anisotropic_rotation: self
                .anisotropic_rotation
                .lerp(other.anisotropic_rotation, t),
            clearcoat: self.clearcoat.lerp(other.clearcoat, t),
            clearcoat_roughness: self.clearcoat_roughness.lerp(other.clearcoat_roughness, t),
            transmission: self.transmission.lerp(other.transmission, t),
            ior: self.ior.lerp(other.ior, t),
        }
    }
}

#[allow(clippy::cast_lossless)]
impl Disney {
    const IOR_METALLIC: f64 = 1.45;

    fn ior_pair(&self, omega: Vec3d) -> (f64, f64) {
        if omega.z < 0.0 {
            (f64::from(self.ior), 1.0)
        } else {
            (1.0, f64::from(self.ior))
        }
    }

    fn ggx(&self) -> GGX {
        let alpha = self.roughness as f64 * self.roughness as f64;
        let alpha = alpha.max(0.001);

        #[allow(clippy::suboptimal_flops)]
        let aspect = (1.0 - 0.9 * self.anisotropic as f64).sqrt();
        GGX {
            alpha_x: alpha / aspect,
            alpha_y: alpha * aspect,
        }
    }

    fn clear_coat_ggx(&self) -> GGX {
        let alpha = self.clearcoat_roughness as f64 * self.clearcoat_roughness as f64;
        let alpha = alpha.max(0.001);
        GGX {
            alpha_x: alpha,
            alpha_y: alpha,
        }
    }

    fn rotate_anisotropic_directions(&self, omega: Vec3d) -> Vec3d {
        if self.anisotropic_rotation < EPS || self.anisotropic_rotation > (1.0 - EPS) {
            omega
        } else {
            utils::rotate_around_z(
                omega,
                -self.anisotropic_rotation as f64 * std::f64::consts::PI,
            )
        }
    }

    fn rotate_anisotropic_directions_back(&self, omega: Vec3d) -> Vec3d {
        if self.anisotropic_rotation < EPS || self.anisotropic_rotation > (1.0 - EPS) {
            omega
        } else {
            utils::rotate_around_z(
                omega,
                self.anisotropic_rotation as f64 * std::f64::consts::PI,
            )
        }
    }

    fn evaluate_diffuse_lobe(&self, omega_o: Vec3d, omega_i: Vec3d) -> RgbD {
        if omega_i.z * omega_o.z <= 0.0 || (1.0 - self.metallic) * (1.0 - self.transmission) < EPS {
            return RgbD::ZERO;
        }

        let fi = utils::pow5(1.0 - omega_i.z.abs());
        let fo = utils::pow5(1.0 - omega_o.z.abs());
        // TODO use trig identities to avoid computing h. the angle between h and omega_i is half
        // of the angle between omega_i and omega_o
        let h = (omega_o + omega_i).normalize();
        let h_dot_i = h.dot(omega_i);
        let rr = 2.0 * self.roughness as f64 * h_dot_i.sq();

        #[allow(clippy::suboptimal_flops)]
        let f_lambert = (1.0 - 0.5 * fi) * (1.0 - 0.5 * fo);
        #[allow(clippy::suboptimal_flops)]
        let f_retro = rr * (fi + fo + fi * fo * (rr - 1.0));
        let f_sheen = utils::pow5(1.0 - h_dot_i.abs()) * self.sheen as f64;

        let base_color = self.base_color.safe_cast();
        let base_color_luminance = base_color.luminance();
        let c_tint = if base_color_luminance > 1e-4 {
            base_color / base_color_luminance
        } else {
            RgbD::ONE
        };
        let c_sheen = RgbD::lerp(RgbD::ONE, c_tint, self.sheen_tint as f64);

        #[allow(clippy::suboptimal_flops)]
        let diffuse =
            (f_lambert + 0.5 * f_retro) * base_color / std::f64::consts::PI + c_sheen * f_sheen;
        diffuse * (1.0 - self.metallic as f64) * (1.0 - self.transmission as f64)
    }

    fn evaluate_clearcoat_lobe(&self, omega_o: Vec3d, omega_i: Vec3d) -> RgbD {
        if self.clearcoat < EPS {
            return RgbD::ZERO;
        }

        let m = utils::positive_reflect_normal(omega_i, omega_o);
        if !GGX::g2_d_satisfied_opt(omega_i, omega_o, m) {
            return RgbD::ZERO;
        }
        let m = m.unwrap();

        let ior = f64::lerp(1.0, 2.0, self.clearcoat as f64);
        let ggx = self.clear_coat_ggx();

        let fresnel = utils::fresnel(omega_i, m, 1.0, ior);
        let masking_shadowing = ggx.geometric(omega_i, omega_o, m);
        let normal_distribution_function = ggx.ndf(m);

        let result = fresnel * masking_shadowing * normal_distribution_function
            / (4.0 * omega_i.z * omega_o.z).abs();

        RgbD::splat(result)
    }

    fn evaluate_specular_lobe(&self, omega_o: Vec3d, omega_i: Vec3d) -> RgbD {
        #[allow(clippy::suboptimal_flops)]
        if 1.0 - (1.0 - self.metallic) * (1.0 - self.transmission) * (1.0 - self.specular) < EPS {
            return RgbD::ZERO;
        }

        let m = utils::positive_reflect_normal(omega_i, omega_o);
        if !GGX::g2_d_satisfied_opt(omega_i, omega_o, m) {
            return RgbD::ZERO;
        }
        let m = m.unwrap();

        let ggx = self.ggx();

        let ior_refl = 1.0
            .lerp(2.0, self.specular as f64)
            .lerp(Self::IOR_METALLIC, self.metallic as f64);

        let (ior_o, ior_t) = if omega_o.z < 0.0 {
            (self.ior as f64, 1.0)
        } else {
            (1.0, self.ior as f64)
        };

        let fresnel_transmission = utils::fresnel(omega_o, m, ior_o, ior_t);
        let fresnel = utils::fresnel(omega_i, m, 1.0, ior_refl);

        let fresnel_conductive = RgbD::lerp(self.base_color.safe_cast(), RgbD::ONE, fresnel);

        let fresnel_final = RgbD::splat(
            fresnel.lerp(fresnel_transmission, self.transmission as f64)
                * (1.0 - self.metallic as f64),
        ) + fresnel_conductive * self.metallic as f64;

        let masking_shadowing = ggx.geometric(omega_i, omega_o, m);
        let normal_distribution_function = ggx.ndf(m);

        fresnel_final * masking_shadowing * normal_distribution_function
            / (4.0 * omega_i.z * omega_o.z).abs()
    }

    fn evaluate_transmissive_lobe(&self, omega_o: Vec3d, omega_i: Vec3d) -> RgbD {
        if omega_i.z * omega_o.z > 0.0 || self.transmission < EPS {
            return RgbD::ZERO;
        }
        let (ior_o, ior_i) = self.ior_pair(omega_o);

        let m = match utils::refract_normal(omega_i, omega_o, ior_i, ior_o)
            .and_then(Vec3d::try_normalize)
        {
            None => return RgbD::ZERO,
            Some(m) if self.ior > 1.0 => m,
            Some(m) => -m,
        };

        if !GGX::g2_d_satisfied(omega_i, omega_o, m) {
            return RgbD::ZERO;
        }

        let fresnel = utils::fresnel(omega_i, m, ior_i, ior_o);
        let masking_shadowing = self.ggx().geometric(omega_i, omega_o, m);
        let ndf = self.ggx().ndf(m);

        #[allow(clippy::suboptimal_flops)]
        let dm_domega_o = ior_o.sq() / (ior_i * omega_i.dot(m) + ior_o * omega_o.dot(m)).sq();

        let fac = ((omega_i.dot(m) / omega_i.z) * (omega_o.dot(m) / omega_o.z)).abs();

        let radiance_correction = ior_i / ior_o;

        let btdf = masking_shadowing
            * ndf
            * (1.0 - fresnel)
            * dm_domega_o
            * fac
            * radiance_correction.sq();

        let blend_factor = (1.0 - self.metallic as f64) * self.transmission as f64;

        self.base_color.safe_cast().sqrt() * blend_factor * btdf
    }

    #[allow(clippy::let_and_return)]
    fn evaluate_all(&self, omega_o: Vec3d, omega_i: Vec3d) -> RgbD {
        assert!(omega_o.is_normalized());
        assert!(omega_i.is_normalized());
        if omega_o.z * omega_i.z > 0.0 {
            let f = self.evaluate_diffuse_lobe(omega_o, omega_i);
            let f = f + self.evaluate_specular_lobe(omega_o, omega_i);
            let f = f + self.evaluate_clearcoat_lobe(omega_o, omega_i);
            f
        } else {
            let f = self.evaluate_transmissive_lobe(omega_o, omega_i);
            f
        }
    }

    // --------------- Sampling routines ---------------------------

    fn sample_diffuse_lobe(omega_o: Vec3d, rdf: Vec2d) -> (Vec3d, f64) {
        utils::sample_diffuse_lobe(omega_o, rdf.x, rdf.y)
    }

    fn sample_reflection_refraction_lobe(&self, omega_o: Vec3d, rdf: Vec3d) -> (Vec3d, f64) {
        let (ior_o, ior_t) = self.ior_pair(omega_o);

        let m = self
            .ggx()
            .sample_vndf(omega_o * omega_o.z.signum(), rdf.x, rdf.y);

        let fresnel_transmission = utils::fresnel(omega_o, m, ior_o, ior_t);

        let transmission_weight =
            (1.0 - fresnel_transmission) * self.transmission as f64 * (1.0 - self.metallic as f64);

        let vndf = self.ggx().vndf(omega_o * omega_o.z.signum(), m);

        if rdf.z < transmission_weight {
            let omega_i = utils::refract_good(omega_o, m, ior_o, ior_t)
                .unwrap()
                .normalize();

            #[allow(clippy::suboptimal_flops)]
            let jacobian = ior_t.sq() * omega_i.dot(m).abs()
                / (ior_t * m.dot(omega_i) + ior_o * m.dot(omega_o)).sq();

            #[allow(clippy::suboptimal_flops)]
            let pdf = transmission_weight * vndf * jacobian
                + self.sample_reflection_lobe_pdf(omega_o, omega_i);

            (omega_i, pdf)
        } else {
            let omega_i = utils::reflect(m, omega_o);

            let jacobian = 1.0 / 4.0 / m.dot(omega_o).abs();

            #[allow(clippy::suboptimal_flops)]
            let pdf = (1.0 - transmission_weight) * vndf * jacobian
                + self.sample_refraction_lobe_pdf(omega_o, omega_i);

            (omega_i, pdf)
        }
    }

    fn sample_clearcoat_lobe(&self, omega_o: Vec3d, rdf: Vec2d) -> (Vec3d, f64) {
        utils::sample_ggx_vndf_reflection_lobe(self.clear_coat_ggx(), omega_o, rdf)
    }

    //****************** Sampling pdfs *****************************
    fn sample_refraction_lobe_pdf(&self, omega_o: Vec3d, omega_i: Vec3d) -> f64 {
        if self.transmission < EPS {
            return 0.0;
        }
        let (ior_o, ior_i) = self.ior_pair(omega_o);
        let m = match utils::refract_normal(omega_i, omega_o, ior_i, ior_o)
            .and_then(Vec3d::try_normalize)
        {
            None => return 0.0,
            Some(m) if m.z >= 0.0 => m,
            Some(m) => -m,
        };

        let fresnel_transmission = utils::fresnel(omega_o, m, ior_o, ior_i);

        let transmission_weight =
            (1.0 - fresnel_transmission) * self.transmission as f64 * (1.0 - self.metallic as f64);

        let vndf = self.ggx().vndf(omega_o * omega_o.z.signum(), m);

        #[allow(clippy::suboptimal_flops)]
        let jacobian = ior_i.sq() * omega_i.dot(m).abs()
            / (ior_i * m.dot(omega_i) + ior_o * m.dot(omega_o)).sq();

        jacobian * vndf * transmission_weight
    }

    fn sample_reflection_lobe_pdf(&self, omega_o: Vec3d, omega_i: Vec3d) -> f64 {
        #[allow(clippy::suboptimal_flops)]
        if 1.0 - (1.0 - self.metallic) * (1.0 - self.transmission) * (1.0 - self.specular) < EPS {
            return 0.0;
        }
        // allow opposite sides, so that we do not lie about the pdf
        let m = utils::positive_reflect_normal_allow_opposite_sides(omega_i, omega_o);
        let Some(m) = m else { return 0.0 };

        let (ior_o, ior_t) = self.ior_pair(omega_o);

        let fresnel_transmission = utils::fresnel(omega_o, m, ior_o, ior_t);

        let transmission_weight =
            (1.0 - fresnel_transmission) * self.transmission as f64 * (1.0 - self.metallic as f64);

        let vndf_pdf = utils::sample_ggx_vndf_reflection_lobe_pdf(self.ggx(), omega_o, m);

        vndf_pdf * (1.0 - transmission_weight)
    }

    fn sample_clearcoat_lobe_pdf(&self, omega_o: Vec3d, omega_i: Vec3d) -> f64 {
        if self.clearcoat < EPS {
            return 0.0;
        }

        // allow opposite sides, so that we do not lie about the pdf
        let m = utils::positive_reflect_normal_allow_opposite_sides(omega_i, omega_o);
        let Some(m) = m else { return 0.0 };
        utils::sample_ggx_vndf_reflection_lobe_pdf(self.clear_coat_ggx(), omega_o, m)
    }

    fn sample_diffuse_lobe_pdf(omega_o: Vec3d, omega_i: Vec3d) -> f64 {
        utils::sample_diffuse_lobe_pdf(omega_o, omega_i)
    }

    fn compute_weights(&self) -> SampleWeights {
        let clear_coat = self.clearcoat as f64 * 0.3;
        let plastic_reflection_weight = self.specular as f64 * 0.5;
        let dielectric_reflection_refraction_weight =
            plastic_reflection_weight.lerp(1.0, self.transmission as f64);

        let reflection_refraction = f64::lerp(
            dielectric_reflection_refraction_weight,
            1.0,
            self.metallic as f64,
        );

        let diffuse = 1.0 - reflection_refraction;
        SampleWeights {
            clear_coat_lobe: clear_coat,
            reflection_refraction_lobe: (1.0 - clear_coat) * reflection_refraction,
            diffuse_lobe: (1.0 - clear_coat) * diffuse,
        }
    }

    fn sample_mixed_lobe(&self, omega_o: Vec3d, rdf: Vec3d) -> (Vec3d, f64) {
        let weights = self.compute_weights();

        let choice = rdf.z;
        if choice < weights.clear_coat_lobe {
            // Clearcoat
            let (omega_i, clearcoat_pdf) =
                self.sample_clearcoat_lobe(omega_o, Vec2d { x: rdf.x, y: rdf.y });
            let spec_pdf = self.sample_reflection_lobe_pdf(omega_o, omega_i);
            let lamb_pdf = Self::sample_diffuse_lobe_pdf(omega_o, omega_i);
            let trans_pdf = self.sample_refraction_lobe_pdf(omega_o, omega_i);

            #[allow(clippy::suboptimal_flops)]
            let combined_pdf = clearcoat_pdf * weights.clear_coat_lobe
                + (spec_pdf + trans_pdf) * weights.reflection_refraction_lobe
                + lamb_pdf * weights.diffuse_lobe;
            (omega_i, combined_pdf)
        } else if choice < weights.clear_coat_lobe + weights.reflection_refraction_lobe {
            // Reflection
            let (omega_i, spec_pdf) = self.sample_reflection_refraction_lobe(
                omega_o,
                Vec3d {
                    x: rdf.x,
                    y: rdf.y,
                    z: (choice - weights.clear_coat_lobe) / weights.reflection_refraction_lobe,
                },
            );
            let clearcoat_pdf = self.sample_clearcoat_lobe_pdf(omega_o, omega_i);
            let lamb_pdf = Self::sample_diffuse_lobe_pdf(omega_o, omega_i);

            #[allow(clippy::suboptimal_flops)]
            let combined_pdf = clearcoat_pdf * weights.clear_coat_lobe
                + spec_pdf * weights.reflection_refraction_lobe
                + lamb_pdf * weights.diffuse_lobe;
            (omega_i, combined_pdf)
        } else {
            // Lambert
            let (omega_i, lamb_pdf) =
                Self::sample_diffuse_lobe(omega_o, Vec2d { x: rdf.x, y: rdf.y });
            let clearcoat_pdf = self.sample_clearcoat_lobe_pdf(omega_o, omega_i);
            let spec_pdf = self.sample_reflection_lobe_pdf(omega_o, omega_i);
            let trans_pdf = self.sample_refraction_lobe_pdf(omega_o, omega_i);

            #[allow(clippy::suboptimal_flops)]
            let combined_pdf = clearcoat_pdf * weights.clear_coat_lobe
                + (spec_pdf + trans_pdf) * weights.reflection_refraction_lobe
                + lamb_pdf * weights.diffuse_lobe;
            (omega_i, combined_pdf)
        }
    }

    fn sample_mixed_lobe_pdf(&self, omega_o: Vec3d, omega_i: Vec3d) -> f64 {
        let weights = self.compute_weights();

        let clearcoat_pdf = self.sample_clearcoat_lobe_pdf(omega_o, omega_i);
        let spec_pdf = self.sample_reflection_lobe_pdf(omega_o, omega_i);
        let lamb_pdf = Self::sample_diffuse_lobe_pdf(omega_o, omega_i);
        let trans_pdf = self.sample_refraction_lobe_pdf(omega_o, omega_i);

        #[allow(clippy::suboptimal_flops)]
        {
            clearcoat_pdf * weights.clear_coat_lobe
                + (spec_pdf + trans_pdf) * weights.reflection_refraction_lobe
                + lamb_pdf * weights.diffuse_lobe
        }
    }
}

impl BSDF for Disney {
    fn evaluate(&self, omega_o: Vec3d, omega_i: Vec3d) -> RgbD {
        let omega_o = self.rotate_anisotropic_directions(omega_o);
        let omega_i = self.rotate_anisotropic_directions(omega_i);
        self.evaluate_all(omega_o, omega_i)
    }

    fn sample_incoming(&self, omega_o: Vec3d, rdf: Vec3d) -> SampleIncomingResponse {
        assert!(omega_o.is_normalized());

        let omega_o_rot = self.rotate_anisotropic_directions(omega_o);
        let (omega_i_rot, pdf) = self.sample_mixed_lobe(omega_o_rot, rdf);
        let omega_i = self.rotate_anisotropic_directions_back(omega_i_rot);
        SampleIncomingResponse {
            omega_i,
            bsdf: self.evaluate(omega_o, omega_i),
            emission: self.emission.safe_cast(),
            pdf,
        }
    }

    fn sample_emission_pdf(&self, omega_o: Vec3d) -> f64 {
        assert!(omega_o.is_normalized());
        omega_o.z.abs() / (2.0 * std::f64::consts::PI)
    }

    fn sample_emission(&self, rdf: Vec2d) -> SampleEmissionResponse {
        let (omega_o, pdf) = utils::spherical_sample_abs_cos_weighted_uv(rdf.x, rdf.y);
        SampleEmissionResponse {
            omega_o,
            emission: self.emission.safe_cast(),
            pdf,
        }
    }
    fn emission(&self, omega_o: Vec3d) -> RgbD {
        assert!(omega_o.is_normalized());
        self.emission.safe_cast()
    }

    fn sample_incoming_pdf(&self, omega_o: Vec3d, omega_i: Vec3d) -> f64 {
        assert!(omega_o.is_normalized() && omega_i.is_normalized());
        let omega_o = self.rotate_anisotropic_directions(omega_o);
        let omega_i = self.rotate_anisotropic_directions(omega_i);
        self.sample_mixed_lobe_pdf(omega_o, omega_i)
    }

    fn base_color(&self, _omega_o: Vec3d) -> RgbD {
        self.base_color.safe_cast() + self.emission.safe_cast()
    }
}

#[cfg(test)]
impl crate::core::TransmissiveBsdf for Disney {
    fn ior(&self) -> f64 {
        f64::from(self.ior)
    }
}

impl Default for Disney {
    fn default() -> Self {
        Self {
            base_color: RgbF::ONE,
            emission: RgbF::ZERO,
            roughness: 0.5,
            sheen: 0.0,
            sheen_tint: 0.0,
            specular: 0.5,
            metallic: 0.0,
            anisotropic: 0.0,
            anisotropic_rotation: 0.0,
            clearcoat: 0.0,
            clearcoat_roughness: 0.0,
            transmission: 0.0,
            ior: 1.45,
        }
    }
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
#[allow(clippy::float_cmp)]
#[allow(clippy::cast_lossless)]
#[allow(clippy::suboptimal_flops)]
mod tests {

    use crate::conductive::Conductive;
    use crate::ggx::GGX;
    use crate::rough_glass::RoughGlass;
    use crate::{RgbD, RgbF, BSDF};

    use super::super::test_utils;
    use super::Disney;
    #[test]
    fn disney() {
        let mat = Disney {
            base_color: RgbF::ONE,
            roughness: 0.3,
            sheen: 0.5,
            sheen_tint: 0.0,
            specular: 0.5,
            metallic: 0.5,
            anisotropic: 0.5,
            anisotropic_rotation: 0.2,
            clearcoat: 0.5,
            clearcoat_roughness: 0.1,
            ior: 1.3,
            transmission: 0.5,
            ..Default::default()
        };
        let weights = mat.compute_weights();
        assert_eq!(
            1.0,
            weights.clear_coat_lobe + weights.reflection_refraction_lobe + weights.diffuse_lobe,
            "sample weights must add up to 1"
        );

        test_utils::test_bsdf_sample_eval(&mat);
        test_utils::test_bsdf_reciprocity_glass(&mat);
    }

    #[test]
    fn integrate_inverse_pdf() {
        let mat = Disney {
            base_color: RgbF::ONE,
            emission: RgbF::ZERO,
            roughness: 0.3,
            sheen: 0.5,
            sheen_tint: 0.0,
            specular: 0.5,
            metallic: 0.5,
            anisotropic: 0.5,
            anisotropic_rotation: 0.2,
            clearcoat: 0.5,
            clearcoat_roughness: 0.1,
            ior: 1.3,
            transmission: 0.5,
        };

        test_utils::test_integrate_inverse_pdf(&mat);
    }

    #[test]
    fn compare_with_conductive() {
        let base_color = RgbF::new(0.5, 0.7, 0.9);
        let roughness: f64 = 0.3;
        let anisotropic: f64 = 0.4;
        let alpha = roughness * roughness;
        let alpha = alpha.max(0.001);
        let aspect = (1.0 - 0.9 * anisotropic).sqrt();

        // constant value for metals
        let ior = Disney::IOR_METALLIC;

        let mat = Disney {
            base_color,
            roughness: roughness as f32,
            metallic: 1.0,
            anisotropic: anisotropic as f32,
            ..Default::default()
        };

        let conductive = Conductive {
            color: base_color,
            ggx: GGX {
                alpha_x: (alpha / aspect),
                alpha_y: (alpha * aspect),
            },
            ior: ior as f32,
        };

        let mut rd = fastrand::Rng::new();

        for i in 0..100_000 {
            let omega_i = test_utils::spherical_sample(&mut rd);
            let omega_o = test_utils::spherical_sample(&mut rd);

            let bsdf = mat.evaluate(omega_o, omega_i);

            let reference = conductive.evaluate(omega_o, omega_i);

            let pdf = mat.sample_incoming_pdf(omega_o, omega_i);
            let pdf_reference = conductive.sample_incoming_pdf(omega_o, omega_i);
            test_utils::assert_eq_approx!(
                bsdf,
                reference,
                RgbD::splat(0.0001),
                RgbD::splat(0.001),
                r#"
    bsdf: {bsdf:?}
    reference: {reference:?}
    omega_i: {omega_i:?}
    omega_o: {omega_o:?}
    i: {i}
    "#
            );

            test_utils::assert_eq_approx!(
                pdf,
                pdf_reference,
                0.0001,
                0.001,
                r#"
    pdf: {pdf}
    reference: {pdf_reference}
    omega_i: {omega_i:?}
    omega_o: {omega_o:?}
    i: {i}
    "#
            );
        }
    }

    #[test]
    fn compare_with_rough_glass() {
        let roughness: f64 = 0.3;
        let anisotropic: f64 = 0.4;

        let ior = 1.45;

        let mat = Disney {
            transmission: 1.0,
            ior,
            roughness: roughness as f32,
            anisotropic: anisotropic as f32,
            ..Default::default()
        };

        let rough_glass = RoughGlass {
            ggx: GGX::from_remapped(roughness, anisotropic),
            ior: ior as f64,
        };

        let mut rd = fastrand::Rng::new();

        for i in 0..100_000 {
            let omega_i = test_utils::spherical_sample(&mut rd);
            let omega_o = test_utils::spherical_sample(&mut rd);

            let bsdf = mat.evaluate(omega_o, omega_i);
            let reference = rough_glass.evaluate(omega_o, omega_i);

            let pdf = mat.sample_incoming_pdf(omega_o, omega_i);
            let pdf_reference = rough_glass.sample_incoming_pdf(omega_o, omega_i);
            test_utils::assert_eq_approx!(
                bsdf,
                reference,
                RgbD::splat(0.0001),
                RgbD::splat(0.001),
                r#"
    bsdf: {bsdf:?}
    reference: {reference:?}
    omega_i: {omega_i:?}
    omega_o: {omega_o:?}
    i: {i}
    "#
            );

            test_utils::assert_eq_approx!(
                pdf,
                pdf_reference,
                0.000_001,
                0.0001,
                r#"
    pdf: {pdf}
    reference: {pdf_reference}
    omega_i: {omega_i:?}
    omega_o: {omega_o:?}
    i: {i}
            "#
            );
        }
    }
}
