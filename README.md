# bsdf
A rust implementation of BSDFs for pathtracing

This crate is designed to cover a wide range of materials in a path tracer.
Furthermore, methods for importance sampling are provided.

# Design Decisions
**NOTE: This crate is pretty much in alpha state. Therefore a lot of the following things may
or may not change in the future**

The code is geared towards pathtracing. Direct lighting techniques such as image based lighting or
polygonal lights are **not implemented**.

Lighting calculations are done exclusively in f64s. This is because BSDFs can be extremely
spiky. Using f64 has helped reducing numerical errors. However, material parameters are
stored as f32s for a minimal memory footprint. All materials can be constructed on the fly,
since they (currently) do not rely on precomputed data. At some point in the future we might
address this with generic implementations over f32 and f64.

BSDFs are computed in a local space. That means, the surface is assumed to be the xy-plane
and the z-vector is assumed to be the normal. Therefore incident and exitant vectors must be
rotated  before or after evaluation of the BSDF.

The `|omega_i.dot(n)|`, `|cos theta_i|` or `|omega_i.z|` are not part of the BSDF. The user is
responsible for multiplying them in if necessary (almost always). Pdf's on the other hand,
are meant for high quality importance sampling. Therefore, they try to take this cosine term
into account when generating samples.

`sample_...` functions are deterministic. That means you are responsible for generate f64 in the
range of `0.0..1.0`. This allows you to control the sampling process and the random generator
or low discrepancy sequence in use. These random floats are either passed as a Vec3d or
Vec2d

This crate is built on [glam](https://crates.io/crates/glam) for a simple but fast vector math library at the core.

# Examples
The [pathtracer example](https://github.com/ben-hansske/bsdf/blob/main/examples/pathtracer.rs) shows
you how to integrate this crate into a simple forward pathtracer. The code is kept very close to
[Raytracing in a Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html)
to make it easy to understand.


# References
A lot of pathtracing literature went into this. Here are the most influential papers and other
sources I have used:
* Brent Burley. Physically-based shading at Disney, course notes, revised 2014. In *ACM
    SIGGRAPH, Practical physically-based shading in film and game production,* 2012.
* Eric Heitz. Understanding the masking-shadowing function in microfacet-based brdfs.
    *Journal of Computer Graphics Techniques, 3(2):32–91,* 2014.
* Eric Veach. *Robust monte carlo methods for light transport simulation.* PhD thesis, Stanford University, 1997.
* Bruce Walter, Stephen R. Marschner, Hongsong Li, and Kenneth E. Torrance. Microfacet models for refraction through rough surfaces. In *Proceedings of the Eurographics Symposium on Rendering,* 2007.
* Eric Heitz, Sampling the GGX Distribution of Visible Normals, *Journal of Computer Graphics Techniques (JCGT)*, vol. 7, no. 4, 1–13, 2018
    <http://jcgt.org/published/0007/04/01/>
* Brent Burley. Extending the Disney BRDF to a BSDF with integrated subsurface scattering. *SIGGRAPH Course*, 19, 2015.
* Walt Disneys BRDF Explorer: <https://github.com/wdas/brdf/blob/main/src/brdfs/disney.brdf>
* Blenders Principled BSDF: <https://github.com/dfelinto/blender/blob/master/intern/cycles/kernel/osl/shaders/node_principled_bsdf.osl>
* PBRTs implementation of the Disney BSDF: <https://github.com/mmp/pbrt-v3/blob/master/src/materials/disney.cpp>
