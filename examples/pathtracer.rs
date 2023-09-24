// This is example is more or less just raytracing in a weekend
// It's a simple forward pathtracer with no advanced sampling techniques
// It's just to demonstrate how to use this library.
use bsdf::{disney::Disney, RgbD, RgbF, Vec3d, BSDF};
use glam::DMat3;

#[derive(Copy, Clone)]
struct Sphere {
    center: Vec3d,
    radius: f64,
}

#[derive(Copy, Clone)]
struct Ray {
    origin: Vec3d,
    direction: Vec3d,
}

#[derive(Copy, Clone)]
struct HitRecord {
    t: f64, // hit distance
    pos: Vec3d,
    normal: Vec3d,
}

impl Sphere {
    // simple shere ray intersection test
    fn hit(&self, ray: Ray, ray_tmin: f64, ray_tmax: f64) -> Option<HitRecord> {
        let oc = ray.origin - self.center;
        let a = ray.direction.length_squared();
        let half_b = Vec3d::dot(oc, ray.direction);
        let c = oc.length_squared() - self.radius * self.radius;

        let discriminant = half_b * half_b - a * c;
        if discriminant < 0.0 {
            return None;
        }
        let sqrtd = f64::sqrt(discriminant);

        // Find the nearest root that lies in the acceptable range.
        let mut root = (-half_b - sqrtd) / a;
        if root <= ray_tmin || ray_tmax <= root {
            root = (-half_b + sqrtd) / a;
            if root <= ray_tmin || ray_tmax <= root {
                return None;
            }
        }

        let pos = ray.origin + ray.direction * root;
        Some(HitRecord {
            t: root,
            pos,
            normal: (pos - self.center) / self.radius,
        })
    }
}

struct World {
    spheres: Vec<Sphere>,
    materials: Vec<Disney>,
}

impl From<Vec<((f64, f64, f64, f64), Disney)>> for World {
    fn from(value: Vec<((f64, f64, f64, f64), Disney)>) -> Self {
        let mut spheres = Vec::with_capacity(value.len());
        let mut materials = Vec::with_capacity(value.len());
        for ((x, y, z, radius), material) in value {
            spheres.push(Sphere {
                center: Vec3d { x, y, z },
                radius,
            });
            materials.push(material);
        }
        Self { spheres, materials }
    }
}

enum WorldHit<'t> {
    Surface {
        material: &'t Disney,
        pos: Vec3d,
        tangent_space: DMat3,
    },
    Background {
        color: RgbD,
    },
}

fn tangent_space(normal: Vec3d) -> DMat3 {
    let mut tan = Vec3d::new(0.0, 0.0, 1.0);
    if normal.dot(tan).abs() > 0.9999 {
        tan = Vec3d::new(0.0, 1.0, 0.0);
    }
    let bi = normal.cross(tan).normalize();
    let tan = bi.cross(normal).normalize();

    // we need the tan, bi and normal to be the rows of the matrix, therefore, we transpose.
    DMat3 {
        x_axis: tan,
        y_axis: bi,
        z_axis: normal,
        // }
    }
    .transpose()

    // that way, multiplying with tangent_space with omega_o becomes:
    // Vec3 {
    // x: omega_o.dot(tan),
    // y: omega_o.dot(bi),
    // z: omega_o.dot(normal)
    // }
}

impl World {
    fn find_hit(&self, ray: Ray, ray_tmin: f64, ray_tmax: f64) -> WorldHit {
        let mut hit = None;
        let mut closest_so_far = ray_tmax;
        let mut hit_id = std::usize::MAX;

        for (id, sphere) in self.spheres.iter().enumerate() {
            if let Some(current_hit) = sphere.hit(ray, ray_tmin, closest_so_far) {
                hit = Some(current_hit);
                closest_so_far = current_hit.t;
                hit_id = id;
            }
        }

        if let Some(hit) = hit {
            WorldHit::Surface {
                material: &self.materials[hit_id],
                pos: hit.pos,
                tangent_space: tangent_space(hit.normal),
            }
        } else {
            let unit_direction = ray.direction.normalize();
            let a = 0.5 * (unit_direction.y + 1.0);
            WorldHit::Background {
                // color: RgbD::new(0.7, 0.7, 0.9),
                color: (1.0 - a) * RgbD::new(1.0, 1.0, 1.0) + a * RgbD::new(0.5, 0.7, 1.0),
                // color: RgbD::ZERO
            }
        }
    }
}

fn random_walk(world: &World, mut ray: Ray, rd: &mut fastrand::Rng) -> RgbD {
    let mut accumulated = RgbD::ZERO;
    let mut factor = RgbD::ONE;

    // russian roulette
    let rr_delta = 0.1;
    #[allow(clippy::never_loop)]
    for depth in 0..50 {
        match world.find_hit(ray, 1e-5, std::f64::MAX) {
            WorldHit::Surface {
                material,
                pos,
                tangent_space,
            } => {
                // bring outgoing direction into local space
                let omega_o = (tangent_space * -ray.direction).normalize();
                let bsdf::SampleIncomingResponse {
                    omega_i,
                    bsdf,
                    emission,
                    pdf,
                } = material.sample_incoming(omega_o, Vec3d::new(rd.f64(), rd.f64(), rd.f64()));

                accumulated += factor * emission;

                // the cosine term is not part of the bsdf. Therefore it must be included here!
                let contrib_factor = bsdf * omega_i.z.abs() / pdf;
                // roussion roulette: always do 5 bounces, after that randomly terminate the path
                let rr_probab = if depth > 5 {
                    (contrib_factor.length() / rr_delta).clamp(0.0, 1.0)
                } else {
                    1.0
                };
                if rr_probab <= rd.f64() {
                    break;
                }
                factor *= contrib_factor / rr_probab;

                // transform incoming direction into global space
                // tangent_space is a pure rotation matrix, therefore its transposed is its
                // inverse
                ray = Ray {
                    origin: pos,
                    direction: (tangent_space.transpose() * omega_i).normalize(),
                };
            }
            WorldHit::Background { color } => {
                accumulated += factor * color;
                break;
            }
        }
    }
    accumulated
}

// I have no idea what the parameters of png mean, but i guess this works for now
fn save_image(path: &std::path::Path, buffer: &[u8], width: u32, height: u32) {
    let file = std::fs::File::create(path).unwrap();
    let mut writer = std::io::BufWriter::new(file);

    let mut encoder = png::Encoder::new(&mut writer, width, height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_source_gamma(png::ScaledFloat::new(1.0 / 2.2));

    let source_chromaticities = png::SourceChromaticities::new(
        // Using unscaled instantiation here
        (0.31270, 0.32900),
        (0.64000, 0.33000),
        (0.30000, 0.60000),
        (0.15000, 0.06000),
    );

    encoder.set_source_chromaticities(source_chromaticities);
    let mut writer = encoder.write_header().unwrap();

    writer.write_image_data(buffer).unwrap();
}

fn main() {
    let world: World = vec![
        (
            (0.0, 0.0, -1000.0, 1000.0),
            Disney {
                base_color: RgbF::ONE * 0.1,
                specular: 0.5,
                roughness: 0.3,
                ..Default::default()
            },
        ),
        (
            (0.6, 0.0, 0.5, 0.5),
            Disney {
                base_color: RgbF::new(0.8, 0.4, 0.1),
                roughness: 0.1,
                specular: 1.0,
                ..Default::default()
            },
        ),
        (
            (-0.3, -0.8, 0.3, 0.3),
            Disney {
                base_color: RgbF::ZERO,
                emission: RgbF::new(0.7, 0.7, 1.0) * 10.0,
                ..Default::default()
            },
        ),
        (
            (0.2, -1.3, 0.2, 0.2),
            Disney {
                base_color: RgbF::new(1.0, 1.0, 1.0),
                transmission: 1.0,
                ior: 1.45,
                roughness: 0.2,
                ..Default::default()
            },
        ),
        (
            (-1.3, 0.0, 0.3, 0.3),
            Disney {
                base_color: RgbF::new(0.6, 0.9, 0.8),
                metallic: 1.0,
                roughness: 0.4,
                anisotropic: 0.8,
                anisotropic_rotation: 0.25,
                ..Default::default()
            },
        ),
    ]
    .into();
    let image_size = (1920, 1080);
    let num_samples = 50;

    let cam_center = Vec3d::new(0.0, -5.0, 1.0);
    let cam_target = Vec3d::new(0.0, 0.0, 0.5);
    let forward = (cam_target - cam_center).normalize();
    let up = Vec3d::Z;
    // ensures that image is not distorted by image_size.0 and image_size.1 being
    // different
    let right = forward.cross(up).normalize() * 2.0 * image_size.0 as f64 / image_size.1 as f64;
    let up = -right.cross(forward).normalize() * 2.0;

    let mut image: Vec<u8> = vec![0; 3 * image_size.0 * image_size.1];

    let focal_length = 8.0;
    let forward = forward * focal_length;

    let mut rd = fastrand::Rng::new();

    for y in 0..image_size.1 {
        for x in 0..image_size.0 {
            let mut color = RgbD::ZERO;
            for _ in 0..num_samples {
                // from 0 to 1
                let uv_x = (x as f64 + rd.f64()) / image_size.0 as f64;
                let uv_y = (y as f64 + rd.f64()) / image_size.1 as f64;

                let cam_x = uv_x * 2.0 - 1.0;
                let cam_y = uv_y * 2.0 - 1.0;

                let direction = (forward + right * cam_x + up * cam_y).normalize();

                let ray = Ray {
                    origin: cam_center,
                    direction,
                };

                color += random_walk(&world, ray, &mut rd);
            }
            color /= num_samples as f64;

            let ri = (color.x * 255.0).clamp(0.0, 255.0).floor() as u8;
            let gi = (color.y * 255.0).clamp(0.0, 255.0).floor() as u8;
            let bi = (color.z * 255.0).clamp(0.0, 255.0).floor() as u8;

            image[(y * image_size.0 + x) * 3] = ri;
            image[(y * image_size.0 + x) * 3 + 1] = gi;
            image[(y * image_size.0 + x) * 3 + 2] = bi;
        }
        println!("Row {y} of {} rows finished.", image_size.1);
    }

    save_image(
        std::path::Path::new("image.png"),
        &image,
        image_size.0 as u32,
        image_size.1 as u32,
    );
}
