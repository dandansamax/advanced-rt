

// ----------------------------------------------------------------------------
// struct
// ----------------------------------------------------------------------------
struct Material {
  ambient: vec3<f32>
}
struct Ray {
  orig: vec3<f32>,
  dir: vec3<f32>,
  t_min: f32,
  t_max: f32
}
struct HitRecord {
    p: vec3<f32>,
    normal: vec3<f32>,
    t: f32,
    hit_material: Material,
    hit_found: bool  
}
struct Sphere {
  center: vec3<f32>,
  radius: f32,
  material: Material
}
struct Triangle {
    a: vec3<f32>,
    b: vec3<f32>,
    c: vec3<f32>,
    uv_a: vec2<f32>,
    uv_b: vec2<f32>,
    uv_c: vec2<f32>,
    material: Material
}
struct Cube {
  min: vec3<f32>,
  max: vec3<f32>,
  material: Material
} 
struct Cone {
  center: vec3<f32>, // base center
  radius: f32,
  height: f32,
  material: Material
} 
struct Camera {
  origin: vec3<f32>,
  a: vec3<f32>,
  b: vec3<f32>,
  w: vec3<f32>,
  u: vec3<f32>,
  v: vec3<f32>,
  lookat: vec3<f32>,
  dir: vec3<f32>,
  focus_length: f32,
}

struct LightPanel {
    corner_point: vec3<f32>,
    a: vec3<f32>,
    b: vec3<f32>,
    color: vec3<f32>
}

let pi: f32 = 3.1415926; 
let supersample_n: u32 = 5u; 
let N: u32 = 25u;

// ----------------------------------------------------------------------------
// global variables
// ----------------------------------------------------------------------------
var<private> camera: Camera;
var<private> light_panel: LightPanel;
var<private> Ka:f32 = 0.4;
var<private> Kd:f32 = 0.4; 
var<private> Ks:f32 = 0.2; 

var<private> pixel_position: vec2<f32>;
var<private> image_resolution: vec2<f32>;
var<private> backcolor: vec4<f32>;

var<private> seed: u32 = 0u;

var<private> light_points: array<vec3<f32>, N>;
var<private> lens_points: array<vec3<f32>, N>;
var<private> glossy_points: array<vec2<f32>, N>;
// world objects
var<private> world_spheres_count: i32 = 1;
var<private> world_spheres: array<Sphere, 1>;

var<private> world_triangles_count: i32 = 2;
var<private> world_triangles: array<Triangle, 2>;

var<private> world_cones_count: i32 = 1;
var<private> world_cones: array<Cone, 1>;


// world objects
var<private> world_cubes_count: i32 = 1;
var<private> world_cubes: array<Cube, 1>;

// Vertex shader

struct CameraUniform {
    view_proj: mat4x4<f32>,
};
struct Metainfo {
    resolution: vec2<u32>,
    theta: f32,
};

// Fragment shader

@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0)@binding(1)
var s_diffuse: sampler;

@group(1) @binding(0)
var<uniform> camerauniform: CameraUniform;

@group(2) @binding(0)
var<uniform> metainfo: Metainfo;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

//-----------------------
// math functions
//-----------------------
fn modulo(x: f32, y: f32) -> f32 {
    return x - (y * floor(x / y));
}

//-----------------------
// shading functions
//-----------------------
fn compute_diffuse(lightDir: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
  //Intensity of the diffuse light.
    var ndotL = max(dot(normal, lightDir), 0.0);
    return  light_panel.color * ndotL;
}

fn perturbing_dir(lightDir: vec3<f32>, index: u32) -> vec3<f32> {
    let w = normalize(lightDir);
    let u = normalize(cross(lightDir, vec3<f32>(0.0,0.1,0.0)));
    let v = cross(w,u);

    let delta = (-0.5 + u) * glossy_points[index].x + (-0.5 + v) * glossy_points[index].y;
    return normalize(lightDir + delta * 0.3);
}

fn compute_specular(viewDir: vec3<f32>, lightDir: vec3<f32>, normal: vec3<f32>, index: u32) -> vec3<f32> {
    let new_lightDir = perturbing_dir(lightDir, index);

    let phong_exponent = 32.0;
    // Specular
    let        V = normalize(-viewDir);
    let        R = reflect(-new_lightDir, normal);
    let      specular = pow(max(dot(V, R), 0.0), phong_exponent);
    return light_panel.color * specular;
}
fn get_checkerboard_texture_color(uv: vec2<f32>) -> vec3<f32> {
    var cols = 10.0;
    var rows = 10.0;
    var total = floor(uv.x * cols) + floor(uv.y * rows);
    if modulo(total, 2.0) == 0.0 {
        return vec3<f32>(0.0, 0.4, 0.0);
    } else {
        return vec3<f32>(0.8);
    }
}
fn get_background_color() -> vec3<f32> {
    return backcolor.xyz;
    // let t = pixel_position.y / image_resolution.y;
    // return t * vec3<f32>(0.2, 0.2, 0.2) + (1.0 - t) * vec3<f32>(1.0, 1.0, 1.0);
}
@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coords = model.tex_coords;
    // out.clip_position = camera.view_proj * vec4<f32>(model.position, 1.0);
    out.clip_position = vec4<f32>(model.position, 1.0);
    return out;
}

fn get_hit_point(ray: Ray, t: f32) -> vec3<f32> {
    return ray.orig + ray.dir * t;
}

fn sphere_intersection_test(ray: Ray, sphere: Sphere) -> HitRecord {
    var hit_rec: HitRecord;
    hit_rec.hit_found = false;
  //==================
    let oc = ray.orig - sphere.center;
    let a = dot(ray.dir, ray.dir);
    let half_b = dot(oc, ray.dir);
    let c = dot(oc, oc) - sphere.radius * sphere.radius;
    let discriminant = half_b * half_b - a * c;
    if discriminant < 0.0 {
        return hit_rec;
    }

    let sqrtd = sqrt(discriminant);
  // Find the nearest root that lies in the acceptable range.
    var root = (-half_b - sqrtd) / a;
    if (root < ray.t_min) || (ray.t_max < root) {
        root = (-half_b + sqrtd) / a;
        if root < ray.t_min || ray.t_max < root {
            return hit_rec;
        }
    }
    hit_rec.hit_found = true;
    hit_rec.hit_material = sphere.material;
    hit_rec.t = root;
    hit_rec.p = get_hit_point(ray, hit_rec.t);
    hit_rec.normal = normalize((hit_rec.p - sphere.center) / sphere.radius);
    if dot(ray.dir, hit_rec.normal) >= 0.0 {
        hit_rec.normal = hit_rec.normal * (-1.0);
    }
    return hit_rec;
}
fn triangle_intersection_test(ray: Ray, tri: Triangle) -> HitRecord {
    var hit_rec: HitRecord;
    hit_rec.hit_found = false;
  //==================

    var e1 = tri.b - tri.a;
    var e2 = tri.c - tri.a;
    var q = cross(ray.dir, e2);
    let a = dot(e1, q);
    
  // No hit found so far
    if a < ray.t_min {
        return hit_rec;
    }

    var f = 1.0 / a;
    var s = ray.orig - tri.a;
    var u = f * dot(s, q);

    if u < 0.0 || u > 1.0 {
        return hit_rec;
    }

    var rt = cross(s, e1);
    var v = f * dot(ray.dir, rt);

    if v < 0.0 || (u + v) > 1.0 {
        return hit_rec;
    }

    var w = (1.0 - u - v);

  // Hit found
    hit_rec.hit_found = true;
    hit_rec.t = f * dot(e2, rt);
    hit_rec.p = get_hit_point(ray, hit_rec.t);
    hit_rec.normal = normalize(cross(e1, e2));
    var uv = u * tri.uv_b + v * tri.uv_c + w * tri.uv_a;
    var material = tri.material;
    material.ambient = get_checkerboard_texture_color(uv);
    hit_rec.hit_material = material;
    return hit_rec;
}
fn cube_intersection_test(ray: Ray, cube: Cube) -> HitRecord {
    var hit_rec: HitRecord;
    hit_rec.hit_found = false;
  //==================
    let xn = vec3<f32>(1.0, 0.0, 0.0);
    let yn = vec3<f32>(0.0, 1.0, 0.0);
    let zn = vec3<f32>(0.0, 0.0, 1.0);

  // x -> yz-plane
    var tmin = (cube.min.x - ray.orig.x) / ray.dir.x;
    var tmax = (cube.max.x - ray.orig.x) / ray.dir.x;
    var normal = xn;
    if tmin > tmax {
      //swap(tmin, tmax); 
        let temp = tmin;
        tmin = tmax;
        tmax = temp;
    }

    // y -> xz-plane
    var tymin = (cube.min.y - ray.orig.y) / ray.dir.y;
    var tymax = (cube.max.y - ray.orig.y) / ray.dir.y;

    if tymin > tymax {
      //swap(tymin, tymax); 
        let temp = tymin;
        tymin = tymax;
        tymax = temp;
    }

    if (tmin > tymax) || (tymin > tmax) {
        return hit_rec; //No hit;
    }

    if tymin > tmin {
        tmin = tymin;
        normal = yn;
    }

    if tymax < tmax {
        tmax = tymax;
    } 
 
    // z -> xy-plane
    var tzmin = (cube.min.z - ray.orig.z) / ray.dir.z;
    var tzmax = (cube.max.z - ray.orig.z) / ray.dir.z;

    if tzmin > tzmax {
      //swap(tzmin, tzmax);
        let temp = tzmin;
        tzmin = tzmax;
        tzmax = temp;
    }

    if (tmin > tzmax) || (tzmin > tmax) {
        return hit_rec; //No hit;
    }

    if tzmin > tmin {
        tmin = tzmin;
        normal = zn;
    }

    if tzmax < tmax {
        tmax = tzmax;
    }
 
    // Hit found
    hit_rec.hit_found = true;
    hit_rec.t = tmin;
    hit_rec.p = get_hit_point(ray, tmin);
    hit_rec.hit_material = cube.material;
    hit_rec.normal = normalize(normal);
    return hit_rec;
}
fn cone_intersection_test(ray: Ray, cone: Cone) -> HitRecord {
    var hit_rec: HitRecord;
    hit_rec.hit_found = false;
  //==================

    let oc = ray.orig - cone.center;
    let d = cone.height - ray.orig.y + cone.center.y;

    let ratio = (cone.radius / cone.height) * (cone.radius / cone.height);

    let a = (ray.dir.x * ray.dir.x) + (ray.dir.z * ray.dir.z) - (ratio * (ray.dir.y * ray.dir.y));
    let b = (2.0 * oc.x * ray.dir.x) + (2.0 * oc.z * ray.dir.z) + (2.0 * ratio * d * ray.dir.y);
    let c = (oc.x * oc.x) + (oc.z * oc.z) - (ratio * (d * d));

    let delta = b * b - 4.0 * (a * c);
    if abs(delta) <= 0.0 {
        return hit_rec;   // No hit 
    }

    let t1 = (-b - sqrt(delta)) / (2.0 * a);
    let t2 = (-b + sqrt(delta)) / (2.0 * a);

    var t = t1;
    if t1 > t2 || t1 < 0.0 {
        t = t2;
    }
    let y = ray.orig.y + t * ray.dir.y;
    if !((y > cone.center.y) && (y < cone.center.y + cone.height)) {
        return hit_rec;   // No hit 
    }
  // Hit found
    hit_rec.hit_found = true;
    hit_rec.t = t;
    hit_rec.p = get_hit_point(ray, t);

    let r = sqrt((hit_rec.p.x - cone.center.x) * (hit_rec.p.x - cone.center.x) + (hit_rec.p.z - cone.center.z) * (hit_rec.p.z - cone.center.z));
    hit_rec.normal = normalize(vec3<f32>(hit_rec.p.x - cone.center.x, r * (cone.radius / cone.height), hit_rec.p.z - cone.center.z));
    hit_rec.hit_material = cone.material;
    return hit_rec;
}

fn trace_ray(ray: Ray) -> HitRecord {
    var hitWorld_rec: HitRecord;
    var hit_found = false;
    var closest_so_far = ray.t_max;
   //=========================
    for (var i: i32 = 0; i < world_spheres_count; i++) {
        let temp_rec: HitRecord = sphere_intersection_test(ray, world_spheres[i]);
        if temp_rec.hit_found {
            hit_found = true;
            if closest_so_far > temp_rec.t {
                closest_so_far = temp_rec.t;
                hitWorld_rec = temp_rec;
            }
        }
    }
  //============================
    for (var i: i32 = 0; i < world_cones_count; i++) {
        let temp_rec: HitRecord = cone_intersection_test(ray, world_cones[i]);
        if temp_rec.hit_found {
            hit_found = true;
            if closest_so_far > temp_rec.t {
                closest_so_far = temp_rec.t;
                hitWorld_rec = temp_rec;
            }
        }
    }
  //============================
    for (var i: i32 = 0; i < world_cubes_count; i++) {
        let temp_rec: HitRecord = cube_intersection_test(ray, world_cubes[i]);
        if temp_rec.hit_found {
            hit_found = true;
            if closest_so_far > temp_rec.t {
                closest_so_far = temp_rec.t;
                hitWorld_rec = temp_rec;
            }
        }
    }
  //============================
    for (var i: i32 = 0; i < world_triangles_count; i++) {
        let temp_rec: HitRecord = triangle_intersection_test(ray, world_triangles[i]);
        if temp_rec.hit_found {
            hit_found = true;
            if closest_so_far > temp_rec.t {
                closest_so_far = temp_rec.t;
                hitWorld_rec = temp_rec;
            }
        }
    }
   //=========================
    hitWorld_rec.hit_found = hit_found;
    return hitWorld_rec;
}

fn compute_shading(ray: Ray, rec: HitRecord, index: u32) -> vec3<f32> {
    // ambient
    let ambient = rec.hit_material.ambient;

    // diffuse
    var lightDir = light_points[index] - rec.p;
    let lightDistance = length(lightDir);
    lightDir = normalize(lightDir);
    let diffuse = compute_diffuse(lightDir, rec.normal);
    var specular = vec3<f32>(0.0, 0.0, 0.0);
    var attenuation = 1.0;
    // Tracing shadow ray only if the light is visible from the surface
    if dot(rec.normal, lightDir) > 0.0 {
        var shadow_ray: Ray;
        shadow_ray.orig = rec.p;
        shadow_ray.dir = lightDir;
        shadow_ray.t_min = 0.001;
        shadow_ray.t_max = lightDistance - shadow_ray.t_min;
        var shadow_rec = trace_ray(shadow_ray);
        if shadow_rec.hit_found {
            attenuation = 0.3;
        } else {
         // Compute specular only if not in shadow
            specular = compute_specular(ray.dir, lightDir, rec.normal, index);
        }
    }
    return ambient * Ka + (diffuse * Kd + specular * Ks) * attenuation;
}
 
fn cal_spherical_position(pos: vec3<f32>) -> vec2<f32>{
    let x = (pi + atan2(pos.y, pos.x)) / (2.0 * pi);
    let y = (pi - acos(pos.z / length(pos))) / pi; 
    return vec2<f32>(x,y);
}

fn get_face_coor(face: i32, coor: vec2<f32>) ->vec2<f32> {
    var c = vec2<f32>(coor.x / 4.0, coor.y / 3.0);
    var start: vec2<f32>;
    if face == 0 {
        start = vec2<f32>(2.0 / 4.0, 1.0 / 3.0);
    }
    else if face == 1 {
        start = vec2<f32>(0.0 / 4.0, 1.0 / 3.0);
    }
    else if face == 2 {
        start = vec2<f32>(1.0 / 4.0, 2.0 / 3.0);
    }
    else if face == 3 {
        start = vec2<f32>(1.0 / 4.0, 0.0 / 3.0);
    }
    else if face == 4 {
        start = vec2<f32>(1.0 / 4.0, 1.0 / 3.0);
    }
    else if face == 5 {
        start = vec2<f32>(3.0 / 4.0, 1.0 / 3.0);
    }
    return start + c;
}

fn cal_cube_position(pos: vec3<f32>) -> vec2<f32>{
    var face: i32;
    var coor: vec2<f32>;
    if abs(pos.x)>abs(pos.y)&&abs(pos.x)>abs(pos.z){
        if pos.x > 0.0 {
            face = 0;
            coor = vec2<f32>((1.0 - pos.z / abs(pos.x)) / 2.0, (1.0 + pos.y / abs(pos.x)) / 2.0); 
        }
        else{
            face = 1;
            coor = vec2<f32>((1.0 + pos.z / abs(pos.x)) / 2.0, (1.0 + pos.y / abs(pos.x)) / 2.0); 
        }
    }
    else if abs(pos.y)>abs(pos.z){
        if pos.y > 0.0 {
            face = 2;
            coor = vec2<f32>((1.0 + pos.x / abs(pos.y)) / 2.0, (1.0 - pos.z / abs(pos.y)) / 2.0); 
        }
        else{
            face = 3;
            coor = vec2<f32>((1.0 + pos.x / abs(pos.y)) / 2.0, (1.0 + pos.z / abs(pos.y)) / 2.0); 
        }
    }
    else {
        if pos.z > 0.0 {
            face = 4;
            coor = vec2<f32>((1.0 + pos.x / abs(pos.z)) / 2.0, (1.0 + pos.y / abs(pos.z)) / 2.0); 
        }
        else{
            face = 5;
            coor = vec2<f32>((1.0 - pos.x / abs(pos.z)) / 2.0, (1.0 + pos.y / abs(pos.z)) / 2.0); 
        }
    }
    return get_face_coor(face, coor);
}

// Trace ray and return the resulting contribution of this ray
fn get_sample_color(ray: Ray, index: u32) -> vec3<f32> {
    var final_pixel_color = vec3<f32>(0.0, 0.0, 0.0);
    var rec = trace_ray(ray);
    // let texture_position = cal_spherical_position(ray.dir);
    let texture_position = cal_cube_position(ray.dir);
    backcolor = textureSample(t_diffuse, s_diffuse, texture_position);
    if !rec.hit_found // if hit background  
    {
        final_pixel_color = get_background_color();
    } else {
        final_pixel_color = compute_shading(ray, rec, index);
    }
    // final_pixel_color = get_background_color();
    return final_pixel_color;
}

fn setup_light() {
    // light.position = vec3<f32>(3.0, 3.0, 1.0);
    // light.color = vec3<f32>(1.0, 1.0, 1.0);
    light_panel.corner_point = vec3<f32>(3.0, 3.0, 1.0);
    light_panel.a = vec3<f32>(0.5, 0.0, 0.0);
    light_panel.b = vec3<f32>(0.0, 0.5, 0.0);
    light_panel.color = vec3<f32>(1.0, 1.0, 1.0);
}
fn setup_camera() {

  // Rotate the camera around
    var look_from = vec3<f32>(0.0, 1.0, 1.0);
    var theta = metainfo.theta;
    var rot = mat3x3<f32>(cos(theta), 0., -sin(theta), 0., 1., 0., -sin(theta), 0., cos(theta));

    camera.origin = rot * look_from;
    camera.lookat = vec3<f32>(0.0, 0.0, 0.0);
    camera.dir = normalize(camera.lookat - camera.origin);

    camera.focus_length = length(camera.lookat - camera.origin);

    camera.w = normalize(camera.dir * (-1.0));
    camera.u = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), camera.w));
    camera.v = cross(camera.w, camera.u);
    
    camera.a = camera.u * 0.00;
    camera.b = camera.v * 0.00;
}
fn setup_scene_objects() {
  
// ----------------------------------------------------------------------------
// Scene definition
// ----------------------------------------------------------------------------
  // -- Sphere[0] -- 
    world_spheres[0].center = vec3<f32>(-0.125, 0.25, 0.35);
    world_spheres[0].radius = 0.25;
    world_spheres[0].material.ambient = vec3<f32>(0.7, 0.0, 0.0);

  // -- cone[0] -- 
    world_cones[0].center = vec3<f32>(0.18, 0.0, -0.12);
    world_cones[0].radius = 0.25;
    world_cones[0].height = 0.75;
    world_cones[0].material.ambient = vec3<f32>(0.0, 0.4, 0.7);

  // -- cube[0] -- 
  //world_cubes[0].min=vec3<f32>( -1.0,  0.0, -1.0);
  //world_cubes[0].max=vec3<f32>(  1.0,  0.9, -1.05);
  //world_cubes[0].material.ambient=vec3<f32>(0.7,0.4,0.7);

  // -- Triangle[0] -- 
    world_triangles[0].a = vec3<f32>(-2.0, 0.0, -2.0);
    world_triangles[0].b = vec3<f32>(-2.0, 0.0, 2.0);
    world_triangles[0].c = vec3<f32>(2.0, 0.0, -2.0);
    world_triangles[0].uv_a = vec2<f32>(0.0, 0.0);
    world_triangles[0].uv_b = vec2<f32>(1.0, 0.0);
    world_triangles[0].uv_c = vec2<f32>(0.0, 1.0);
    world_triangles[0].material.ambient = vec3<f32>(0.0, 0.0, 0.0);

  // -- Triangle[1] -- 
    world_triangles[1].a = vec3<f32>(-2.0, 0.0, 2.0);
    world_triangles[1].b = vec3<f32>(2.0, 0.0, 2.0);
    world_triangles[1].c = vec3<f32>(2.0, 0.0, -2.0);
    world_triangles[1].uv_a = vec2<f32>(1.0, 0.0);
    world_triangles[1].uv_b = vec2<f32>(1.0, 1.0);
    world_triangles[1].uv_c = vec2<f32>(0.0, 1.0);
    world_triangles[1].material.ambient = vec3<f32>(0.0, 0.0, 0.0);
}
fn get_ray(camera: Camera, ui: f32, vj: f32, index: u32) -> Ray {
    var ray: Ray;
    ray.orig = lens_points[index];
    ray.dir = normalize(camera.focus_length * ((camera.w * (-1.0)) + (camera.u * ui) + (camera.v * vj)) + camera.origin - ray.orig);
    ray.t_min = 0.0;
    ray.t_max = 10000.0;
    return ray;
}

fn tea(val0:u32, val1:u32)->u32{
// "GPU Random Numbers via the Tiny Encryption Algorithm"
  var v0 = val0;
  var v1 = val1;
  var s0 = u32(0);
  for (var n: i32 = 0; n < 16; n++) {
    s0 += 0x9e3779b9u;
    v0 += ((v1 << 4u) + 0xa341316cu) ^ (v1 + s0) ^ ((v1 >> 5u) + 0xc8013ea4u);
    v1 += ((v0 << 4u) + 0xad90777du) ^ (v0 + s0) ^ ((v0 >> 5u) + 0x7e95761eu);
  }

  return v0;
}

fn lcg() -> u32{
// Generate a random unsigned int in [0, 2^24) given the previous RNG state
// using the Numerical Recipes linear congruential generator
  let LCG_A = 1664525u;
  let LCG_C = 1013904223u;
  seed = (LCG_A * seed + LCG_C);
  return seed & 0x00FFFFFFu;
}

fn rnd() -> f32{
  // Generate a random float in [0, 1) 
  return (f32(lcg()) / f32(0x01000000));
}

fn generate_lights() {
    
    for (var i = 0u; i < supersample_n; i++) {
        for (var j = 0u; j < supersample_n; j++) {
            let ui = (f32(i) + rnd()) / f32(supersample_n);
            let vi = (f32(j) + rnd()) / f32(supersample_n);
            light_points[i*supersample_n+j] = ui * light_panel.a + vi * light_panel.b + light_panel.corner_point;
        }
    }

    // shuffle
    for (var i = N - 1u; i >= 1u; i--){
        let j = u32(floor(rnd()*f32(i + 1u)) + 0.00001);
        let tmp = light_points[i];
        light_points[j] = light_points[i];
        light_points[i] = tmp;
    }
}

fn generate_lens() {
    
    for (var i = 0u; i < supersample_n; i++) {
        for (var j = 0u; j < supersample_n; j++) {
            let ui = (f32(i) + rnd()) / f32(supersample_n);
            let vi = (f32(j) + rnd()) / f32(supersample_n);
            lens_points[i*supersample_n+j] = (-0.5 + ui) * camera.a + (-0.5 + vi) * camera.b + camera.origin;
        }
    }

    // shuffle
    for (var i = N - 1u; i >= 1u; i--){
        let j = u32(floor(rnd()*f32(i + 1u)) + 0.00001);
        let tmp = lens_points[i];
        lens_points[j] = lens_points[i];
        lens_points[i] = tmp;
    }
}

fn generate_glossy() {
    
    for (var i = 0u; i < supersample_n; i++) {
        for (var j = 0u; j < supersample_n; j++) {
            let ui = (f32(i) + rnd()) / f32(supersample_n);
            let vi = (f32(j) + rnd()) / f32(supersample_n);
            glossy_points[i*supersample_n+j] = vec2<f32>(ui, vi);
        }
    }

    // shuffle
    for (var i = N - 1u; i >= 1u; i--){
        let j = u32(floor(rnd()*f32(i + 1u)) + 0.00001);
        let tmp = glossy_points[i];
        glossy_points[j] = glossy_points[i];
        glossy_points[i] = tmp;
    }
}
fn get_pixel_color() -> vec3<f32> {
  // setup scene
    let top = 0.88;
    let right = (image_resolution.x / image_resolution.y) * top;
    let left = right * (-1.0);
    let bottom = top * (-1.0);

    generate_lights();
    generate_lens();
    generate_glossy();

    var pixel_color = vec3<f32>(0.0, 0.0, 0.0);

    for (var i = 0u; i < supersample_n; i++) {
        for (var j = 0u; j < supersample_n; j++) {
            let ui = left + (right - left) * ((pixel_position.x + (f32(i) + rnd()) / f32(supersample_n)) / image_resolution.x);
            let vi = bottom + (top - bottom) * ((pixel_position.y + (f32(j) + rnd()) / f32(supersample_n)) / image_resolution.y);
            let index =  i * supersample_n + j;
            var ray: Ray = get_ray(camera, ui, vi, index);
            pixel_color = pixel_color + get_sample_color(ray, index);
        }
    }
    pixel_color = pixel_color / f32(supersample_n * supersample_n);
    return pixel_color;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    image_resolution = vec2<f32>(metainfo.resolution);
    let position = in.clip_position.xyz;
    pixel_position = vec2<f32>(position.x, (image_resolution.y - position.y));

    let v= 12u;
    seed = tea(u32(position.x) + u32(position.y) * metainfo.resolution.x, v);

    setup_camera();
    setup_light();
    setup_scene_objects();

    let pixel_color = get_pixel_color();

    return vec4<f32>(pixel_color, 1.0);
}
