#ifndef SBDPT_COMMONS
#define SBDPT_COMMONS


vec3 vcm_connect_cam(const vec3 cam_pos, const vec3 cam_nrm, vec3 n_s,
                     const float cam_A, const vec3 pos, const in VCMState state,
                     const float eta_vm, const vec3 wo, const Material mat,
                     out ivec2 coords) {
    vec3 L = vec3(0);
    vec3 dir = cam_pos - pos;
    float len = length(dir);
    dir /= len;
    float cos_y = dot(dir, n_s);
    float cos_theta = dot(cam_nrm, -dir);
    if (cos_theta <= 0.) {
        return L;
    }

    // if(dot(n_s, dir) < 0) {
    //     n_s *= -1;
    // }
    // pdf_rev / pdf_fwd
    // in the case of light coming to camera
    // simplifies to abs(cos(theta)) / (A * cos^3(theta) * len^2)
    float cos_3_theta = cos_theta * cos_theta * cos_theta;
    const float cam_pdf_ratio = abs(cos_y) / (cam_A * cos_3_theta * len * len);
    vec3 ray_origin = offset_ray2(pos, n_s);
    float pdf_rev, pdf_fwd;
    const vec3 f = eval_bsdf(n_s, wo, mat, 0, dot(payload.n_s, wo) > 0, dir,
                             pdf_fwd, pdf_rev, cos_y);
    if (f == vec3(0)) {
        return L;
    }
    if (cam_pdf_ratio > 0.0) {
        any_hit_payload.hit = 1;
        traceRayEXT(tlas,
                    gl_RayFlagsTerminateOnFirstHitEXT |
                        gl_RayFlagsSkipClosestHitShaderEXT,
                    0xFF, 1, 0, 1, ray_origin, 0, dir, len - EPS, 1);
        if (any_hit_payload.hit == 0) {
            const float w_light = (cam_pdf_ratio / (screen_size)) *
                                  (eta_vm + state.d_vcm + pdf_rev * state.d_vc);

            const float mis_weight = 1. / (1. + w_light);
            // We / pdf_we * abs(cos_theta) = cam_pdf_ratio
            L = mis_weight * state.throughput * cam_pdf_ratio * f / screen_size;
            // if(isnan(luminance(L))) {
            //     debugPrintfEXT("%v3f\n", state.throughput);
            // }
        }
    }
    dir = -dir;
    vec4 target = ubo.view * vec4(dir.x, dir.y, dir.z, 0);
    target /= target.z;
    target = -ubo.projection * target;
    vec2 screen_dims = vec2(pc_ray.size_x, pc_ray.size_y);
    coords = ivec2(0.5 * (1 + target.xy) * screen_dims - 0.5);
    if (coords.x < 0 || coords.x >= pc_ray.size_x || coords.y < 0 ||
        coords.y >= pc_ray.size_y || dot(dir, cam_nrm) < 0) {
        return vec3(0);
    }
    return L;
}

// TODO test if pdf are right, check if we have to divide by lightPickProb
bool vcm_generate_light_sample(float eta_vc, out VCMState light_state,
                               out bool finite) {
    // Sample light
    uint light_idx;
    uint light_triangle_idx;
    uint light_material_idx;
    vec2 uv_unused;
    LightRecord light_record;
    vec3 wi, pos;
    float pdf_pos, pdf_dir, pdf_emit, pdf_direct;
    float cos_theta;

    const vec3 Le = sample_light_Le(
        seed, pc_ray.num_lights, pc_ray.light_triangle_count, cos_theta,
        light_record, pos, wi, pdf_pos, pdf_dir, pdf_emit, pdf_direct);

    if (pdf_dir <= 0) {
        return false;
    }
    light_state.pos = pos;
    light_state.area = 1.0 / pdf_pos;
    light_state.wi = wi;
    //light_state.throughput = Le * cos_theta / (pdf_dir * pdf_pos);
    light_state.throughput = Le * cos_theta / pdf_emit;

    // Partially evaluate pdfs (area formulation)
    // At s = 0 this is p_rev / p_fwd, in the case of area lights:
    // p_rev = p_connect = 1/area, p_fwd = cos_theta / (PI * area)
    // Note that pdf_fwd is in area formulation, so cos_y / r^2 is missing
    // currently.
    light_state.d_vcm = pdf_direct / pdf_emit;
    // g_prev / p_fwd
    // Note that g_prev component in d_vc and d_vm lags by 1 iter
    // So we initialize g_prev to cos_theta of the current iter
    // Also note that 1/r^2 in the geometry term cancels for vc and vm
    // By convention pdf_fwd sample the i'th vertex from i-1
    // g_prev or pdf_prev samples from i'th vertex to i-1
    // In that sense, cos_theta terms will be common in g_prev and pdf_pwd
    // Similar argument, with the eta
    finite = is_light_finite(light_record.flags);
    if (!is_light_delta(light_record.flags)) {
        light_state.d_vc = (finite ? cos_theta : 1) / pdf_emit;
    } else {
        light_state.d_vc = 0;
    }
    light_state.d_vm = light_state.d_vc * eta_vc;
    return true;
}

/*void init_light_spawn_reservoir(out LightSpawnReservoir lsr) {

    lsr.M = 0;
    lsr.W = 0.f;
    lsr.w_sum = 0.f;
    LightSpawnSample s;
    s.wi, s.pos, s.L_o = vec3(0);
    s.pdf_pos, s.pdf_dir, s.pdf_emit, s.pdf_direct = 0.f;
    s.is_delta, s.is_finite = false;
    lsr.light_spawn_sample = s;
}

bool update_light_reservoir(inout LightSpawnReservoir r, inout float w_sum, in LightSpawnSample s, in float w_i) {
    w_sum += w_i;
    r.M++;
    if (rand(seed) * w_sum <= w_i) {
        r.light_spawn_sample = s;
        return true;
    }
    return false;
}


bool generate_light_sample_and_temporal_resampling(float eta_vc, out VCMState light_state,
                               out bool finite, in VCMState camera_state) {
    // Sample light
    uint light_idx;
    uint light_triangle_idx;
    uint light_material_idx;
    vec2 uv_unused;
    LightRecord light_record;
    //vec3 wi, pos;
    //float pdf_pos, pdf_dir, pdf_emit, pdf_direct;
    //float cos_theta;

    LightSpawnSample light_spawn_sample;

    const vec3 Le = sample_light_Le(
        seed, pc_ray.num_lights, pc_ray.light_triangle_count, light_spawn_sample.cos_theta,
        light_record, light_spawn_sample.pos, light_spawn_sample.wi, light_spawn_sample.pdf_pos, light_spawn_sample.pdf_dir, light_spawn_sample.pdf_emit, light_spawn_sample.pdf_direct);


    light_spawn_sample.is_finite = is_light_finite(light_record.flags);
    light_spawn_sample.is_delta = is_light_delta(light_record.flags);

    //light.
    

    // Get light hitpos, from fill_light
    traceRayEXT(tlas, flags, 0xFF, 0, 0, 0, light_state.pos, tmin,
                    light_state.wi, tmax, 0);

    VCMVertex light_first_hit;
    bool generate_cam = true;
    // No intersection found
    if (payload.material_idx == -1) {
        //break; 
        generate_cam = false;
    }
    else {
        vec3 wo = light_state.pos - payload.pos;

        vec3 n_s = payload.n_s;
        vec3 n_g = payload.n_g;
        bool side = true;
        if (dot(payload.n_g, wo) <= 0.)
            n_g = -n_g;
        if (dot(n_g, n_s) < 0) {
            n_s = -n_s;
            side = false;
        }
        //TODO change into length(wo)
        float dist = length(payload.pos - light_state.pos);
        float dist_sqr = dist * dist;
        wo /= dist;
        const Material mat = load_material(payload.material_idx, payload.uv);
        const bool mat_specular =
            (mat.bsdf_props & BSDF_SPECULAR) == BSDF_SPECULAR;
        // Complete the missing geometry terms
        float cos_theta_wo = abs(dot(wo, n_s));

        // TODO use own reservoir struct
        light_first_hit.wo = wo;
        light_first_hit.n_s = n_s;
        light_first_hit.pos = payload.pos;
        light_first_hit.uv = payload.uv;
        light_first_hit.material_idx = payload.material_idx;
        light_first_hit.throughput = Le * light_spawn_sample.cos_theta;
        light_first_hit.side = uint(side);
        
    }
    
    // Get camera first hitpos 
    // TODO get first non specular hitpos
    traceRayEXT(tlas, flags, 0xFF, 0, 0, 0, camera_state.pos, tmin,
                    camera_state.wi, tmax, 0);

    light_spawn_sample.L_o = vec3(0);

    if (payload.material_idx == -1 || !generate_cam) { 
        // no intersection = no need to sample light
        
    } else {
        vec3 wo = camera_state.pos - payload.pos;
        float dist = length(payload.pos - camera_state.pos);
        float dist_sqr = dist * dist;
        wo /= dist;
        vec3 n_s = payload.n_s;
        vec3 n_g = payload.n_g;
        bool side = true;
        if (dot(payload.n_g, wo) < 0.)
            n_g = -n_g;
        if (dot(n_g, n_s) < 0) {
            n_s = -n_s;
            side = false;
        }
        const Material mat = load_material(payload.material_idx, payload.uv);
        // TODO handle specular, at the moment only test diffuse scene
        const bool mat_specular =
            (mat.bsdf_props & BSDF_SPECULAR) == BSDF_SPECULAR;
        //TODO CHANGE
        camera_state.pos = offset_ray(payload.pos, n_g);

        // from connect_light_vertices
        vec3 dir = light_first_hit.pos - payload.pos;
        const float len = length(dir);
        const float len_sqr = len * len;
        dir /= len;
        const float cos_cam = dot(n_s, dir);
        const float cos_light = dot(light_first_hit.n_s, -dir);
        const float G = cos_light * cos_cam / len_sqr;

        if (G > 0) {
            float cam_pdf_fwd, light_pdf_fwd, light_pdf_rev;
            const vec3 f_cam =
                eval_bsdf(n_s, wo, mat, 1, side, dir, cam_pdf_fwd, cos_cam);
            const Material light_mat =
                load_material(light_first_hit.material_idx,
                              light_first_hit.uv);

            const vec3 f_light =
                eval_bsdf(light_first_hit.n_s,
                          light_first_hit.wo, light_mat, 0,
                          light_first_hit.side == 1, -dir,
                          light_pdf_fwd, light_pdf_rev, cos_light);

            if (f_light != vec3(0) && f_cam != vec3(0)) {
                const vec3 ray_origin = offset_ray2(payload.pos, n_s);
                any_hit_payload.hit = 1;
                traceRayEXT(tlas,
                            gl_RayFlagsTerminateOnFirstHitEXT |
                                gl_RayFlagsSkipClosestHitShaderEXT,
                            0xFF, 1, 0, 1, ray_origin, 0, dir, len - EPS, 1);
                const bool visible = any_hit_payload.hit == 0;
                if (visible) {
                    light_spawn_sample.L_o = G * camera_state.throughput *
                          light_first_hit.throughput * f_cam *
                          f_light;
                }
            }
        }
    }
    
    LightSpawnReservoir lsr;

    // Resample according to res luminance
    if (pc_ray.do_spatiotemporal == 0) {
        //init reservoir
        init_light_spawn_reservoir(lsr);
    } else {
        lsr = light_spawn_reservoirs_temporal.d[pixel_idx];
    }

    float sample_tf = luminance(light_spawn_sample.L_o);
    float temporal_tf = luminance(lsr.light_spawn_sample.L_o);

    float wi = light_spawn_sample.pdf_emit <= 0.f ? 0.f : sample_tf / light_spawn_sample.pdf_emit;
    
    float w_sum = lsr.W * lsr.M * temporal_tf;

    bool reservoir_sample_changed = update_light_reservoir(lsr, w_sum, light_spawn_sample, wi);

    w_sum /= lsr.M;

    lsr.M = min(lsr.M, 50);
    float current_tf = luminance(lsr.light_spawn_sample.L_o);

    lsr.W = current_tf <= 0.f ? 0.f : w_sum / current_tf;
    // TODO Tiple check this, should this be lsr.W? But seems right atm
    // Test if 0 check needed
    lsr.light_spawn_sample.pdf_emit = lsr.W <= 0.f ? 0.f : 1.f / lsr.W;

    light_spawn_reservoirs_temporal.d[pixel_idx] = lsr;

    light_spawn_sample = lsr.light_spawn_sample;



    //TODO make unbiased with chance and adjust PDFs
    // chance to ignore resampling and take original sample = 10%
    // only needed when sample does not change
    if(!reservoir_sample_changed && rand(seed) < 0.1f) {
        
    }


    if (light_spawn_sample.pdf_dir <= 0) {
        return false; // TODO check: this should never happen?
    }
    
    light_state.pos = light_spawn_sample.pos;
    light_state.area = 1.0 / light_spawn_sample.pdf_pos;
    light_state.wi = light_spawn_sample.wi;
    // TODO when properly integrating do lsr_L_o * lsr.W
    light_state.throughput = light_first_hit.throughput * lsr.W;
    // pdf_emit was changed into 1/W here to normalize the random process
    light_state.d_vcm = light_spawn_sample.pdf_direct / light_spawn_sample.pdf_emit;
    
    finite = light_spawn_sample.is_finite;
    if (!light_spawn_sample.is_delta) {
        light_state.d_vc = (finite ? light_spawn_sample.cos_theta : 1) / light_spawn_sample.pdf_emit;
    } else {
        light_state.d_vc = 0;
    }
    light_state.d_vm = light_state.d_vc * eta_vc;
    return true;
}*/




vec3 vcm_get_light_radiance(in const Material mat,
                            in const VCMState camera_state, int d) {
    if (d == 1) {
        return mat.emissive_factor;
    }
    const float pdf_light_pos =
        1.0 / (payload.area * pc_ray.light_triangle_count);

    const float pdf_light_dir = abs(dot(payload.n_s, -camera_state.wi)) / PI;
    const float w_camera =
        pdf_light_pos * camera_state.d_vcm +
        (pc_ray.use_vc == 1 || pc_ray.use_vm == 1
             ? (pdf_light_pos * pdf_light_dir) * camera_state.d_vc
             : 0);
    const float mis_weight = 1. / (1. + w_camera);
    return mis_weight * mat.emissive_factor;
}

vec3 vcm_connect_light(vec3 n_s, vec3 wo, Material mat, bool side, float eta_vm,
                       VCMState camera_state, out float pdf_rev, out vec3 f) {
    vec3 wi;
    float wi_len;
    float pdf_pos_w;
    float pdf_pos_dir_w;
    LightRecord record;
    float cos_y;
    vec3 res = vec3(0);

    const vec3 Le =
        sample_light_Li(seed, payload.pos, pc_ray.num_lights, wi, wi_len,
                        pdf_pos_w, pdf_pos_dir_w, record, cos_y);

    const float cos_x = dot(wi, n_s);
    const vec3 ray_origin = offset_ray2(payload.pos, n_s);
    any_hit_payload.hit = 1;
    float pdf_fwd;
    f = eval_bsdf(n_s, wo, mat, 1, side, wi, pdf_fwd, pdf_rev, cos_x);
    if (f != vec3(0)) {
        traceRayEXT(tlas,
                    gl_RayFlagsTerminateOnFirstHitEXT |
                        gl_RayFlagsSkipClosestHitShaderEXT,
                    0xFF, 1, 0, 1, ray_origin, 0, wi, wi_len - EPS, 1);
        const bool visible = any_hit_payload.hit == 0;
        if (visible) {
            if (is_light_delta(record.flags)) {
                pdf_fwd = 0;
            }
            const float w_light =
                pdf_fwd / (pdf_pos_w / pc_ray.light_triangle_count);
            const float w_cam =
                pdf_pos_dir_w * abs(cos_x) / (pdf_pos_w * cos_y) *
                (eta_vm + camera_state.d_vcm + camera_state.d_vc * pdf_rev);
            const float mis_weight = 1. / (1. + w_light + w_cam);
            if (mis_weight > 0) {
                res = mis_weight * abs(cos_x) * f * camera_state.throughput *
                      Le / (pdf_pos_w / pc_ray.light_triangle_count);
            }
        }
    }
    return res;
}

#define light_vtx(i) vcm_lights.d[i]
vec3 vcm_connect_light_vertices(uint light_path_len, uint light_path_idx,
                                int depth, vec3 n_s, vec3 wo, Material mat,
                                bool side, float eta_vm, VCMState camera_state,
                                float pdf_rev) {
    vec3 res = vec3(0);
    for (int i = 0; i < light_path_len; i++) {
        uint s = light_vtx(light_path_idx + i).path_len;
        uint mdepth = s + depth - 1;
        if (mdepth >= pc_ray.max_depth) {
            break;
        }
        vec3 dir = light_vtx(light_path_idx + i).pos - payload.pos;
        const float len = length(dir);
        const float len_sqr = len * len;
        dir /= len;
        const float cos_cam = dot(n_s, dir);
        const float cos_light = dot(light_vtx(light_path_idx + i).n_s, -dir);
        const float G = cos_light * cos_cam / len_sqr;
        if (G > 0) {
            float cam_pdf_fwd, light_pdf_fwd, light_pdf_rev;
            const vec3 f_cam =
                eval_bsdf(n_s, wo, mat, 1, side, dir, cam_pdf_fwd, cos_cam);
            const Material light_mat =
                load_material(light_vtx(light_path_idx + i).material_idx,
                              light_vtx(light_path_idx + i).uv);
            // TODO: what about anisotropic BSDFS?
            const vec3 f_light =
                eval_bsdf(light_vtx(light_path_idx + i).n_s,
                          light_vtx(light_path_idx + i).wo, light_mat, 0,
                          light_vtx(light_path_idx + i).side == 1, -dir,
                          light_pdf_fwd, light_pdf_rev, cos_light);
            if (f_light != vec3(0) && f_cam != vec3(0)) {
                cam_pdf_fwd *= abs(cos_light) / len_sqr;
                light_pdf_fwd *= abs(cos_cam) / len_sqr;
                const float w_light =
                    cam_pdf_fwd *
                    (eta_vm + light_vtx(light_path_idx + i).d_vcm +
                     light_pdf_rev * light_vtx(light_path_idx + i).d_vc);
                const float w_camera =
                    light_pdf_fwd *
                    (eta_vm + camera_state.d_vcm + pdf_rev * camera_state.d_vc);
                const float mis_weight = 1. / (1 + w_camera + w_light);
                const vec3 ray_origin = offset_ray2(payload.pos, n_s);
                any_hit_payload.hit = 1;
                traceRayEXT(tlas,
                            gl_RayFlagsTerminateOnFirstHitEXT |
                                gl_RayFlagsSkipClosestHitShaderEXT,
                            0xFF, 1, 0, 1, ray_origin, 0, dir, len - EPS, 1);
                const bool visible = any_hit_payload.hit == 0;
                if (visible) {
                    res = mis_weight * G * camera_state.throughput *
                          light_vtx(light_path_idx + i).throughput * f_cam *
                          f_light;
                }
            }
        }
    }
    return res;
}
#undef light_vtx



void vcm_fill_light(vec3 origin, VCMState vcm_state, bool finite_light,
                     float eta_vcm, float eta_vc, float eta_vm) {
#define light_vtx(i) vcm_lights.d[vcm_light_path_idx + i]
    const vec3 cam_pos = origin;
    const vec3 cam_nrm = vec3(-ubo.inv_view * vec4(0, 0, 1, 0));
    const float radius = pc_ray.radius;
    const float radius_sqr = radius * radius;
    vec4 area_int = (ubo.inv_projection * vec4(2. / gl_LaunchSizeEXT.x,
                                               2. / gl_LaunchSizeEXT.y, 0, 1));
    area_int /= (area_int.w);
    const float cam_area = abs(area_int.x * area_int.y);
    int depth;
    int path_idx = 0;
    bool specular = false;
    light_vtx(path_idx).path_len = 0;
    for (depth = 1;; depth++) {
        traceRayEXT(tlas, flags, 0xFF, 0, 0, 0, vcm_state.pos, tmin,
                    vcm_state.wi, tmax, 0);
        if (payload.material_idx == -1) {
            break;
        }
        vec3 wo = vcm_state.pos - payload.pos;

        vec3 n_s = payload.n_s;
        vec3 n_g = payload.n_g;
        bool side = true;
        if (dot(payload.n_g, wo) <= 0.)
            n_g = -n_g;
        if (dot(n_g, n_s) < 0) {
            n_s = -n_s;
            side = false;
        }
        float cos_wo = dot(wo, n_s);
        float dist = length(payload.pos - vcm_state.pos);
        float dist_sqr = dist * dist;
        wo /= dist;
        const Material mat = load_material(payload.material_idx, payload.uv);
        const bool mat_specular =
            (mat.bsdf_props & BSDF_SPECULAR) == BSDF_SPECULAR;
        // Complete the missing geometry terms
        float cos_theta_wo = abs(dot(wo, n_s));

        if (depth > 1 || finite_light) {
            vcm_state.d_vcm *= dist_sqr;
        }
        vcm_state.d_vcm /= cos_theta_wo;
        vcm_state.d_vc /= cos_theta_wo;
        vcm_state.d_vm /= cos_theta_wo;
        if ((!mat_specular && (pc_ray.use_vc == 1 || pc_ray.use_vm == 1))) {

            // Copy to light vertex buffer
            // light_vtx(path_idx).wi = vcm_state.wi;
            light_vtx(path_idx).wo = wo; //-vcm_state.wi;
            light_vtx(path_idx).n_s = n_s;
            light_vtx(path_idx).pos = payload.pos;
            light_vtx(path_idx).uv = payload.uv;
            light_vtx(path_idx).material_idx = payload.material_idx;
            light_vtx(path_idx).area = payload.area;
            light_vtx(path_idx).throughput = vcm_state.throughput;
            light_vtx(path_idx).d_vcm = vcm_state.d_vcm;
            light_vtx(path_idx).d_vc = vcm_state.d_vc;
            light_vtx(path_idx).d_vm = vcm_state.d_vm;
            light_vtx(path_idx).path_len = depth + 1;
            light_vtx(path_idx).side = uint(side);
            path_idx++;
        }
        if (depth >= pc_ray.max_depth) {
            break;
        }
        // Reverse pdf in solid angle form, since we have geometry term
        // at the outer paranthesis
        if (!mat_specular && (pc_ray.use_vc == 1 && depth < pc_ray.max_depth)) {
            // Connect to camera
            ivec2 coords;
            vec3 splat_col =
                vcm_connect_cam(cam_pos, cam_nrm, n_s, cam_area, payload.pos,
                                vcm_state, eta_vm, wo, mat, coords);
			
            if (luminance(splat_col) > 0) {
                uint idx = coords.x * gl_LaunchSizeEXT.y +
                        coords.y;
                tmp_col.d[idx] += splat_col;
            }
//#endif
        }

        // Continue the walk
        float pdf_dir;
        float cos_theta;
		
        const vec3 f = sample_bsdf(n_s, wo, mat, 0, side, vcm_state.wi, pdf_dir,
                                   cos_theta, seed);

        const bool same_hemisphere = same_hemisphere(vcm_state.wi, wo, n_s);

        const bool mat_transmissive =
            (mat.bsdf_props & BSDF_TRANSMISSIVE) == BSDF_TRANSMISSIVE;
        if (f == vec3(0) || pdf_dir == 0 ||
            (!same_hemisphere && !mat_transmissive)) {
            break;
        }
        float pdf_rev = pdf_dir;
        if (!mat_specular) {
            pdf_rev = bsdf_pdf(mat, n_s, vcm_state.wi, wo);
        }
        const float abs_cos_theta = abs(cos_theta);

        vcm_state.pos = offset_ray(payload.pos, n_g);
        // Note, same cancellations also occur here from now on
        // see _vcm_generate_light_sample_
        if (!mat_specular) {
            vcm_state.d_vc =
                (abs_cos_theta / pdf_dir) *
                (eta_vm + vcm_state.d_vcm + pdf_rev * vcm_state.d_vc);
            vcm_state.d_vm =
                (abs_cos_theta / pdf_dir) *
                (1 + vcm_state.d_vcm * eta_vc + pdf_rev * vcm_state.d_vm);
            vcm_state.d_vcm = 1.0 / pdf_dir;
        } else {
            // Specular pdf has value = inf, so d_vcm = 0;
            vcm_state.d_vcm = 0;
            // pdf_fwd = pdf_rev = delta -> cancels
            vcm_state.d_vc *= abs_cos_theta;
            vcm_state.d_vm *= abs_cos_theta;
            specular = true;
        }
        vcm_state.throughput *= f * abs_cos_theta / pdf_dir;
        vcm_state.n_s = n_s;
        vcm_state.area = payload.area;
        vcm_state.material_idx = payload.material_idx;
    }
    light_path_cnts.d[pixel_idx] = path_idx;
}
#undef light_vtx

vec3 vcm_trace_eye(VCMState camera_state, float eta_vcm, float eta_vc,
                   float eta_vm) {

    float avg_len = 0;
    uint cnt = 1;
    const float radius = pc_ray.radius;
    const float radius_sqr = radius * radius;

    // TODO no randomization needed
    //uint light_path_idx = uint(rand(seed) * screen_size);
	uint light_path_idx = (gl_LaunchIDEXT.x * gl_LaunchSizeEXT.y + gl_LaunchIDEXT.y);
	//uint light_path_idx = vcm_light_path_idx;
    uint light_path_len = light_path_cnts.d[light_path_idx];
    light_path_idx *= (pc_ray.max_depth + 1);
    vec3 col = vec3(0);
    int depth;
    const float normalization_factor = 1. / (PI * radius_sqr * screen_size);

    for (depth = 1;; depth++) {
        traceRayEXT(tlas, flags, 0xFF, 0, 0, 0, camera_state.pos, tmin,
                    camera_state.wi, tmax, 0);

        if (payload.material_idx == -1) {
            // TODO:
            col += camera_state.throughput * pc_ray.sky_col;
            break;
        }
        vec3 wo = camera_state.pos - payload.pos;
        float dist = length(payload.pos - camera_state.pos);
        float dist_sqr = dist * dist;
        wo /= dist;
        vec3 n_s = payload.n_s;
        vec3 n_g = payload.n_g;
        bool side = true;
        if (dot(payload.n_g, wo) < 0.)
            n_g = -n_g;
        if (dot(n_g, n_s) < 0) {
            n_s = -n_s;
            side = false;
        }
        float cos_wo = abs(dot(wo, n_s));

        const Material mat = load_material(payload.material_idx, payload.uv);
        const bool mat_specular =
            (mat.bsdf_props & BSDF_SPECULAR) == BSDF_SPECULAR;
        // Complete the missing geometry terms
        camera_state.d_vcm *= dist_sqr;
        camera_state.d_vcm /= cos_wo;
        camera_state.d_vc /= cos_wo;
        camera_state.d_vm /= cos_wo;
        // Get the radiance
        if (luminance(mat.emissive_factor) > 0) {
            col += camera_state.throughput *
                   vcm_get_light_radiance(mat, camera_state, depth);
            // if (pc_ray.use_vc == 1 || pc_ray.use_vm == 1) {
            //     // break;
            // }
        }
        // Connect to light
        float pdf_rev;
        vec3 f;
        if (!mat_specular && depth < pc_ray.max_depth) {
            col += vcm_connect_light(n_s, wo, mat, side, eta_vm, camera_state,
                                     pdf_rev, f);
        }

        // Connect to light vertices
        if (!mat_specular) {
            col += vcm_connect_light_vertices(light_path_len, light_path_idx,
                                              depth, n_s, wo, mat, side, eta_vm,
                                              camera_state, pdf_rev);
        }

        if (depth >= pc_ray.max_depth) {
            break;
        }

        // Scattering
        float pdf_dir;
        float cos_theta;

        f = sample_bsdf(n_s, wo, mat, 1, side, camera_state.wi, pdf_dir,
                        cos_theta, seed);

        const bool mat_transmissive =
            (mat.bsdf_props & BSDF_TRANSMISSIVE) == BSDF_TRANSMISSIVE;
        const bool same_hemisphere = same_hemisphere(camera_state.wi, wo, n_s);
        if (f == vec3(0) || pdf_dir == 0 ||
            (!same_hemisphere && !mat_transmissive)) {
            break;
        }
        pdf_rev = pdf_dir;
        if (!mat_specular) {
            pdf_rev = bsdf_pdf(mat, n_s, camera_state.wi, wo);
        }
        const float abs_cos_theta = abs(cos_theta);

        camera_state.pos = offset_ray(payload.pos, n_g);
        // Note, same cancellations also occur here from now on
        // see _vcm_generate_light_sample_
        if (!mat_specular) {
            camera_state.d_vc =
                ((abs_cos_theta) / pdf_dir) *
                (eta_vm + camera_state.d_vcm + pdf_rev * camera_state.d_vc);
            camera_state.d_vm =
                ((abs_cos_theta) / pdf_dir) *
                (1 + camera_state.d_vcm * eta_vc + pdf_rev * camera_state.d_vm);
            camera_state.d_vcm = 1.0 / pdf_dir;
        } else {
            camera_state.d_vcm = 0;
            camera_state.d_vc *= abs_cos_theta;
            camera_state.d_vm *= abs_cos_theta;
        }

        camera_state.throughput *= f * abs_cos_theta / pdf_dir;
        camera_state.n_s = n_s;
        camera_state.area = payload.area;
        cnt++;
    }

#undef splat
    return col;
}
#endif