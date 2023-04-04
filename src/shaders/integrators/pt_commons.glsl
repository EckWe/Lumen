#ifndef PT_COMMONS
#define PT_COMMONS


vec3 uniform_sample_light(const Material mat, vec3 pos, const bool side,
                          const vec3 n_s, const vec3 wo,
                          const bool is_specular) {
    vec3 res = vec3(0);
    // Sample light
    vec3 wi;
    float wi_len;
    float pdf_light_w;
    float pdf_light_a;
    LightRecord record;
    float cos_from_light;
    const vec3 Le =
        sample_light_Li_pdf_pos(seed, pos, pc_ray.num_lights, pdf_light_w, wi, wi_len,
                        pdf_light_a, cos_from_light, record);
    const vec3 p = offset_ray2(pos, n_s);
    float bsdf_pdf;
    float cos_x = dot(n_s, wi);
    const uint props = is_specular ? BSDF_ALL : BSDF_ALL & ~BSDF_SPECULAR;
    vec3 f = eval_bsdf(n_s, wo, mat, 1, side, wi, bsdf_pdf, cos_x);
    float pdf_light;
    any_hit_payload.hit = 1;
    traceRayEXT(tlas,
                gl_RayFlagsTerminateOnFirstHitEXT |
                    gl_RayFlagsSkipClosestHitShaderEXT,
                0xFF, 1, 0, 1, p, 0, wi, wi_len - EPS, 1);
    const bool visible = any_hit_payload.hit == 0;
    float old;
    if (visible && pdf_light_w > 0) {
        const float mis_weight =
            is_light_delta(record.flags) ? 1 : 1 / (1 + bsdf_pdf / pdf_light_w);
        res += mis_weight * f * abs(cos_x) * Le / pdf_light_w;
      
        old = mis_weight;
        
    }
    if (get_light_type(record.flags) == LIGHT_AREA) {
        // Sample BSDF
        f = sample_bsdf(n_s, wo, mat, 1, side, wi, bsdf_pdf, cos_x, seed);
        if (bsdf_pdf != 0) {
            traceRayEXT(tlas, flags, 0xFF, 0, 0, 0, p, tmin, wi, tmax, 0);
            if (payload.material_idx == record.material_idx &&
                payload.triangle_idx == record.triangle_idx) {
                const float wi_len = length(payload.pos - pos);
                const float g = abs(dot(payload.n_s, -wi)) / (wi_len * wi_len);
                const float mis_weight =
                    1. / (1 + pdf_light_a / (g * bsdf_pdf));
                res += f * mis_weight * abs(cos_x) * Le / bsdf_pdf;
               
            }
        }
    }
    return res;
}

vec3 uniform_sample_env_light(const vec2 rands_pos, out float pdf, out vec3 wi, out float wi_len) {

    // Uniform sample sphere to get direction - pbrt
	float z = 1.f - 2.f * rands_pos.x;
	float r = sqrt(max(0.f, 1.f - z * z));
	float phi = 2 * PI * rands_pos.y;
	wi = normalize(vec3(r * cos(phi), r * sin(phi), z));
	wi_len = 10000; // TODO check world radius or inf light
	pdf = 1.f / (2.f * PI2);
	vec2 uv = dir_to_latlong(wi);
	vec3 env_col = texture(scene_textures[pc_ray.num_textures],uv).xyz;
	

	/* if (gl_LaunchIDEXT.x == 300 && gl_LaunchIDEXT.y == 300) {
		debugPrintfEXT("%f, %f, %f .. %f, %f \n", rands_pos.x, rands_pos.y, wi.z, uv.x, uv.y);
	}*/

	return env_col;

}

// TODO double check even spherical direction distribution
// hardcoded for warped spherical environment map in resolution 4096x2048 - here 512x256 is used for importance sampling
/*vec3 importance_sample_env_light(const vec2 rands_pos, out float pdf, out vec3 wi, out float wi_len) {

	vec2 rnd = rands_pos;
	ivec2 pos = ivec2(0, 0);
	int tex_array_offset = pc_ray.num_textures;
	int top_mip_level = 9;

	// aspect ratio of envmap is 2:1, first decide left/right on level 1 (2 by 1 pixels)
	float left = texelFetch(scene_textures[tex_array_offset + top_mip_level - 1], pos, 0).x;
	float right = texelFetch(scene_textures[tex_array_offset + top_mip_level - 1], pos + ivec2(1, 0), 0).x;

	float left_n = left / (left + right);
	

	if (rnd.x < left_n) {
		rnd.x /= left_n;
	} else {
		pos.x += 1;
		rnd.x = (rnd.x - left_n) / (1.f - left_n);
	}

	// decide on 2x2 texel position from now on
	for (int mip = top_mip_level - 2; mip >= 0; mip--) {
		pos *= 2;

		vec4 v;
		v.x = texelFetch(scene_textures[tex_array_offset + mip], pos, 0).x;
		v.y = texelFetch(scene_textures[tex_array_offset + mip], pos + ivec2(1, 0), 0).x;
		v.z = texelFetch(scene_textures[tex_array_offset + mip], pos + ivec2(0, 1), 0).x;
		v.w = texelFetch(scene_textures[tex_array_offset + mip], pos + ivec2(1, 1), 0).x;

		left = v.x + v.z;
		right = v.y + v.w;

		left_n = left / (left + right);

		ivec2 offset;

		if (rnd.x < left_n) {
			offset.x = 0;
			rnd.x = rnd.x / left_n;
		} else {
			offset.x = 1;
			rnd.x = (rnd.x - left_n) / (1.f - left_n);
		}

		float upper_n = bool(offset.x) ? (v.y / right) : (v.x / left);

		if (rnd.y < upper_n) {
			offset.y = 0;
			rnd.y = rnd.y / upper_n;
		} else {
			offset.y = 1;
			rnd.y = (rnd.y - upper_n) / (1.f - upper_n);
		}

		pos += offset;
	}
	vec2 final_pos = pos + rnd;
	vec2 uv = final_pos / vec2(512, 256);

	wi = latlong_to_dir(uv);
	// TODO double check
	wi_len = 10000;
	float avg_intensity = texelFetch(scene_textures[tex_array_offset + top_mip_level], ivec2(0, 0), 0).x;
	pdf = texelFetch(scene_textures[tex_array_offset], pos, 0).x / avg_intensity;
	pdf = pdf * (1 / (2.f * PI2));

	vec3 result = texture(scene_textures[tex_array_offset - 1], uv).xyz;

	return result;
}

vec3 importance_sample_env_light_pdf(in vec3 dir, out float pdf) {

	vec2 uv = dir_to_latlong(dir);
	// TODO not hardcoded and try original res for mip map -> change at creation
	ivec2 pos = ivec2(uv * vec2(512, 256));
	int tex_array_offset = pc_ray.num_textures;
	float avg_intensity = texelFetch(scene_textures[tex_array_offset + top_mip_level], ivec2(0, 0), 0).x;
	pdf = texelFetch(scene_textures[tex_array_offset], pos, 0).x / avg_intensity;
	pdf = pdf * (1 / (2.f * PI2));

	vec3 result = texture(scene_textures[tex_array_offset - 1], uv).xyz;
	return result;
}*/


vec3 sample_env_light(const Material mat, vec3 pos, const bool side, const vec3 n_s, const vec3 wo,
						  const bool is_specular) {
	vec3 res = vec3(0);
	// Sample light
	vec3 wi;
	float wi_len;
	float pdf_light_w;
	vec2 samples = vec2(rand(seed), rand(seed));

	//float cos_from_light;
	const vec3 Le = //uniform_sample_env_light(samples, pos, pdf_light_w, wi, wi_len);
		importance_sample_env_light(samples, pdf_light_w, wi, wi_len, pc_ray.num_textures);

	const vec3 p = offset_ray2(pos, n_s);
	float bsdf_pdf;
	float cos_x = dot(n_s, wi);
	const uint props = is_specular ? BSDF_ALL : BSDF_ALL & ~BSDF_SPECULAR;
	vec3 f = eval_bsdf(n_s, wo, mat, 1, side, wi, bsdf_pdf, cos_x);

	any_hit_payload.hit = 1;
	traceRayEXT(tlas, gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT, 0xFF, 1, 0, 1, p, 0, wi,
				wi_len - EPS, 1);
	const bool visible = any_hit_payload.hit == 0;
	
	if (visible && pdf_light_w > 0) {
		const float mis_weight = 1;	 // is_light_delta(record.flags) ? 1 : 1 / (1 + bsdf_pdf / pdf_light_w);
		res += mis_weight * f * abs(cos_x) * Le / pdf_light_w;
	}
	
	return res;
}

bool path_trace(inout vec3 throughput, inout bool specular, inout vec3 direction, inout vec3 col, inout vec3 origin, int depth) {

    const vec3 hit_pos = payload.pos;

    const bool found_isect = payload.material_idx != -1;

	if (!found_isect) {
		vec2 uv = dir_to_latlong(direction);
		vec3 env_col = texture(scene_textures[pc_ray.num_textures], uv).xyz;
		col += throughput * env_col;  // pc_ray.sky_col;

		return false;
	}
	const Material hit_mat = load_material(payload.material_idx, payload.uv);
	if (depth == 0 || specular) {
		//col += throughput * hit_mat.emissive_factor;
		
	}
	const vec3 wo = -direction;
	vec3 n_s = payload.n_s;
	bool side = true;
	vec3 n_g = payload.n_g;
	if (dot(payload.n_g, wo) < 0.) 
        n_g = -n_g;
	if (dot(n_g, payload.n_s) < 0) {
		n_s = -n_s;
		side = false;
	}

    origin.xyz = offset_ray(payload.pos, n_g);
    

	if ((hit_mat.bsdf_props & BSDF_SPECULAR) == 0) {
		const float light_pick_pdf = 1. / pc_ray.light_triangle_count;
		//if (depth > 0)
		col += throughput * uniform_sample_light(hit_mat, payload.pos, side, n_s, wo, specular) / light_pick_pdf;
		//col += throughput * sample_env_light(hit_mat, payload.pos, side, n_s, wo, specular);
	}
	// TODO test self intersection offset fix
	//float refl_dir = dot(n_g, direction);
	//vec3 offset_n = n_g;
	//if (refl_dir < 0 && hit_mat.bsdf_type == BSDF_GLASS) 
    //    offset_n = -offset_n;
	
    

	// Sample direction & update throughput
	float pdf, cos_theta;
	const vec3 f = sample_bsdf(n_s, wo, hit_mat, 1 /*radiance=cam*/, side, direction, pdf, cos_theta, seed);

	/*float refl_dir = dot(n_g, direction);
	vec3 offset_n = n_g;
	if (refl_dir < 0 && hit_mat.bsdf_type == BSDF_GLASS)
	     offset_n = -offset_n;

	origin.xyz = offset_ray(hit_pos, offset_n);*/


	if (pdf == 0) {
		return false;
	}
	throughput *= f * abs(cos_theta) / pdf;
	specular = (hit_mat.bsdf_props & BSDF_SPECULAR) != 0;
	float rr_scale = 1.0;
	if (hit_mat.bsdf_type == BSDF_GLASS) {
		rr_scale *= side ? 1. / hit_mat.ior : hit_mat.ior;
	}
	if (depth > RR_MIN_DEPTH) {
		float rr_prob = min(0.95f, luminance(throughput) * rr_scale);
		if (rr_prob == 0 || rr_prob < rand(seed))
			return false;
		else
			throughput /= rr_prob;
	}
    

	return true;
}
#endif