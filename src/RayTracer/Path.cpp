#include "LumenPCH.h"
#include "Path.h"

void Path::init() {
	Integrator::init();
	SceneDesc desc;
	desc.vertex_addr = vertex_buffer.get_device_address();
	desc.index_addr = index_buffer.get_device_address();
	desc.normal_addr = normal_buffer.get_device_address();
	desc.uv_addr = uv_buffer.get_device_address();
	desc.material_addr = materials_buffer.get_device_address();
	desc.prim_info_addr = prim_lookup_buffer.get_device_address();
	scene_desc_buffer.create("Scene Desc", &instance->vkb.ctx,
							 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
							 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_SHARING_MODE_EXCLUSIVE, sizeof(SceneDesc), &desc,
							 true);
	pc_ray.frame_num = 0;
	pc_ray.size_x = instance->width;
	pc_ray.size_y = instance->height;
}

void Path::render() {
	CommandBuffer cmd(&instance->vkb.ctx, /*start*/ true, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
	pc_ray.light_pos = scene_ubo.light_pos;
	pc_ray.light_type = 0;
	pc_ray.light_intensity = 10;
	pc_ray.num_lights = (int)lights.size();
	pc_ray.time = rand() % UINT_MAX;
	pc_ray.max_depth = lumen_scene->config.path_length;
	pc_ray.sky_col = lumen_scene->config.sky_col;
	
	pc_ray.total_light_area = total_light_area;
	pc_ray.light_triangle_count = total_light_triangle_cnt;
	pc_ray.num_textures = lumen_scene->textures.size();
	instance->vkb.rg
		->add_rt("Path", {.shaders = {{"src/shaders/integrators/path/path.rgen"},
									  {"src/shaders/ray.rmiss"},
									  {"src/shaders/ray_shadow.rmiss"},
									  {"src/shaders/ray.rchit"},
									  {"src/shaders/ray.rahit"}},
						  .dims = {instance->width, instance->height},
						  .accel = instance->vkb.tlas.accel})
		.push_constants(&pc_ray)
		.bind({
			output_tex,
			prim_lookup_buffer,
			scene_ubo_buffer,
			scene_desc_buffer,
		})
		
		.bind(mesh_lights_buffer)

		.bind_texture_array(diffuse_textures)
		//.write(output_tex) // Needed if the automatic shader inference is disabled
		.bind_tlas(instance->vkb.tlas);
		//.bind(env_map.env_tex);
	instance->vkb.rg->run_and_submit(cmd);

	pc_ray.total_frame_num++;
}

bool Path::update() {
	pc_ray.frame_num++;
	bool updated = Integrator::update();
	if (updated) {
		pc_ray.frame_num = 1;
	}
	return updated;
}

void Path::destroy() { Integrator::destroy(); }
