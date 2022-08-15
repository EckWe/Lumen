#pragma once
#include "Integrator.h"
class VCM : public Integrator {
   public:
	VCM(LumenInstance* scene, LumenScene* lumen_scene) : Integrator(scene, lumen_scene) {}
	virtual void init() override;
	virtual void render() override;
	virtual bool update() override;
	virtual void destroy() override;

   private:
	void create_offscreen_resources();
	void create_descriptors();
	void create_blas();
	void create_tlas();
	// void create_rt_pipelines();
	// void create_compute_pipelines();
	PushConstantRay pc_ray{};
	VkDescriptorPool desc_pool;
	VkDescriptorSetLayout desc_set_layout;
	VkDescriptorSet desc_set;
	std::unique_ptr<Pipeline> vcm_light_pipeline;
	std::unique_ptr<Pipeline> vcm_eye_pipeline;
	std::unique_ptr<Pipeline> vcm_spawn_light_pipeline;
	std::unique_ptr<Pipeline> vcm_sample_pipeline;
	std::unique_ptr<Pipeline> init_reservoirs_pipeline;
	std::unique_ptr<Pipeline> select_reservoirs_pipeline;
	std::unique_ptr<Pipeline> update_reservoirs_pipeline;
	std::unique_ptr<Pipeline> check_reservoirs_pipeline;

	Buffer photon_buffer;
	Buffer vcm_light_vertices_buffer;
	Buffer light_path_cnt_buffer;
	Buffer color_storage_buffer;
	Buffer vcm_reservoir_buffer;
	Buffer light_samples_buffer;
	Buffer should_resample_buffer;
	Buffer light_state_buffer;
	Buffer angle_struct_buffer;
	Buffer angle_struct_cpu_buffer;
	Buffer avg_buffer;
	bool do_spatiotemporal = false;
	uint32_t total_frame_cnt = 0;
};
