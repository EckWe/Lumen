#pragma once
#include "Integrator.h"
class ReSTIRGI : public Integrator {
public:
	ReSTIRGI(LumenInstance* scene, LumenScene* lumen_scene) :
		Integrator(scene, lumen_scene) {}
	virtual void init() override;
	virtual void render() override;
	virtual bool update() override;
	virtual void destroy() override;
	virtual void reload() override;
private:
	Buffer restir_samples_buffer;
	Buffer restir_samples_old_buffer;
	Buffer temporal_reservoir_buffer;
	Buffer spatial_reservoir_buffer;
	Buffer tmp_col_buffer;
	void create_offscreen_resources();
	void create_descriptors();
	void create_blas();
	void create_tlas();
	void create_rt_pipelines();
	void create_compute_pipelines();
	virtual void update_uniform_buffers() override;
	PushConstantRay pc_ray{};
	VkDescriptorPool desc_pool;
	VkDescriptorSetLayout desc_set_layout;
	VkDescriptorSet desc_set;
	std::unique_ptr<Pipeline> rt_pipeline;
	std::unique_ptr<Pipeline> spatial_reuse_pipeline;
	std::unique_ptr<Pipeline> temporal_reuse_pipeline;
	std::unique_ptr<Pipeline> output_pipeline;
	bool do_spatiotemporal = false;

};

