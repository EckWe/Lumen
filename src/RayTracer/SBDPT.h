#pragma once
#include "Integrator.h"
class SBDPT : public Integrator {
   public:
	SBDPT(LumenInstance* scene, LumenScene* lumen_scene) : Integrator(scene, lumen_scene) {}
	virtual void init() override;
	virtual void render() override;
	virtual bool update() override;
	virtual void destroy() override;

   private:
	PushConstantRay pc_ray{};
	VkDescriptorPool desc_pool;
	VkDescriptorSetLayout desc_set_layout;
	VkDescriptorSet desc_set;
	Buffer light_state_buffer;
	Buffer light_vertices_buffer;
	Buffer light_path_cnt_buffer;
	Buffer temporal_light_origin_reservoirs;
	Buffer color_storage_buffer;
	Buffer light_transfer_buffer;
	Buffer spatial_light_origin_reservoirs;
	Buffer light_vertices_reservoirs_buffer;
	Buffer light_path_reservoirs_buffer;

	bool do_spatiotemporal = false;
};
