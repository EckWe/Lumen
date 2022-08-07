#pragma once
#include "LumenPCH.h"
#include "Framework/CommandBuffer.h"
#include "Framework/Pipeline.h"
#include "Framework/Shader.h"
#include "Framework/Texture.h"
#include "Framework/EventPool.h"
#include "Framework/RenderGraphTypes.h"

struct AccelKHR {
	VkAccelerationStructureKHR accel = VK_NULL_HANDLE;
	Buffer buffer;
};

class RenderGraph;
class RenderPass;

enum class PassType {
	Compute,
	RT,
	Graphics
};


struct ResourceBinding {
	const Buffer* buf = nullptr;
	const Texture2D* tex = nullptr;
	VkSampler sampler = nullptr;
	ResourceBinding(const Buffer& buf) : buf(&buf) {}
	ResourceBinding(const Texture2D& tex) : tex(&tex) {}
	ResourceBinding(const Texture2D& tex, VkSampler sampler) : tex(&tex), sampler(sampler) {}

};



struct BufferSyncDescriptor {
	// Read-after-write is the default dependency implicitly
	VkAccessFlags src_access_flags = VK_ACCESS_SHADER_WRITE_BIT;
	VkAccessFlags dst_access_flags = VK_ACCESS_SHADER_READ_BIT;
	uint32_t opposing_pass_idx;
	VkEvent event = nullptr;
};

struct ImageSyncDescriptor {
	VkImageLayout old_layout;
	VkImageLayout new_layout;
	uint32_t opposing_pass_idx;
	VkImageAspectFlags image_aspect;
	VkEvent event = nullptr;
};

class RenderPass {
public:
	RenderPass(PassType type, Pipeline* pipeline, 
			   const std::string& name, RenderGraph* rg, uint32_t pass_idx, 
			   const GraphicsPassSettings& gfx_settings, bool cached = false) :
		type(type), pipeline(pipeline), name(name), rg(rg), pass_idx(pass_idx),
		gfx_settings(std::make_unique<GraphicsPassSettings>(gfx_settings)), cached(cached) {}

	RenderPass(PassType type, Pipeline* pipeline,
			   const std::string& name, RenderGraph* rg, uint32_t pass_idx,
			   const RTPassSettings& rt_settings, bool cached = false) :
		type(type), pipeline(pipeline), name(name), rg(rg), pass_idx(pass_idx),
		rt_settings(std::make_unique<RTPassSettings>(rt_settings)), cached(cached) {}

	RenderPass(PassType type, Pipeline* pipeline,
			   const std::string& name, RenderGraph* rg, uint32_t pass_idx,
			   const ComputePassSettings& compute_settings, bool cached = false) :
		type(type), pipeline(pipeline), name(name), rg(rg), pass_idx(pass_idx),
		compute_settings(std::make_unique<ComputePassSettings>(compute_settings)), cached(cached) {}

	RenderPass& bind(const ResourceBinding& binding);
	RenderPass& bind(const Texture2D& tex, VkSampler sampler);
	RenderPass& bind(std::initializer_list<ResourceBinding> bindings);
	RenderPass& bind_texture_array(const std::vector<Texture2D>& texes);
	RenderPass& bind_buffer_array(const std::vector<Buffer>& buffers);
	RenderPass& bind_tlas(const AccelKHR& tlas);
	RenderPass& read(Buffer& buffer);
	RenderPass& read(Texture2D& tex);
	RenderPass& read(std::initializer_list<std::reference_wrapper<Buffer>> buffers);
	RenderPass& read(std::initializer_list<std::reference_wrapper<Texture2D>> texes);
	RenderPass& write(Texture2D& tex);
	RenderPass& write(Buffer& buffer);
	RenderPass& write(Buffer& buffer, VkAccessFlags access_flags);
	RenderPass& write(std::initializer_list<std::reference_wrapper<Buffer>> buffers);
	RenderPass& write(std::initializer_list<std::reference_wrapper<Texture2D>> texes);

	RenderPass& skip_execution();
	RenderPass& push_constants(void* data);
	RenderPass& zero(Buffer& buffer);
	RenderPass& zero(std::initializer_list<std::reference_wrapper<Buffer>> buffers);
	RenderPass& zero(Buffer& buffer, bool cond);
	void finalize();
	friend RenderGraph;
private:

	void run(VkCommandBuffer cmd);
	void register_dependencies(Buffer& buffer, VkAccessFlags dst_access_flags);
	void register_dependencies(Texture2D& tex, VkImageLayout target_layout);

	std::string name;
	RenderGraph* rg;
	PassType type;
	Pipeline* pipeline;
	uint32_t pass_idx;
	std::vector<uint32_t> descriptor_counts;
	void* push_constant_data = nullptr;
	//struct {
	//	void* data = nullptr;
	//	uint32_t size = 0;
	//} push_constant;
	bool cached;

	/*
		Note:
		The assumption is that a SyncDescriptor is unique to a pass (either via Buffer or Image).
		Which is reasonable because each pass is comprised of a single shader dispatch
	*/
	std::unordered_map<VkBuffer, BufferSyncDescriptor> set_signals_buffer;
	std::unordered_map<VkBuffer, BufferSyncDescriptor> wait_signals_buffer;

	std::unordered_map<VkImage, ImageSyncDescriptor> set_signals_img;
	std::unordered_map<VkImage, ImageSyncDescriptor> wait_signals_img;

	std::vector<ResourceBinding> bound_resources;
	DescriptorInfo descriptor_infos[32] = {};

	std::vector<std::pair<Texture2D*, VkImageLayout>> layout_transitions;

	struct BufferBarrier {
		VkBuffer buffer;
		VkAccessFlags src_access_flags = VK_ACCESS_SHADER_WRITE_BIT;
		VkAccessFlags dst_access_flags = VK_ACCESS_SHADER_READ_BIT;
	};
	std::vector<Buffer*> buffer_zeros;
	std::vector<BufferBarrier> buffer_barriers;
	/*
		Potentially 1 descriptor pool for a pass where we have to keep the
		TLAS descriptor, because we can't push its descriptor with a template as of Vulkan 1.3
	*/
	VkDescriptorPool tlas_descriptor_pool = nullptr;
	VkDescriptorSet tlas_descriptor_set = nullptr;
	VkWriteDescriptorSetAccelerationStructureKHR tlas_info = {};

	std::unique_ptr<GraphicsPassSettings> gfx_settings = nullptr;
	std::unique_ptr<RTPassSettings> rt_settings = nullptr;
	std::unique_ptr<ComputePassSettings> compute_settings = nullptr;

	bool disable_execution = false;
};

class RenderGraph {

public:
	RenderGraph(VulkanContext* ctx) : ctx(ctx) { pipeline_tasks.reserve(32); }
	RenderPass& current_pass() { return passes.back(); }

	RenderPass& add_rt(const std::string& name, const RTPassSettings& settings);
	RenderPass& add_gfx(const std::string& name, const GraphicsPassSettings& settings);
	RenderPass& add_compute(const std::string& name, const ComputePassSettings& settings);
	RenderPass& get_current_pass();
	void run(VkCommandBuffer cmd);
	void reset(VkCommandBuffer cmd);
	friend RenderPass;
	bool recording = true;
	EventPool event_pool;
private:
	struct BufferSyncResources {
		std::vector<VkBufferMemoryBarrier2> buffer_bariers;
		std::vector<VkDependencyInfo> dependency_infos;
	};
	struct ImageSyncResources {
		std::vector<VkImageMemoryBarrier2> img_barriers;
		std::vector<VkDependencyInfo> dependency_infos;
	};

	struct PipelineStorage {
		std::unique_ptr<Pipeline> pipeline;
		uint32_t offset_idx; // Offset index relative to the pass_idx the pipeline holds
		std::vector<uint32_t> pass_idxs;
	};
	VulkanContext* ctx = nullptr;
	std::vector<RenderPass> passes;
	std::unordered_map<std::string, PipelineStorage> pipeline_cache;
	std::unordered_map<std::string, Shader> shader_cache;
	std::vector<std::pair<std::function<void(RenderPass*)>, uint32_t>> pipeline_tasks;
	// Sync related data
	std::vector<BufferSyncResources> buffer_sync_resources;
	std::vector<ImageSyncResources> img_sync_resources;
	std::unordered_map<VkBuffer, std::pair<uint32_t, VkAccessFlags>> buffer_resource_map; // Buffer handle - { Write Pass Idx, Access Type }
	std::unordered_map<VkImage, uint32_t> img_resource_map; // Tex2D handle - Pass Idx
};
