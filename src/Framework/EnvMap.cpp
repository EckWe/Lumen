#include "LumenPCH.h"
#include "EnvMap.h"
#include <stb_image.h>
#include "VkUtils.h"

#define LUMINANCE(r, g, b) (r * 0.2126f + g * 0.7152f + b * 0.0722f)

EnvMap::EnvMap() {}

EnvMap::EnvMap(const std::string texture_path, LumenInstance* instance) {
	
	this->instance = instance;

	VkSamplerCreateInfo sampler_ci = vk::sampler_create_info();
	sampler_ci.minFilter = VK_FILTER_NEAREST;
	sampler_ci.magFilter = VK_FILTER_NEAREST;
	sampler_ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
	sampler_ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	sampler_ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	sampler_ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	vk::check(vkCreateSampler(instance->vkb.ctx.device, &sampler_ci, nullptr, &env_sampler),
			  "Could not create image sampler");

	int x, y, n;
	float* data = stbi_loadf(texture_path.c_str(), &x, &y, &n, 4);
	auto size = x * y * 4 * 4;
	auto img_dims = VkExtent2D{(uint32_t)x, (uint32_t)y};
	auto ci = make_img2d_ci(img_dims, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT, false);
	env_tex.load_from_data(&instance->vkb.ctx, data, size, ci, env_sampler, false);


	/* int x, y, n;
	unsigned char* data = stbi_load(texture_path.c_str(), &x, &y, &n, 4);
	auto size = x * y * 4;
	auto img_dims = VkExtent2D{(uint32_t)x, (uint32_t)y};
	auto ci = make_img2d_ci(img_dims, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_USAGE_SAMPLED_BIT, false);
	env_tex.load_from_data(&instance->vkb.ctx, data, size, ci, env_sampler, false);*/

	env_res_x = x;
	env_res_y = y;
	create_importance_map(texture_path);

	stbi_image_free(data);
}

void EnvMap::destroy() { 
	//env_tex.destroy();
	for (int i = 0; i < importance_mip_maps.size(); i++) {
		//importance_mip_maps[i].destroy();
	}
	//vkDestroySampler(instance->vkb.ctx.device, env_sampler, nullptr);
	//vkDestroySampler(instance->vkb.ctx.device, importance_sampler, nullptr);
}

void EnvMap::create_importance_map(const std::string texture_path) {
	
	// TODO write importance map to texture and adjust image, make specific sampler

	assert((env_res_x % map_factor == 0) && env_res_y % (map_factor) == 0);



	int dim_x, dim_y, n;
	float* data = stbi_loadf(texture_path.c_str(), &dim_x, &dim_y, &n, 4);
	
	int importance_size_x = dim_x / map_factor;
	int importance_size_y = dim_y / map_factor;

	VkSamplerCreateInfo sampler_ci = vk::sampler_create_info();
	sampler_ci.minFilter = VK_FILTER_NEAREST;
	sampler_ci.magFilter = VK_FILTER_NEAREST;
	sampler_ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
	sampler_ci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	sampler_ci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	sampler_ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

	vk::check(vkCreateSampler(instance->vkb.ctx.device, &sampler_ci, nullptr, &importance_sampler),
			  "Could not create importance sampler");


	auto size = importance_size_x * importance_size_y * sizeof(float);  // size in byte
	auto img_dims = VkExtent2D{(uint32_t)importance_size_x, (uint32_t)importance_size_y};

	auto ci = make_img2d_ci(img_dims, VK_FORMAT_R32_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT, false);


	const int float_per_hdr_rgba = 4;

	// 4096x2048x4 floats to 512x256x1 floats
	float* importance_data = new float[importance_size_x * importance_size_y];

	for (int i = 0; i < importance_size_y; i++) {
		for (int j = 0; j < importance_size_x; j++) {

			int region_start_index = (j + i * dim_x) * map_factor; // index in element range
			float temp_imp = 0;

			for (int y = 0; y < map_factor; y++) {
				for (int x = 0; x < map_factor; x++) {
					
					int element_start_index = (region_start_index + x + y * dim_x) * float_per_hdr_rgba;
					float r = data[element_start_index];
					float g = data[element_start_index + 1];
					float b = data[element_start_index + 2];
					
					temp_imp += LUMINANCE(r, g, b);
				}
			}
			importance_data[j + importance_size_x * i] = temp_imp / (map_factor * map_factor);
		}
	}

	int mip_map_levels = calc_mip_levels(img_dims);

	importance_mip_maps = std::vector<Texture2D>(mip_map_levels);

	importance_mip_maps[0].load_from_data(&instance->vkb.ctx, importance_data, size, ci, importance_sampler, false);
	
	int prev_width = importance_size_x;
	importance_size_x = importance_size_x > 1 ? importance_size_x / 2 : 1;
	importance_size_y = importance_size_y > 1 ? importance_size_y / 2 : 1;

	std::vector<std::vector<float>> mips(mip_map_levels - 1);

	mips[0] = std::vector<float>(importance_size_x * importance_size_y);
	for (int y = 0; y < importance_size_y; y++) {
		for (int x = 0; x < importance_size_x; x++) {

			int region_start_index = 2*x + 2*y*prev_width;
			float temp_imp = importance_data[region_start_index];
			temp_imp += importance_data[region_start_index + 1];
			temp_imp += importance_data[region_start_index + prev_width];
			temp_imp += importance_data[region_start_index + prev_width + 1];
			mips[0][x + y * importance_size_x] = temp_imp / 4;
		}
	}

	size = importance_size_x * importance_size_y * sizeof(float);	// size in byte
	img_dims = VkExtent2D{(uint32_t)importance_size_x, (uint32_t)importance_size_y};
	ci = make_img2d_ci(img_dims, VK_FORMAT_R32_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT, false);
	importance_mip_maps[1].load_from_data(&instance->vkb.ctx, mips[0].data(), size, ci, importance_sampler, false);

	prev_width = importance_size_x;
	int prev_height = importance_size_y;
	importance_size_x = importance_size_x > 1 ? importance_size_x / 2 : 1;
	importance_size_y = importance_size_y > 1 ? importance_size_y / 2 : 1;			
	

	for (int i = 1; i < mips.size(); i++) {

		mips[i] = std::vector<float>(importance_size_x * importance_size_y);

		for (int y = 0; y < importance_size_y; y++) {
			for (int x = 0; x < importance_size_x; x++) {
				if (prev_width == 1) {
					int region_start_index = 2 * x + 2 * y * prev_width;
					float temp_imp = mips[i - 1][region_start_index];
					temp_imp += mips[i - 1][region_start_index + prev_width];
					mips[i][x + y * importance_size_x] = temp_imp / 2;
					
				}
				else if (prev_height == 1) {
					int region_start_index = 2 * x + 2 * y * prev_width;
					float temp_imp = mips[i - 1][region_start_index];
					temp_imp += mips[i - 1][region_start_index + 1];
					mips[i][x + y * importance_size_x] = temp_imp / 2;
					
				} else {
					int region_start_index = 2 * x + 2 * y * prev_width;
					float temp_imp = mips[i-1][region_start_index];
					temp_imp += mips[i - 1][region_start_index + 1];
					temp_imp += mips[i - 1][region_start_index + prev_width];
					temp_imp += mips[i - 1][region_start_index + prev_width + 1];
					mips[i][x + y * importance_size_x] = temp_imp / 4;
				}
			}
		}

		size = importance_size_x * importance_size_y * sizeof(float);  // size in byte
		img_dims = VkExtent2D{(uint32_t)importance_size_x, (uint32_t)importance_size_y};
		ci = make_img2d_ci(img_dims, VK_FORMAT_R32_SFLOAT, VK_IMAGE_USAGE_SAMPLED_BIT, false);
		importance_mip_maps[i+1].load_from_data(&instance->vkb.ctx, mips[i].data(), size, ci, importance_sampler, false);

		prev_width = importance_size_x;
		prev_height = importance_size_y;
		importance_size_x = importance_size_x > 1 ? importance_size_x / 2 : 1;
		importance_size_y = importance_size_y > 1 ? importance_size_y / 2 : 1;
	}

	
	stbi_image_free(data);
	delete[] importance_data;
	
}
