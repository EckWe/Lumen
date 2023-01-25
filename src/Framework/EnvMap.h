#pragma once
#include "LumenPCH.h"
#include "Texture.h"
#include "LumenInstance.h"
#include <shaders/commons.h>

class EnvMap {
   public:
	EnvMap();	
	EnvMap(const std::string texturePath, LumenInstance* instance);
	Texture2D env_tex;
	std::vector<Texture2D> importance_mip_maps;
	LumenInstance* instance;
	void destroy();
	   
   private:
	int env_res_x;
	int env_res_y;

	
	VkSampler importance_sampler;
	VkSampler env_sampler;
	int map_factor = 8;

	
	void create_importance_map(const std::string texture_path);
		
};