#include "VKBase.h"

VKBase::VKBase(int width, int height, bool fullscreen, bool debug) {
	this->width = width;
	this->height = height;
	this->fullscreen = fullscreen;
	this->enable_validation_layers = debug;
	create_instance();
}

std::vector<const char*> VKBase::get_req_extensions() {
	uint32_t glfwExtensionCount = 0;
	const char** glfwExtensions;
	glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

	std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

	if (enable_validation_layers) {
		extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	}

	return extensions;
}

VKBase::QueueFamilyIndices VKBase::find_queue_families(VkPhysicalDevice device) {

	QueueFamilyIndices indices;
	uint32_t queue_family_count = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);
	std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
	vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.data());

	int i = 0;
	for (const auto& queueFamily : queue_families) {
		if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
			indices.gfx_family = i;
		}

		VkBool32 presentSupport = false;
		vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

		if (presentSupport) {
			indices.present_family = i;
		}

		if (indices.is_complete()) {
			break;
		}

		i++;
	}

	return indices;
}

VKBase::SwapChainSupportDetails VKBase::query_swapchain_support(VkPhysicalDevice device) {
	// Basically returns present modes and surface modes in a struct

	SwapChainSupportDetails details;

	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

	uint32_t format_cnt;
	vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_cnt, nullptr);

	if (format_cnt != 0) {
		details.formats.resize(format_cnt);
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_cnt, details.formats.data());
	}

	uint32_t present_mode_cnt;
	vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &present_mode_cnt, nullptr);

	if (present_mode_cnt != 0) {
		details.present_modes.resize(present_mode_cnt);
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &present_mode_cnt, details.present_modes.data());
	}
	return details;
}

std::vector<char> VKBase::read_file(const std::string& filename) {
	{
		std::ifstream file(filename, std::ios::ate | std::ios::binary);

		if (!file.is_open()) {
			throw std::runtime_error("Failed to open file");
		}

		size_t fileSize = (size_t)file.tellg();
		std::vector<char> buffer(fileSize);

		file.seekg(0);
		file.read(buffer.data(), fileSize);

		file.close();

		return buffer;
	}
}

VkShaderModule VKBase::create_shader(const std::vector<char>& code) {
		VkShaderModuleCreateInfo shader_module_CI{};
		shader_module_CI.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		shader_module_CI.codeSize = code.size();
		shader_module_CI.pCode = reinterpret_cast<const uint32_t*>(code.data());

		VkShaderModule shaderModule;
		if (vkCreateShaderModule(device, &shader_module_CI, nullptr, &shaderModule) != VK_SUCCESS) {
			throw std::runtime_error("failed to create shader module!");
		}

		return shaderModule;
	}


VkResult VKBase::vkExt_create_debug_messenger(
	VkInstance instance,
	const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
	const VkAllocationCallbacks* pAllocator,
	VkDebugUtilsMessengerEXT* pDebugMessenger) {

	auto func = (PFN_vkCreateDebugUtilsMessengerEXT)
		vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");

	if (func != nullptr) {
		return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
	} else {
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}

 void VKBase::vkExt_destroy_debug_messenger(VkInstance instance, 
	VkDebugUtilsMessengerEXT debug_messenger,
	const VkAllocationCallbacks* pAllocator) {
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)
		vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
	if (func != nullptr) {
		func(instance, debug_messenger, pAllocator);
	}


}

static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
	VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
	VkDebugUtilsMessageTypeFlagsEXT messageType,
	const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
	void* pUserData) {

	std::cerr << "Validation layer: " << pCallbackData->pMessage << std::endl;

	return VK_FALSE;
}

void VKBase::setup_debug_messenger() {


	auto ci = vks::debug_messenger_CI(debug_callback);


	if (vkExt_create_debug_messenger(instance, &ci, nullptr, &debug_messenger) != VK_SUCCESS) {
		throw std::runtime_error("Failed to set up debug messenger!");
	}


}

void VKBase::init_vulkan() {

	create_instance();
	if (enable_validation_layers) {
		setup_debug_messenger();
	}

	// Creates VKSurface
	create_surface();

	pick_physical_device();
	create_logical_device();
	create_swapchain();
	create_image_views();
	create_render_pass();
	create_gfx_pipeline();
	create_framebuffers();
	create_command_pool();
	create_command_buffers();
	create_sync_primitives();

}

static void fb_resize_callback(GLFWwindow* window, int width, int height) {
	auto app = reinterpret_cast<VKBase*>(glfwGetWindowUserPointer(window));
	app->resized = true;
}

void VKBase::init_window() {
	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
	window = glfwCreateWindow(width, height, "Vulkan", fullscreen ?
		glfwGetPrimaryMonitor() : nullptr, nullptr);
	glfwSetWindowUserPointer(window, this);
	glfwSetFramebufferSizeCallback(window, fb_resize_callback);
}

void VKBase::render_loop() {
	while (!glfwWindowShouldClose(window)) {
		glfwPollEvents();
		draw_frame();
	}

	vkDeviceWaitIdle(device);


}

void VKBase::draw_frame() {
	vkWaitForFences(device, 1, &in_flight_fences[current_frame], VK_TRUE, UINT64_MAX);

	uint32_t image_idx;
	VkResult result = vkAcquireNextImageKHR(
		device, 
		swapchain,
		UINT64_MAX, 
		image_available_sem[current_frame], 
		VK_NULL_HANDLE,
		&image_idx);

	if (result == VK_ERROR_OUT_OF_DATE_KHR) {
		// Window resize
		recreate_swap_chain();
		return;
	} else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
		throw std::runtime_error("Failed to acquire new swap chain image");
	}

	if (images_in_flight[image_idx] != VK_NULL_HANDLE) {
		vkWaitForFences(device, 1, &images_in_flight[image_idx], VK_TRUE, UINT64_MAX);
	}
	images_in_flight[image_idx] = in_flight_fences[current_frame];

	VkSubmitInfo submit_info = vks::submit_info();

	VkSemaphore wait_semaphores[] = { image_available_sem[current_frame] };
	VkPipelineStageFlags wait_stages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
	submit_info.waitSemaphoreCount = 1;
	submit_info.pWaitSemaphores = wait_semaphores;
	submit_info.pWaitDstStageMask = wait_stages;

	submit_info.commandBufferCount = 1;
	submit_info.pCommandBuffers = &command_buffers[image_idx];

	VkSemaphore signal_semaphores[] = { render_finished_sem[current_frame] };
	submit_info.signalSemaphoreCount = 1;
	submit_info.pSignalSemaphores = signal_semaphores;

	vkResetFences(device, 1, &in_flight_fences[current_frame]);

	if (vkQueueSubmit(gfx_queue, 1, &submit_info, in_flight_fences[current_frame]) != VK_SUCCESS) {
		throw std::runtime_error("Failed to submit draw command buffer");
	}

	VkPresentInfoKHR present_info{};
	present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

	present_info.waitSemaphoreCount = 1;
	present_info.pWaitSemaphores = signal_semaphores;

	VkSwapchainKHR swapchains[] = { swapchain };
	present_info.swapchainCount = 1;
	present_info.pSwapchains = swapchains;

	present_info.pImageIndices = &image_idx;

	result = vkQueuePresentKHR(present_queue, &present_info);

	if (result == VK_ERROR_OUT_OF_DATE_KHR ||
		result == VK_SUBOPTIMAL_KHR ||
		resized) {
		resized = false;
		recreate_swap_chain();
	} else if (result != VK_SUCCESS) {
		throw std::runtime_error("Failed to present swap chain image");
	}

	current_frame = (current_frame + 1) % MAX_FRAMES_IN_FLIGHT;

}

void VKBase::cleanup_swapchain() {
	// Order:
		// 1- Destroy Framebuffers
		// 2- Destroy Commandbuffers
		// 3- Destroy the pipelines
		// 4- Destroy pipeline layout	
		// 5- Destroy render pass
		// 6- Destroy image views
		// 7- Destroy swapchain
	for (auto framebuffer : swapchain_framebuffers) {
		vkDestroyFramebuffer(device, framebuffer, nullptr);
	}

	vkFreeCommandBuffers(device, command_pool,
		static_cast<uint32_t>(command_buffers.size()), command_buffers.data());

	vkDestroyPipeline(device, gfx_pipeline, nullptr);
	vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
	vkDestroyRenderPass(device, render_pass, nullptr);

	for (auto image_view : swapchain_image_views) {
		vkDestroyImageView(device, image_view, nullptr);
	}

	vkDestroySwapchainKHR(device, swapchain, nullptr);
}

void VKBase::cleanup() {
	cleanup_swapchain();

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		vkDestroySemaphore(device, image_available_sem[i], nullptr);
		vkDestroySemaphore(device, render_finished_sem[i], nullptr);
		vkDestroyFence(device, in_flight_fences[i], nullptr);
	}

	vkDestroyCommandPool(device, command_pool, nullptr);
	vkDestroySurfaceKHR(instance, surface, nullptr);

	vkDestroyDevice(device, nullptr);
	if (enable_validation_layers) {
		vkExt_destroy_debug_messenger(instance, debug_messenger, nullptr);
	}

	vkDestroyInstance(instance, nullptr);
	glfwDestroyWindow(window);

	glfwTerminate();

}



void VKBase::create_instance() {

	if (enable_validation_layers && !check_validation_layer_support()) {
		throw std::runtime_error("Validation layers requested, but not available!");
	}
	VkApplicationInfo app_info{};
	app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	app_info.pApplicationName = "Lumen";
	app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
	app_info.pEngineName = "Lumen";
	app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
	app_info.apiVersion = VK_API_VERSION_1_0;

	VkInstanceCreateInfo instance_CI{};
	instance_CI.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	instance_CI.pApplicationInfo = &app_info;

	auto extensions = get_req_extensions();
	instance_CI.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
	instance_CI.ppEnabledExtensionNames = extensions.data();


	if (enable_validation_layers) {
		instance_CI.enabledLayerCount = static_cast<uint32_t>(validation_layers_lst.size());
		instance_CI.ppEnabledLayerNames = validation_layers_lst.data();
		VkDebugUtilsMessengerCreateInfoEXT debug_CI = vks::debug_messenger_CI(debug_callback);
		instance_CI.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debug_CI;
	} else {
		instance_CI.enabledLayerCount = 0;
		instance_CI.pNext = nullptr;
	}

	if (vkCreateInstance(&instance_CI, nullptr, &instance) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create instance");
	}

}



void VKBase::create_surface() {
	if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create window surface");
	}
}

void VKBase::pick_physical_device() {
	uint32_t device_cnt = 0;
	vkEnumeratePhysicalDevices(instance, &device_cnt, nullptr);

	if (device_cnt == 0) {
		throw std::runtime_error("Failed to find GPUs with Vulkan support");
	}

	std::vector<VkPhysicalDevice> devices(device_cnt);
	vkEnumeratePhysicalDevices(instance, &device_cnt, devices.data());



	auto is_suitable = [&](VkPhysicalDevice device) {
		// Is device suitable?
		QueueFamilyIndices indices = find_queue_families(device);

		// Check device extension support
		bool extensions_supported = [&](VkPhysicalDevice device) {

			uint32_t extension_cnt;
			vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_cnt, nullptr);

			std::vector<VkExtensionProperties> available_extensions(extension_cnt);
			vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_cnt, available_extensions.data());

			std::set<std::string> required_extensions(device_extensions.begin(), device_extensions.end());

			for (const auto& extension : available_extensions) {
				required_extensions.erase(extension.extensionName);
			}

			return required_extensions.empty();

		}(device);

		// Query swaphcain support
		bool swapchain_adequate = false;
		if (extensions_supported) {
			SwapChainSupportDetails swapchain_support = query_swapchain_support(device);
			// If we have a format and present mode, it's adequate
			swapchain_adequate = !swapchain_support.formats.empty() &&
				!swapchain_support.present_modes.empty();
		}
		// If we have the appropiate queue families, extensions and adequate swapchain, return true
		return indices.is_complete() && extensions_supported && swapchain_adequate;


	};
	for (const auto& device : devices) {
		if (is_suitable(device)) {
			physical_device = device;
			break;
		}
	}

	if (physical_device == VK_NULL_HANDLE) {
		throw std::runtime_error("Failed to find a suitable GPU");
	}

}

void VKBase::create_logical_device() {

	QueueFamilyIndices indices = find_queue_families(physical_device);

	std::vector<VkDeviceQueueCreateInfo> queue_CIs;
	std::set<uint32_t> unique_queue_families = {
		indices.gfx_family.value(),
		indices.present_family.value()
	};

	float queue_priority = 1.0f;
	for (uint32_t queue_family_idx : unique_queue_families) {
		VkDeviceQueueCreateInfo queue_CI{};
		queue_CI.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queue_CI.queueFamilyIndex = queue_family_idx;
		queue_CI.queueCount = 1;
		queue_CI.pQueuePriorities = &queue_priority;
		queue_CIs.push_back(queue_CI);
	}

	VkPhysicalDeviceFeatures device_features{};

	VkDeviceCreateInfo logical_device_CI{};
	logical_device_CI.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

	logical_device_CI.queueCreateInfoCount = static_cast<uint32_t>(queue_CIs.size());
	logical_device_CI.pQueueCreateInfos = queue_CIs.data();

	logical_device_CI.pEnabledFeatures = &device_features;

	logical_device_CI.enabledExtensionCount = static_cast<uint32_t>(device_extensions.size());
	logical_device_CI.ppEnabledExtensionNames = device_extensions.data();

	if (enable_validation_layers) {
		logical_device_CI.enabledLayerCount = static_cast<uint32_t>(validation_layers_lst.size());
		logical_device_CI.ppEnabledLayerNames = validation_layers_lst.data();
	} else {
		logical_device_CI.enabledLayerCount = 0;
	}

	if (vkCreateDevice(physical_device, &logical_device_CI, nullptr, &device) != VK_SUCCESS) {
		throw std::runtime_error("Failed to create logical device");
	}

	vkGetDeviceQueue(device, indices.gfx_family.value(), 0, &gfx_queue);
	vkGetDeviceQueue(device, indices.present_family.value(), 0, &present_queue);

}

void VKBase::create_swapchain() {

	SwapChainSupportDetails swapchain_support = query_swapchain_support(physical_device);

	// Pick surface format, present mode and extent(preferrably width and height)
	VkSurfaceFormatKHR surface_format = [&](const std::vector<VkSurfaceFormatKHR>& available_formats) {
		for (const auto& available_format : available_formats) {
			// Preferrably SRGB 32 for now
			if (available_format.format == VK_FORMAT_B8G8R8A8_SRGB && 
				available_format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {

				return available_format;
			}
		}
		return available_formats[0];
	}(swapchain_support.formats);

	VkPresentModeKHR present_mode = [&](const std::vector<VkPresentModeKHR>& present_modes) {
		for (const auto& available_present_mode : present_modes) {
			// For now we prefer Mailbox
			if (available_present_mode == VK_PRESENT_MODE_MAILBOX_KHR) {
				return available_present_mode;
			}
		}

		return VK_PRESENT_MODE_FIFO_KHR;
	}(swapchain_support.present_modes);
	VkExtent2D extent = [&](const VkSurfaceCapabilitiesKHR& capabilities) {
		// Choose swap chain extent
		if (capabilities.currentExtent.width != UINT32_MAX) {
			return capabilities.currentExtent;
		} else {
			int width, height;
			glfwGetFramebufferSize(window, &width, &height);

			VkExtent2D actual_extent = { 
				 static_cast<uint32_t>(width), 
				 static_cast<uint32_t>(height) 
			};

			// Clamp width and height
			actual_extent.width = std::max(capabilities.minImageExtent.width,
				std::min(capabilities.maxImageExtent.width, actual_extent.width));

			actual_extent.height = std::max(capabilities.minImageExtent.height,
				std::min(capabilities.maxImageExtent.height, actual_extent.height));

			return actual_extent;
		}
	}(swapchain_support.capabilities);

	uint32_t image_cnt = swapchain_support.capabilities.minImageCount + 1;
	if (swapchain_support.capabilities.maxImageCount > 0 && 
		image_cnt > swapchain_support.capabilities.maxImageCount) {

		image_cnt = swapchain_support.capabilities.maxImageCount;
	}

	VkSwapchainCreateInfoKHR swapchain_CI{};
	swapchain_CI.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	swapchain_CI.surface = surface;

	swapchain_CI.minImageCount = image_cnt;
	swapchain_CI.imageFormat = surface_format.format;
	swapchain_CI.imageColorSpace = surface_format.colorSpace;
	swapchain_CI.imageExtent = extent;
	swapchain_CI.imageArrayLayers = 1;
	swapchain_CI.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

	QueueFamilyIndices indices = find_queue_families(physical_device);
	uint32_t queue_family_indices_arr[] = { 
		indices.gfx_family.value(), 
		indices.present_family.value() 
	};

	if (indices.gfx_family != indices.present_family) {
		swapchain_CI.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
		swapchain_CI.queueFamilyIndexCount = 2;
		swapchain_CI.pQueueFamilyIndices = queue_family_indices_arr;
	} else {
		swapchain_CI.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
	}

	swapchain_CI.preTransform = swapchain_support.capabilities.currentTransform;
	swapchain_CI.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	swapchain_CI.presentMode = present_mode;
	swapchain_CI.clipped = VK_TRUE;

	swapchain_CI.oldSwapchain = VK_NULL_HANDLE;

	if (vkCreateSwapchainKHR(device, &swapchain_CI, nullptr, &swapchain) != VK_SUCCESS) {
		throw std::runtime_error("failed to create swap chain!");
	}

	vkGetSwapchainImagesKHR(device, swapchain, &image_cnt, nullptr);
	swapchain_images.resize(image_cnt);
	vkGetSwapchainImagesKHR(device, swapchain, &image_cnt, swapchain_images.data());

	swapchain_image_format = surface_format.format;
	swapchain_extent = extent;


}

void VKBase::create_image_views() {

	swapchain_image_views.resize(swapchain_images.size());

	for (size_t i = 0; i < swapchain_images.size(); i++) {
		VkImageViewCreateInfo image_view_CI = vks::image_view_CI();
		image_view_CI.image = swapchain_images[i];
		image_view_CI.viewType = VK_IMAGE_VIEW_TYPE_2D;
		image_view_CI.format = swapchain_image_format;
		image_view_CI.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
		image_view_CI.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
		image_view_CI.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
		image_view_CI.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
		image_view_CI.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		image_view_CI.subresourceRange.baseMipLevel = 0;
		image_view_CI.subresourceRange.levelCount = 1;
		image_view_CI.subresourceRange.baseArrayLayer = 0;
		image_view_CI.subresourceRange.layerCount = 1;

		if (vkCreateImageView(device, &image_view_CI, nullptr, &swapchain_image_views[i]) != VK_SUCCESS) {
			throw std::runtime_error("failed to create image views!");
		}
	}


}
void VKBase::create_command_pool() {
	QueueFamilyIndices queue_family_idxs = find_queue_families(physical_device);

	VkCommandPoolCreateInfo pool_info = vks::command_pool_CI();
	pool_info.queueFamilyIndex = queue_family_idxs.gfx_family.value();

	if (vkCreateCommandPool(device, &pool_info, nullptr, &command_pool) != VK_SUCCESS) {
		throw std::runtime_error("failed to create command pool!");
	}

}

void VKBase::create_command_buffers() {

}

void VKBase::create_sync_primitives() {
	image_available_sem.resize(MAX_FRAMES_IN_FLIGHT);
	render_finished_sem.resize(MAX_FRAMES_IN_FLIGHT);
	in_flight_fences.resize(MAX_FRAMES_IN_FLIGHT);
	images_in_flight.resize(swapchain_images.size(), VK_NULL_HANDLE);

	VkSemaphoreCreateInfo semaphore_info = vks::semaphore_create_info();

	VkFenceCreateInfo fence_info = vks::fence_create_info();
	fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

	for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		if (vkCreateSemaphore(device, &semaphore_info, nullptr, &image_available_sem[i]) != VK_SUCCESS ||
			vkCreateSemaphore(device, &semaphore_info, nullptr, &render_finished_sem[i]) != VK_SUCCESS ||
			vkCreateFence(device, &fence_info, nullptr, &in_flight_fences[i]) != VK_SUCCESS) {
			throw std::runtime_error("Failed to create synchronization primitives for a frame");
		}
	}

}



void VKBase::recreate_swap_chain() {
	int width = 0, height = 0;
	glfwGetFramebufferSize(window, &width, &height);
	while (width == 0 || height == 0) {
		// Window is minimized
		glfwGetFramebufferSize(window, &width, &height);
		glfwWaitEvents();
	}

	vkDeviceWaitIdle(device);
	cleanup_swapchain();

	create_swapchain();
	create_image_views();
	create_render_pass();
	create_gfx_pipeline();
	create_framebuffers();
	create_command_buffers();


}


bool VKBase::check_validation_layer_support() {
	uint32_t layer_cnt;
	vkEnumerateInstanceLayerProperties(&layer_cnt, nullptr);

	std::vector<VkLayerProperties> available_layers(layer_cnt);
	vkEnumerateInstanceLayerProperties(&layer_cnt, available_layers.data());

	for (const char* layer_name : validation_layers_lst) {
		bool layer_found = false;

		for (const auto& layerProperties : available_layers) {
			if (strcmp(layer_name, layerProperties.layerName) == 0) {
				layer_found = true;
				break;
			}
		}

		if (!layer_found) {
			return false;
		}
	}

	return true;

}

void VKBase::init() {

	init_window();
	init_vulkan();
}
