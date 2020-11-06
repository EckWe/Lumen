#include "LumenPCH.h"
#include "VulkanBase.h"
#include "Utils.h"

uint32_t VertexLayout::stride() {
	uint32_t res = 0;
	for(auto& component : components) {
		switch(component) {
			case vk::Component::L_UV:
				res += 2 * sizeof(float);
				break;
			case vk::Component::L_TANGENT:
				res += 4 * sizeof(float);
			default:
				// Rest are 3 floats
				res += 3 * sizeof(float);
		}
	}
	return res;
}


VulkanBase::VulkanBase(int width, int height, bool fullscreen, bool debug) {
	this->width = width;
	this->height = height;
	this->fullscreen = fullscreen;
	this->enable_validation_layers = debug;
	create_instance();
}

std::vector<const char*> VulkanBase::get_req_extensions() {
	uint32_t glfwExtensionCount = 0;
	const char** glfwExtensions;
	glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

	std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

	if(enable_validation_layers) {
		extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	}
	return extensions;
}

VulkanBase::QueueFamilyIndices VulkanBase::find_queue_families(VkPhysicalDevice device) {

	QueueFamilyIndices indices;
	uint32_t queue_family_count = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);
	queue_families.resize(queue_family_count);
	vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.data());

	int i = 0;
	for(const auto& queueFamily : queue_families) {
		if(queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
			indices.gfx_family = i;
		}

		VkBool32 present_support = false;
		vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &present_support);

		if(present_support) {
			indices.present_family = i;
		}

		if(indices.is_complete()) {
			break;
		}

		i++;
	}
	return indices;
}

VulkanBase::SwapChainSupportDetails VulkanBase::query_swapchain_support(VkPhysicalDevice device) {
	// Basically returns present modes and surface modes in a struct

	SwapChainSupportDetails details;

	vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

	uint32_t format_cnt;
	vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_cnt, nullptr);

	if(format_cnt != 0) {
		details.formats.resize(format_cnt);
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_cnt, details.formats.data());
	}

	uint32_t present_mode_cnt;
	vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &present_mode_cnt, nullptr);

	if(present_mode_cnt != 0) {
		details.present_modes.resize(present_mode_cnt);
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &present_mode_cnt, details.present_modes.data());
	}
	return details;
}


VkShaderModule VulkanBase::create_shader(const std::vector<char>& code) {
	VkShaderModuleCreateInfo shader_module_CI{};
	shader_module_CI.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	shader_module_CI.codeSize = code.size();
	shader_module_CI.pCode = reinterpret_cast<const uint32_t*>(code.data());

	VkShaderModule shaderModule;
	vk::check(
		vkCreateShaderModule(device, &shader_module_CI, nullptr, &shaderModule),
		"Failed to create shader module!"
	);
	return shaderModule;
}


VkResult VulkanBase::vkExt_create_debug_messenger(
	VkInstance instance,
	const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
	const VkAllocationCallbacks* pAllocator,
	VkDebugUtilsMessengerEXT* pDebugMessenger) {

	auto func = (PFN_vkCreateDebugUtilsMessengerEXT)
		vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");

	if(func != nullptr) {
		return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
	} else {
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}

void VulkanBase::vkExt_destroy_debug_messenger(VkInstance instance,
											   VkDebugUtilsMessengerEXT debug_messenger,
											   const VkAllocationCallbacks* pAllocator) {
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)
		vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
	if(func != nullptr) {
		func(instance, debug_messenger, pAllocator);
	}
}

static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
	VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
	VkDebugUtilsMessageTypeFlagsEXT messageType,
	const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
	void* pUserData) {
	LUMEN_ERROR("Validation Error: {0} ", pCallbackData->pMessage);
	return VK_FALSE;
}

void VulkanBase::setup_debug_messenger() {


	auto ci = vk::debug_messenger_CI(debug_callback);

	vk::check(vkExt_create_debug_messenger(instance, &ci, nullptr, &debug_messenger),
			  "Failed to set up debug messenger!"
	);
}

void VulkanBase::init_vulkan() {

	create_instance();
	if(enable_validation_layers) {
		setup_debug_messenger();
	}

	// Creates VKSurface
	create_surface();

	pick_physical_device();
	create_logical_device();
	create_swapchain();
	create_image_views();
	create_render_pass();
	create_framebuffers();
	create_command_pool();
	create_command_buffers();
	create_sync_primitives();
}

static void fb_resize_callback(GLFWwindow* window, int width, int height) {
	auto app = reinterpret_cast<VulkanBase*>(glfwGetWindowUserPointer(window));
	app->resized = true;
}

void VulkanBase::init_window() {
	glfwSetFramebufferSizeCallback(window, fb_resize_callback);
}


void VulkanBase::draw_frame() {
	vkWaitForFences(device, 1, &in_flight_fences[current_frame], VK_TRUE, UINT64_MAX);

	uint32_t image_idx;
	VkResult result = vkAcquireNextImageKHR(
		device,
		swapchain,
		UINT64_MAX,
		image_available_sem[current_frame],
		VK_NULL_HANDLE,
		&image_idx);

	if(EventHandler::consume_event(LumenEvent::EVENT_SHADER_RELOAD)) {
		// We don't want any command buffers in flight, might change in the future
		vkDeviceWaitIdle(device);
		for(auto& old_pipeline : EventHandler::obsolete_pipelines) {
			vkDestroyPipeline(device, old_pipeline, nullptr);
		}
		EventHandler::obsolete_pipelines.clear();
		create_command_buffers();
		build_command_buffers();
	}

	if(result == VK_ERROR_OUT_OF_DATE_KHR) {
		// Window resize
		recreate_swap_chain();
		return;
	} else if(result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
		LUMEN_ERROR("Failed to acquire new swap chain image");
	}

	if(images_in_flight[image_idx] != VK_NULL_HANDLE) {
		vkWaitForFences(device, 1, &images_in_flight[image_idx], VK_TRUE, UINT64_MAX);
	}
	images_in_flight[image_idx] = in_flight_fences[current_frame];

	VkSubmitInfo submit_info = vk::submit_info();
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

	vk::check(vkQueueSubmit(gfx_queue, 1, &submit_info, in_flight_fences[current_frame]),
			  "Failed to submit draw command buffer"
	);
	VkPresentInfoKHR present_info{};
	present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

	present_info.waitSemaphoreCount = 1;
	present_info.pWaitSemaphores = signal_semaphores;

	VkSwapchainKHR swapchains[] = { swapchain };
	present_info.swapchainCount = 1;
	present_info.pSwapchains = swapchains;

	present_info.pImageIndices = &image_idx;

	result = vkQueuePresentKHR(present_queue, &present_info);

	if(result == VK_ERROR_OUT_OF_DATE_KHR ||
	   result == VK_SUBOPTIMAL_KHR ||
	   resized) {
		resized = false;
		recreate_swap_chain();
	} else if(result != VK_SUCCESS) {
		LUMEN_ERROR("Failed to present swap chain image");
	}

	current_frame = (current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
}

void VulkanBase::cleanup_swapchain() {
	// Order:
		// 1- Destroy Framebuffers
		// 2- Destroy Commandbuffers
		// 3- Destroy the pipelines
		// 4- Destroy pipeline layout	
		// 5- Destroy render pass
		// 6- Destroy image views
		// 7- Destroy swapchain
	for(auto framebuffer : swapchain_framebuffers) {
		vkDestroyFramebuffer(device, framebuffer, nullptr);
	}

	vkFreeCommandBuffers(device, command_pool,
						 static_cast<uint32_t>(command_buffers.size()), command_buffers.data());

	vkDestroyPipeline(device, gfx_pipeline, nullptr);
	vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
	vkDestroyRenderPass(device, render_pass, nullptr);

	for(auto image_view : swapchain_image_views) {
		vkDestroyImageView(device, image_view, nullptr);
	}

	vkDestroySwapchainKHR(device, swapchain, nullptr);
}

void VulkanBase::cleanup() {
	cleanup_swapchain();

	vkDestroyDescriptorPool(device, descriptor_pool, nullptr);

	for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		vkDestroySemaphore(device, image_available_sem[i], nullptr);
		vkDestroySemaphore(device, render_finished_sem[i], nullptr);
		vkDestroyFence(device, in_flight_fences[i], nullptr);
	}

	vkDestroyCommandPool(device, command_pool, nullptr);
	vkDestroySurfaceKHR(instance, surface, nullptr);

	vkDestroyDevice(device, nullptr);
	if(enable_validation_layers) {
		vkExt_destroy_debug_messenger(instance, debug_messenger, nullptr);
	}

	vkDestroyInstance(instance, nullptr);
	glfwDestroyWindow(window);
	glfwTerminate();
}

void VulkanBase::create_instance() {

	if(enable_validation_layers && !check_validation_layer_support()) {
		LUMEN_ERROR("Validation layers requested, but not available!");
	}
	VkApplicationInfo app_info{};
	app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	app_info.pApplicationName = "Lumen";
	app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 1);
	app_info.pEngineName = "Lumen Engine";
	app_info.engineVersion = VK_MAKE_VERSION(1, 0, 1);
	app_info.apiVersion = VK_API_VERSION_1_2;

	VkInstanceCreateInfo instance_CI{};
	instance_CI.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	instance_CI.pApplicationInfo = &app_info;

	auto extensions = get_req_extensions();
	instance_CI.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
	instance_CI.ppEnabledExtensionNames = extensions.data();


	if(enable_validation_layers) {
		instance_CI.enabledLayerCount = static_cast<uint32_t>(validation_layers_lst.size());
		instance_CI.ppEnabledLayerNames = validation_layers_lst.data();
		VkDebugUtilsMessengerCreateInfoEXT debug_CI = vk::debug_messenger_CI(debug_callback);
		instance_CI.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debug_CI;
	} else {
		instance_CI.enabledLayerCount = 0;
		instance_CI.pNext = nullptr;
	}

	vk::check(vkCreateInstance(&instance_CI, nullptr, &instance),
			  "Failed to create instance"
	);
}

void VulkanBase::create_surface() {

	vk::check(glfwCreateWindowSurface(instance, window, nullptr, &surface),
			  "Failed to create window surface"
	);
}

void VulkanBase::pick_physical_device() {
	uint32_t device_cnt = 0;
	vkEnumeratePhysicalDevices(instance, &device_cnt, nullptr);
	if(device_cnt == 0) {
		LUMEN_ERROR("Failed to find GPUs with Vulkan support");
	}

	std::vector<VkPhysicalDevice> devices(device_cnt);
	vkEnumeratePhysicalDevices(instance, &device_cnt, devices.data());

	// Is device suitable?
	auto is_suitable = [this](VkPhysicalDevice device) {
		QueueFamilyIndices indices = find_queue_families(device);

		// Check device extension support
		auto extensions_supported = [this](VkPhysicalDevice device) {

			uint32_t extension_cnt;
			vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_cnt, nullptr);

			std::vector<VkExtensionProperties> available_extensions(extension_cnt);
			vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_cnt, available_extensions.data());

			std::set<std::string> required_extensions(device_extensions.begin(), device_extensions.end());

			for(const auto& extension : available_extensions) {
				required_extensions.erase(extension.extensionName);
			}
			return required_extensions.empty();

		}(device);

		// Query swaphcain support
		bool swapchain_adequate = false;
		if(extensions_supported) {
			SwapChainSupportDetails swapchain_support = query_swapchain_support(device);
			// If we have a format and present mode, it's adequate
			swapchain_adequate = !swapchain_support.formats.empty() &&
				!swapchain_support.present_modes.empty();
		}
		// If we have the appropiate queue families, extensions and adequate swapchain, return true
		return indices.is_complete() && extensions_supported && swapchain_adequate;
	};
	for(const auto& device : devices) {
		if(is_suitable(device)) {
			physical_device = device;
			vkGetPhysicalDeviceFeatures(physical_device, &supported_features);
			vkGetPhysicalDeviceProperties(physical_device, &device_properties);
			vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);
			break;
		}
	}

	if(physical_device == VK_NULL_HANDLE) {
		LUMEN_ERROR("Failed to find a suitable GPU");
	}
}

void VulkanBase::create_logical_device() {

	QueueFamilyIndices indices = find_queue_families(physical_device);

	std::vector<VkDeviceQueueCreateInfo> queue_CIs;
	std::set<uint32_t> unique_queue_families = {
		indices.gfx_family.value(),
		indices.present_family.value()
	};

	float queue_priority = 1.0f;
	for(uint32_t queue_family_idx : unique_queue_families) {
		VkDeviceQueueCreateInfo queue_CI{};
		queue_CI.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queue_CI.queueFamilyIndex = queue_family_idx;
		queue_CI.queueCount = 1;
		queue_CI.pQueuePriorities = &queue_priority;
		queue_CIs.push_back(queue_CI);
	}

	VkPhysicalDeviceFeatures device_features{};
	device_features.samplerAnisotropy = VK_TRUE;

	VkDeviceCreateInfo logical_device_CI{};
	logical_device_CI.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

	logical_device_CI.queueCreateInfoCount = static_cast<uint32_t>(queue_CIs.size());
	logical_device_CI.pQueueCreateInfos = queue_CIs.data();

	logical_device_CI.pEnabledFeatures = &device_features;

	logical_device_CI.enabledExtensionCount = static_cast<uint32_t>(device_extensions.size());
	logical_device_CI.ppEnabledExtensionNames = device_extensions.data();

	if(enable_validation_layers) {
		logical_device_CI.enabledLayerCount = static_cast<uint32_t>(validation_layers_lst.size());
		logical_device_CI.ppEnabledLayerNames = validation_layers_lst.data();
	} else {
		logical_device_CI.enabledLayerCount = 0;
	}

	vk::check(vkCreateDevice(physical_device, &logical_device_CI, nullptr, &device),
			  "Failed to create logical device"
	);

	vkGetDeviceQueue(device, indices.gfx_family.value(), 0, &gfx_queue);
	vkGetDeviceQueue(device, indices.present_family.value(), 0, &present_queue);
}

void VulkanBase::create_swapchain() {

	SwapChainSupportDetails swapchain_support = query_swapchain_support(physical_device);

	// Pick surface format, present mode and extent(preferrably width and height):

	VkSurfaceFormatKHR surface_format = [this](const std::vector<VkSurfaceFormatKHR>& available_formats) {
		for(const auto& available_format : available_formats) {
			// Preferrably SRGB32 for now
			if(available_format.format == VK_FORMAT_B8G8R8A8_SRGB &&
			   available_format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
				return available_format;
			}
		}
		return available_formats[0];
	}(swapchain_support.formats);

	VkPresentModeKHR present_mode = [this](const std::vector<VkPresentModeKHR>& present_modes) {
		for(const auto& available_present_mode : present_modes) {
			// For now we prefer Mailbox
			if(available_present_mode == VK_PRESENT_MODE_MAILBOX_KHR) {
				return available_present_mode;
			}
		}

		return VK_PRESENT_MODE_FIFO_KHR;
	}(swapchain_support.present_modes);

	// Choose swap chain extent
	VkExtent2D extent = [this](const VkSurfaceCapabilitiesKHR& capabilities) {
		if(capabilities.currentExtent.width != UINT32_MAX) {
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
	if(swapchain_support.capabilities.maxImageCount > 0 &&
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

	if(indices.gfx_family != indices.present_family) {
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


	vk::check(vkCreateSwapchainKHR(device, &swapchain_CI, nullptr, &swapchain),
			  "Failed to create swap chain!"
	);

	vkGetSwapchainImagesKHR(device, swapchain, &image_cnt, nullptr);
	swapchain_images.resize(image_cnt);
	vkGetSwapchainImagesKHR(device, swapchain, &image_cnt, swapchain_images.data());

	swapchain_image_format = surface_format.format;
	swapchain_extent = extent;
}

void VulkanBase::create_image_views() {

	swapchain_image_views.resize(swapchain_images.size());

	for(size_t i = 0; i < swapchain_images.size(); i++) {
		swapchain_image_views[i] = create_image_view(
			&this->device, swapchain_images[i],
			swapchain_image_format
		);
	}
}
void VulkanBase::create_command_pool() {
	QueueFamilyIndices queue_family_idxs = find_queue_families(physical_device);

	VkCommandPoolCreateInfo pool_info = vk::command_pool_CI();
	pool_info.queueFamilyIndex = queue_family_idxs.gfx_family.value();

	vk::check(vkCreateCommandPool(device, &pool_info, nullptr, &command_pool),
			  "Failed to create command pool!"
	);
}

void VulkanBase::create_framebuffers() {
	swapchain_framebuffers.resize(swapchain_image_views.size());

	for(size_t i = 0; i < swapchain_image_views.size(); i++) {
		VkImageView attachments[] = {
			swapchain_image_views[i]
		};

		VkFramebufferCreateInfo frame_buffer_info = vk::framebuffer_create_info();
		frame_buffer_info.renderPass = render_pass;
		frame_buffer_info.attachmentCount = 1;
		frame_buffer_info.pAttachments = attachments;
		frame_buffer_info.width = swapchain_extent.width;
		frame_buffer_info.height = swapchain_extent.height;
		frame_buffer_info.layers = 1;

		vk::check(vkCreateFramebuffer(device, &frame_buffer_info, nullptr, &swapchain_framebuffers[i]),
				  "Failed to create framebuffer"
		);
	}
}

void VulkanBase::create_command_buffers() {
	command_buffers.resize(swapchain_framebuffers.size());

	VkCommandBufferAllocateInfo alloc_info = vk::command_buffer_allocate_info(
		command_pool,
		VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		(uint32_t) command_buffers.size()
	);

	vk::check(vkAllocateCommandBuffers(device, &alloc_info, command_buffers.data()),
			  "Failed to allocate command buffers!"
	);
}

void VulkanBase::create_sync_primitives() {
	image_available_sem.resize(MAX_FRAMES_IN_FLIGHT);
	render_finished_sem.resize(MAX_FRAMES_IN_FLIGHT);
	in_flight_fences.resize(MAX_FRAMES_IN_FLIGHT);
	images_in_flight.resize(swapchain_images.size(), VK_NULL_HANDLE);

	VkSemaphoreCreateInfo semaphore_info = vk::semaphore_create_info();

	VkFenceCreateInfo fence_info = vk::fence_create_info(VK_FENCE_CREATE_SIGNALED_BIT);


	for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
		vk::check<3>(
			{
				vkCreateSemaphore(device, &semaphore_info, nullptr, &image_available_sem[i]),
				vkCreateSemaphore(device, &semaphore_info, nullptr, &render_finished_sem[i]),
				vkCreateFence(device, &fence_info, nullptr, &in_flight_fences[i])
			},
			"Failed to create synchronization primitives for a frame"
			);
	}
}


// Called after window resize
void VulkanBase::recreate_swap_chain() {
	int width = 0, height = 0;
	glfwGetFramebufferSize(window, &width, &height);
	while(width == 0 || height == 0) {
		// Window is minimized
		glfwGetFramebufferSize(window, &width, &height);
		glfwWaitEvents();
	}
	vkDeviceWaitIdle(device);
	cleanup_swapchain();
	create_swapchain();
	create_image_views();
	create_render_pass();
	create_framebuffers();
	create_command_buffers();
	build_command_buffers();
}




bool VulkanBase::check_validation_layer_support() {
	uint32_t layer_cnt;
	vkEnumerateInstanceLayerProperties(&layer_cnt, nullptr);

	std::vector<VkLayerProperties> available_layers(layer_cnt);
	vkEnumerateInstanceLayerProperties(&layer_cnt, available_layers.data());

	for(const char* layer_name : validation_layers_lst) {
		bool layer_found = false;

		for(const auto& layerProperties : available_layers) {
			if(strcmp(layer_name, layerProperties.layerName) == 0) {
				layer_found = true;
				break;
			}
		}
		if(!layer_found) {
			return false;
		}
	}
	return true;
}

void VulkanBase::init(GLFWwindow* window_ptr) {
	this->window = window_ptr;
	init_window();
	init_vulkan();
	initialized = true;
}
