#include "Framework/Logger.h"
#include <assert.h>
#include <unordered_map>

#ifdef min
#undef min
#endif

#ifdef max
#undef max
#endif
#define GLFW_INCLUDE_VULKAN
#pragma warning(push,0)
#include "Framework/VulkanStructs.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"
#include <imgui.h>
#include <GLFW/glfw3.h>
#pragma warning(pop)
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <optional>
#include <queue>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include "Framework/ThreadPool.h"
#include <glm/ext.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/string_cast.hpp>
