#include <unordered_map>
#include <assert.h> 
#include "core/Logger.h"
#ifdef min
#undef min
#endif

#ifdef max
#undef max
#endif
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <set>
#include <optional>
#include <fstream>
#include <string>
#include <cstdio>
#include <utility>
#include <filesystem>
#include <chrono>
#include <thread>
#include <functional>
#include "gfx/vulkan/VKStructs.h"
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/ext.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/string_cast.hpp>