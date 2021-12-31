#include "LumenPCH.h"
#include "CommandBuffer.h"
CommandBuffer::CommandBuffer(VulkanContext* ctx, bool begin,
							 VkCommandBufferUsageFlags begin_flags, 
							 QueueType type,
	VkCommandBufferLevel level) {
	this->ctx = ctx;
	this->type = type;
	auto cmd_buf_allocate_info =
		vk::command_buffer_allocate_info(ctx->cmd_pools[(int)type], level, 1);
	vk::check(
		vkAllocateCommandBuffers(ctx->device, &cmd_buf_allocate_info, &handle),
		"Could not allocate command buffer");
	if (begin) {
		auto begin_info = vk::command_buffer_begin_info(begin_flags);
		vk::check(vkBeginCommandBuffer(handle, &begin_info),
			"Could not begin the command buffer");
		state = CommandBufferState::RECORDING;
	}
}


void CommandBuffer::begin(VkCommandBufferUsageFlags begin_flags) {
	LUMEN_ASSERT(state != CommandBufferState::RECORDING, "Command buffer is already recording");
	auto begin_info = vk::command_buffer_begin_info(begin_flags);
	vk::check(vkBeginCommandBuffer(handle, &begin_info),
		"Could not begin the command buffer");
	state = CommandBufferState::RECORDING;
}

void CommandBuffer::submit(bool wait_fences, bool queue_wait_idle) {
	vk::check(vkEndCommandBuffer(handle), "Failed to end command buffer");
	state = CommandBufferState::STOPPED;
	VkSubmitInfo submit_info = vk::submit_info();
	submit_info.commandBufferCount = 1;
	submit_info.pCommandBuffers = &handle;
	if (wait_fences) {
		VkFenceCreateInfo fence_info = vk::fence_create_info(0);
		VkFence fence;
		vk::check(vkCreateFence(ctx->device, &fence_info, nullptr, &fence),
			"Fence creation error");
		vk::check(vkQueueSubmit(ctx->queues[(int)type], 1, &submit_info, fence),
			"Queue submission error");
		vk::check(
			vkWaitForFences(ctx->device, 1, &fence, VK_TRUE, 100000000000),
			"Fence wait error");
		vkDestroyFence(ctx->device, fence, nullptr);
	}
	else {
		vk::check(vkQueueSubmit(ctx->queues[(int)type], 1, &submit_info,
			VK_NULL_HANDLE),
			"Queue submission error");
	}
	if (queue_wait_idle) {
		vk::check(vkQueueWaitIdle(ctx->queues[(int)type]),
			"Queue wait error! Check previous submissions");

	}
}

CommandBuffer::~CommandBuffer() {
	if (handle == VK_NULL_HANDLE) {
		return;
	}
	if (state == CommandBufferState::RECORDING) {
		vk::check(vkEndCommandBuffer(handle), "Failed to end command buffer");
	}
	vkFreeCommandBuffers(ctx->device, ctx->cmd_pools[(int)type], 1, &handle);
}
