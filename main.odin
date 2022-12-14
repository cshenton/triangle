package triangle

import "core:fmt"
import "core:runtime"
import SDL "vendor:sdl2"
import vk "vendor:vulkan"

DEFAULT_WIDTH :: 1920
DEFAULT_HEIGHT :: 1080

App :: struct {
	window:                     ^SDL.Window,
	instance:                   vk.Instance,
	debug_messenger:            vk.DebugUtilsMessengerEXT,
	physical_device:            vk.PhysicalDevice,
	device:                     vk.Device,
	graphics_queue:             vk.Queue,
	surface:                    vk.SurfaceKHR,
	swapchain:                  vk.SwapchainKHR,
	swapchain_format:           vk.Format,
	swapchain_extent:           vk.Extent2D,
	swapchain_images:           []vk.Image,
	swapchain_image_views:      []vk.ImageView,
	render_pass:                vk.RenderPass,
	pipeline_layout:            vk.PipelineLayout,
	pipeline:                   vk.Pipeline,
	framebuffers:               []vk.Framebuffer,
	command_pool:               vk.CommandPool,
	command_buffers:            [2]vk.CommandBuffer,
	image_available_semaphores: [2]vk.Semaphore,
	render_finished_semaphores: [2]vk.Semaphore,
	in_flight_fences:           [2]vk.Fence,
	current_frame:              u32,
	quit:                       bool,
}

debug_callback :: proc "c" (
	message_severity: vk.DebugUtilsMessageSeverityFlagsEXT,
	message_type: vk.DebugUtilsMessageTypeFlagsEXT,
	p_callback_data: ^vk.DebugUtilsMessengerCallbackDataEXT,
) -> u32 {
	context = runtime.default_context()
	fmt.printf("validation layer: %s\n", p_callback_data.pMessage)
	return 0
}

app_create_instance :: proc(window: ^SDL.Window) -> (instance: vk.Instance) {
	// Load base procedures
	SDL.Vulkan_LoadLibrary(nil)
	vk.load_proc_addresses(rawptr(SDL.Vulkan_GetVkGetInstanceProcAddr()))

	// Check for validation layers
	validation_layers := [1]cstring{"VK_LAYER_KHRONOS_validation"}
	layer_count: u32
	vk.EnumerateInstanceLayerProperties(&layer_count, nil)
	layers := make([]vk.LayerProperties, layer_count)
	defer delete(layers)
	vk.EnumerateInstanceLayerProperties(&layer_count, &layers[0])
	layer_found := false
	for layer in &layers {
		if string(cstring(&layer.layerName[0])) == "VK_LAYER_KHRONOS_validation" {
			layer_found = true
		}
	}
	if !layer_found {
		panic("Validation layer not found")
	}


	// Query SDL extension
	extension_count: u32
	SDL.Vulkan_GetInstanceExtensions(window, &extension_count, nil)
	extension_names := make([dynamic]cstring, extension_count)
	defer delete(extension_names)
	SDL.Vulkan_GetInstanceExtensions(window, &extension_count, &extension_names[0])
	debug_extension: cstring = vk.EXT_DEBUG_UTILS_EXTENSION_NAME
	append(&extension_names, debug_extension)

	// Create instance
	instance_info := vk.InstanceCreateInfo {
		sType                   = .INSTANCE_CREATE_INFO,
		pApplicationInfo        = &vk.ApplicationInfo{
			sType = .APPLICATION_INFO,
			pApplicationName = "Odin Vulkan Triangle",
			applicationVersion = vk.MAKE_VERSION(1, 0, 0),
			pEngineName = "No Engine",
			engineVersion = vk.MAKE_VERSION(1, 0, 0),
			apiVersion = vk.API_VERSION_1_3,
		},
		enabledExtensionCount   = u32(len(extension_names)),
		ppEnabledExtensionNames = &extension_names[0],
		enabledLayerCount       = 1,
		ppEnabledLayerNames     = &validation_layers[0],
	}
	result := vk.CreateInstance(&instance_info, nil, &instance)
	if result != .SUCCESS {
		panic("Instance creation failed")
	}
	vk.load_proc_addresses(instance)
	return
}

app_create_debug_messenger :: proc(instance: vk.Instance) -> (debug_messenger: vk.DebugUtilsMessengerEXT) {
	debug_create_info := vk.DebugUtilsMessengerCreateInfoEXT {
		sType = .DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
		// messageSeverity = {.ERROR, .INFO, .VERBOSE, .WARNING},
		messageSeverity = {.ERROR, .INFO, .WARNING},
		messageType = {.GENERAL, .PERFORMANCE, .VALIDATION},
		pfnUserCallback = vk.ProcDebugUtilsMessengerCallbackEXT(debug_callback),
		pUserData = nil,
	}
	result := vk.CreateDebugUtilsMessengerEXT(instance, &debug_create_info, nil, &debug_messenger)
	if (result != .SUCCESS) {
		panic("Failed to create debug messenger")
	}
	return
}

app_create_surface :: proc(window: ^SDL.Window, instance: vk.Instance) -> (surface: vk.SurfaceKHR) {
	if !SDL.Vulkan_CreateSurface(window, instance, &surface) {
		panic("Failed to create surface")
	}
	return
}

app_check_physical_device :: proc(
	physical_device: vk.PhysicalDevice,
	surface: vk.SurfaceKHR,
) -> (
	queue_index: u32,
	ok: bool,
) {
	// Ensure the device has the swapchain extension
	has_extension := false
	dev_extension_count: u32
	vk.EnumerateDeviceExtensionProperties(physical_device, nil, &dev_extension_count, nil)
	dev_extensions := make([]vk.ExtensionProperties, dev_extension_count)
	defer delete(dev_extensions)
	vk.EnumerateDeviceExtensionProperties(physical_device, nil, &dev_extension_count, &dev_extensions[0])
	for ext in &dev_extensions {
		if string(cstring(&ext.extensionName[0])) == string(vk.KHR_SWAPCHAIN_EXTENSION_NAME) {
			has_extension = true
		}
	}

	// Ensure the device is a discrete GPU
	device_properties: vk.PhysicalDeviceProperties
	vk.GetPhysicalDeviceProperties(physical_device, &device_properties)
	is_discrete := (device_properties.deviceType == .DISCRETE_GPU)

	// Ensure the device has a queue supporting presentation
	is_present: b32
	queue_family_count: u32
	vk.GetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nil)
	queue_families := make([]vk.QueueFamilyProperties, queue_family_count)
	defer delete(queue_families)
	vk.GetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, &queue_families[0])
	for family, i in queue_families {
		vk.GetPhysicalDeviceSurfaceSupportKHR(physical_device, u32(i), surface, &is_present)
		if (.GRAPHICS in family.queueFlags) && is_present {
			queue_index = u32(i)
			break
		}
	}

	ok = is_discrete && has_extension && is_present
	return
}

app_pick_physical_device :: proc(
	instance: vk.Instance,
	surface: vk.SurfaceKHR,
) -> (
	physical_device: vk.PhysicalDevice,
	queue_index: u32,
) {
	// Find an appropriate physical device
	physical_device_found := false

	device_count: u32
	vk.EnumeratePhysicalDevices(instance, &device_count, nil)
	devices := make([]vk.PhysicalDevice, device_count)
	defer delete(devices)
	vk.EnumeratePhysicalDevices(instance, &device_count, &devices[0])

	for dev in devices {
		dev_queue, ok := app_check_physical_device(dev, surface)
		if !ok {continue}
		queue_index = dev_queue
		physical_device_found = true
		physical_device = dev
		break
	}
	if !physical_device_found {
		panic("Failed to find appropriate physical device")
	}
	return
}

app_create_device :: proc(
	instance: vk.Instance,
	physical_device: vk.PhysicalDevice,
	queue_index: u32,
) -> (
	device: vk.Device,
	queue: vk.Queue,
) {
	validation_layers := [1]cstring{"VK_LAYER_KHRONOS_validation"}
	extensions := [1]cstring{vk.KHR_SWAPCHAIN_EXTENSION_NAME}
	queue_priority := f32(1)
	queue_create_info := vk.DeviceQueueCreateInfo {
		sType            = .DEVICE_QUEUE_CREATE_INFO,
		queueFamilyIndex = queue_index,
		queueCount       = 1,
		pQueuePriorities = &queue_priority,
	}
	device_create_info := vk.DeviceCreateInfo {
		sType                   = .DEVICE_CREATE_INFO,
		pQueueCreateInfos       = &queue_create_info,
		queueCreateInfoCount    = 1,
		pEnabledFeatures        = &vk.PhysicalDeviceFeatures{},
		enabledLayerCount       = len(validation_layers),
		ppEnabledLayerNames     = &validation_layers[0],
		enabledExtensionCount   = len(extensions),
		ppEnabledExtensionNames = &extensions[0],
	}
	result := vk.CreateDevice(physical_device, &device_create_info, nil, &device)
	if (result != .SUCCESS) {
		panic("Failed to create device")
	}
	vk.GetDeviceQueue(device, queue_index, 0, &queue)
	vk.load_proc_addresses(device)
	return
}


app_create_swapchain :: proc(
	physical_device: vk.PhysicalDevice,
	device: vk.Device,
	surface: vk.SurfaceKHR,
) -> (
	swapchain: vk.SwapchainKHR,
	images: []vk.Image,
	format: vk.Format,
	extent: vk.Extent2D,
) {
	// Get device information
	capabilities: vk.SurfaceCapabilitiesKHR
	vk.GetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &capabilities)

	format_count: u32
	vk.GetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &format_count, nil)
	formats := make([]vk.SurfaceFormatKHR, format_count)
	vk.GetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &format_count, &formats[0])

	present_count: u32
	vk.GetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface, &present_count, nil)
	presents := make([]vk.PresentModeKHR, present_count)
	vk.GetPhysicalDeviceSurfacePresentModesKHR(physical_device, surface, &present_count, &presents[0])


	// Choose format
	surface_format := formats[0]
	for swapchain_format in formats {
		if swapchain_format.format == .B8G8R8A8_SRGB && swapchain_format.colorSpace == .SRGB_NONLINEAR {
			surface_format = swapchain_format
			break
		}
	}

	// Choose present mode
	present_mode := vk.PresentModeKHR.FIFO
	for mode in presents {
		if mode != .MAILBOX {continue}
		present_mode = .MAILBOX
	}

	// Choose extents
	extent = capabilities.currentExtent

	image_count := capabilities.minImageCount + 1
	if capabilities.maxImageCount > 0 && image_count > capabilities.maxImageCount {
		image_count = capabilities.maxImageCount
	}

	swapchain_info := vk.SwapchainCreateInfoKHR {
		sType = .SWAPCHAIN_CREATE_INFO_KHR,
		surface = surface,
		minImageCount = image_count,
		imageFormat = surface_format.format,
		imageColorSpace = surface_format.colorSpace,
		imageExtent = extent,
		imageArrayLayers = 1,
		imageUsage = {.COLOR_ATTACHMENT},
		imageSharingMode = .EXCLUSIVE,
		preTransform = capabilities.currentTransform,
		compositeAlpha = {.OPAQUE},
		presentMode = present_mode,
		clipped = true,
	}

	result := vk.CreateSwapchainKHR(device, &swapchain_info, nil, &swapchain)
	if result != .SUCCESS {
		panic("Swapchain creation failed")
	}

	vk.GetSwapchainImagesKHR(device, swapchain, &image_count, nil)
	images = make([]vk.Image, image_count)
	vk.GetSwapchainImagesKHR(device, swapchain, &image_count, &images[0])
	format = surface_format.format
	return
}

app_create_image_views :: proc(device: vk.Device, images: []vk.Image, format: vk.Format) -> (image_views: []vk.ImageView) {
	image_views = make([]vk.ImageView, len(images))
	for i in 0 ..< len(images) {
		subresource := vk.ImageSubresourceRange {
			aspectMask = {.COLOR},
			baseMipLevel = 0,
			levelCount = 1,
			baseArrayLayer = 0,
			layerCount = 1,
		}
		view_info := vk.ImageViewCreateInfo {
			sType = .IMAGE_VIEW_CREATE_INFO,
			image = images[i],
			viewType = .D2,
			format = format,
			components = {.IDENTITY, .IDENTITY, .IDENTITY, .IDENTITY},
			subresourceRange = subresource,
		}
		result := vk.CreateImageView(device, &view_info, nil, &image_views[i])
		if result != .SUCCESS {
			panic("Swapchain image view creation failed")
		}
	}
	return
}

app_create_shader_module :: proc(device: vk.Device, src: []u8) -> (module: vk.ShaderModule) {
	module_info := vk.ShaderModuleCreateInfo {
		sType    = .SHADER_MODULE_CREATE_INFO,
		codeSize = len(src),
		pCode    = cast(^u32)&src[0],
	}

	result := vk.CreateShaderModule(device, &module_info, nil, &module)
	if result != .SUCCESS {
		panic("Shader module creation failed")
	}
	return
}

app_create_render_pass :: proc(device: vk.Device, format: vk.Format) -> (render_pass: vk.RenderPass) {
	color_attachment := vk.AttachmentDescription {
		format = format,
		samples = {._1},
		loadOp = .CLEAR,
		storeOp = .STORE,
		stencilLoadOp = .DONT_CARE,
		stencilStoreOp = .DONT_CARE,
		initialLayout = .UNDEFINED,
		finalLayout = .PRESENT_SRC_KHR,
	}

	color_attachment_ref := vk.AttachmentReference {
		attachment = 0,
		layout     = .COLOR_ATTACHMENT_OPTIMAL,
	}

	subpass := vk.SubpassDescription {
		pipelineBindPoint    = .GRAPHICS,
		colorAttachmentCount = 1,
		pColorAttachments    = &color_attachment_ref,
	}

	render_pass_info := vk.RenderPassCreateInfo {
		sType           = .RENDER_PASS_CREATE_INFO,
		attachmentCount = 1,
		pAttachments    = &color_attachment,
		subpassCount    = 1,
		pSubpasses      = &subpass,
	}

	result := vk.CreateRenderPass(device, &render_pass_info, nil, &render_pass)
	if result != .SUCCESS {
		panic("Render pass creation failed")
	}
	return
}

app_create_graphics_pipeline :: proc(
	device: vk.Device,
	render_pass: vk.RenderPass,
) -> (
	pipeline: vk.Pipeline,
	pipeline_layout: vk.PipelineLayout,
) {

	vert_module := app_create_shader_module(device, vertex_main_src[:])
	frag_module := app_create_shader_module(device, pixel_main_src[:])

	vert_info := vk.PipelineShaderStageCreateInfo {
		sType = .PIPELINE_SHADER_STAGE_CREATE_INFO,
		stage = {.VERTEX},
		module = vert_module,
		pName = "vertex_main",
	}

	frag_info := vk.PipelineShaderStageCreateInfo {
		sType = .PIPELINE_SHADER_STAGE_CREATE_INFO,
		stage = {.FRAGMENT},
		module = frag_module,
		pName = "pixel_main",
	}

	shader_stages := [2]vk.PipelineShaderStageCreateInfo{vert_info, frag_info}

	vertex_input_info := vk.PipelineVertexInputStateCreateInfo {
		sType                           = .PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		vertexBindingDescriptionCount   = 0,
		vertexAttributeDescriptionCount = 0,
	}

	input_assembly_info := vk.PipelineInputAssemblyStateCreateInfo {
		sType                  = .PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		topology               = .TRIANGLE_LIST,
		primitiveRestartEnable = false,
	}

	viewport_info := vk.PipelineViewportStateCreateInfo {
		sType         = .PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		viewportCount = 1,
		scissorCount  = 1,
	}

	rasterizer_info := vk.PipelineRasterizationStateCreateInfo {
		sType = .PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		depthClampEnable = false,
		rasterizerDiscardEnable = false,
		polygonMode = .FILL,
		lineWidth = 1.0,
		cullMode = {.BACK},
		frontFace = .CLOCKWISE,
		depthBiasEnable = false,
	}

	multisample_info := vk.PipelineMultisampleStateCreateInfo {
		sType = .PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		sampleShadingEnable = false,
		rasterizationSamples = {._1},
	}

	color_blend_attachment := vk.PipelineColorBlendAttachmentState {
		colorWriteMask = {.R, .G, .B, .A},
		blendEnable = false,
	}

	color_blend_info := vk.PipelineColorBlendStateCreateInfo {
		sType = .PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		logicOpEnable = false,
		logicOp = .COPY,
		attachmentCount = 1,
		pAttachments = &color_blend_attachment,
		blendConstants = {0.0, 0.0, 0.0, 0.0},
	}

	dynamic_states := [2]vk.DynamicState{.VIEWPORT, .SCISSOR}
	dynamic_state_info := vk.PipelineDynamicStateCreateInfo {
		sType             = .PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		dynamicStateCount = len(dynamic_states),
		pDynamicStates    = &dynamic_states[0],
	}

	pipeline_layout_info := vk.PipelineLayoutCreateInfo {
		sType                  = .PIPELINE_LAYOUT_CREATE_INFO,
		setLayoutCount         = 0,
		pushConstantRangeCount = 0,
	}

	result := vk.CreatePipelineLayout(device, &pipeline_layout_info, nil, &pipeline_layout)
	if result != .SUCCESS {
		panic("Pipeline layout creation failed")
	}

	pipeline_info := vk.GraphicsPipelineCreateInfo {
		sType               = .GRAPHICS_PIPELINE_CREATE_INFO,
		stageCount          = 2,
		pStages             = &shader_stages[0],
		pVertexInputState   = &vertex_input_info,
		pInputAssemblyState = &input_assembly_info,
		pViewportState      = &viewport_info,
		pRasterizationState = &rasterizer_info,
		pMultisampleState   = &multisample_info,
		pColorBlendState    = &color_blend_info,
		pDynamicState       = &dynamic_state_info,
		layout              = pipeline_layout,
		renderPass          = render_pass,
		subpass             = 0,
	}

	result = vk.CreateGraphicsPipelines(device, vk.PipelineCache(0), 1, &pipeline_info, nil, &pipeline)
	if result != .SUCCESS {
		panic("Graphics Pipeline creation failed")
	}

	vk.DestroyShaderModule(device, frag_module, nil)
	vk.DestroyShaderModule(device, vert_module, nil)
	return
}

app_create_framebuffers :: proc(
	device: vk.Device,
	render_pass: vk.RenderPass,
	extent: vk.Extent2D,
	image_views: []vk.ImageView,
) -> (
	framebuffers: []vk.Framebuffer,
) {
	framebuffers = make([]vk.Framebuffer, len(image_views))
	for view, i in image_views {
		attachments := [1]vk.ImageView{view}
		framebuffer_info := vk.FramebufferCreateInfo {
			sType           = .FRAMEBUFFER_CREATE_INFO,
			renderPass      = render_pass,
			attachmentCount = 1,
			pAttachments    = &attachments[0],
			width           = extent.width,
			height          = extent.height,
			layers          = 1,
		}

		result := vk.CreateFramebuffer(device, &framebuffer_info, nil, &framebuffers[i])
		if result != .SUCCESS {
			panic("Framebuffer creation failed")
		}
	}
	return
}

app_create_command_pool :: proc(device: vk.Device, queue_index: u32) -> (command_pool: vk.CommandPool) {
	pool_info := vk.CommandPoolCreateInfo {
		sType = .COMMAND_POOL_CREATE_INFO,
		flags = {.RESET_COMMAND_BUFFER},
		queueFamilyIndex = queue_index,
	}

	result := vk.CreateCommandPool(device, &pool_info, nil, &command_pool)
	if result != .SUCCESS {
		panic("Command Pool creation failed")
	}
	return
}

app_create_command_buffers :: proc(
	device: vk.Device,
	command_pool: vk.CommandPool,
) -> (
	command_buffers: [2]vk.CommandBuffer,
) {
	command_buffer_info := vk.CommandBufferAllocateInfo {
		sType              = .COMMAND_BUFFER_ALLOCATE_INFO,
		commandPool        = command_pool,
		level              = .PRIMARY,
		commandBufferCount = 2,
	}

	result := vk.AllocateCommandBuffers(device, &command_buffer_info, &command_buffers[0])
	if result != .SUCCESS {
		panic("Command Buffer allocation failed")
	}
	return
}

app_record_command_buffer :: proc(
	device: vk.Device,
	framebuffers: []vk.Framebuffer,
	pipeline: vk.Pipeline,
	render_pass: vk.RenderPass,
	extent: vk.Extent2D,
	image_index: u32,
	command_buffer: vk.CommandBuffer,
) {
	begin_info := vk.CommandBufferBeginInfo {
		sType = .COMMAND_BUFFER_BEGIN_INFO,
	}
	result := vk.BeginCommandBuffer(command_buffer, &begin_info)
	if result != .SUCCESS {
		panic("Command Buffer recording failed")
	}

	clear_color := vk.ClearValue {
		color = {float32 = {0, 0, 0, 1}},
	}

	render_pass_info := vk.RenderPassBeginInfo {
		sType = .RENDER_PASS_BEGIN_INFO,
		renderPass = render_pass,
		framebuffer = framebuffers[image_index],
		renderArea = vk.Rect2D{offset = {0, 0}, extent = extent},
		clearValueCount = 1,
		pClearValues = &clear_color,
	}

	vk.CmdBeginRenderPass(command_buffer, &render_pass_info, .INLINE)
	vk.CmdBindPipeline(command_buffer, .GRAPHICS, pipeline)
	viewport := vk.Viewport {
		x        = 0,
		y        = 0,
		width    = f32(extent.width),
		height   = f32(extent.height),
		minDepth = 0.0,
		maxDepth = 1.0,
	}
	vk.CmdSetViewport(command_buffer, 0, 1, &viewport)
	scissor := vk.Rect2D {
		offset = {0, 0},
		extent = extent,
	}
	vk.CmdSetScissor(command_buffer, 0, 1, &scissor)
	vk.CmdDraw(command_buffer, 3, 1, 0, 0)
	vk.CmdEndRenderPass(command_buffer)

	result = vk.EndCommandBuffer(command_buffer)
	if result != .SUCCESS {
		panic("Command Buffer recording failed")
	}
}

app_create_sync_objects :: proc(
	device: vk.Device,
) -> (
	image_available_semaphores: [2]vk.Semaphore,
	render_finished_semaphores: [2]vk.Semaphore,
	in_flight_fences: [2]vk.Fence,
) {
	semaphore_info := vk.SemaphoreCreateInfo {
		sType = .SEMAPHORE_CREATE_INFO,
	}

	fence_info := vk.FenceCreateInfo {
		sType = .FENCE_CREATE_INFO,
		flags = {.SIGNALED},
	}

	for i in 0 ..< 2 {
		result := vk.CreateSemaphore(device, &semaphore_info, nil, &image_available_semaphores[i])
		if result != .SUCCESS {
			panic("Sync object creation failed")
		}

		result = vk.CreateSemaphore(device, &semaphore_info, nil, &render_finished_semaphores[i])
		if result != .SUCCESS {
			panic("Sync object creation failed")
		}

		result = vk.CreateFence(device, &fence_info, nil, &in_flight_fences[i])
		if result != .SUCCESS {
			panic("Sync object creation failed")
		}
	}
	return
}

app_init :: proc() -> (app: App) {
	// Initialise Window
	SDL.Init({.VIDEO})
	window := SDL.CreateWindow(
		"Odin Vulkan Triangle",
		SDL.WINDOWPOS_CENTERED,
		SDL.WINDOWPOS_CENTERED,
		DEFAULT_WIDTH,
		DEFAULT_HEIGHT,
		{.ALLOW_HIGHDPI, .HIDDEN, .VULKAN},
	)

	// Initialise Vulkan Resources
	instance := app_create_instance(window)
	debug_messenger := app_create_debug_messenger(instance)
	surface := app_create_surface(window, instance)
	physical_device, queue_index := app_pick_physical_device(instance, surface)
	device, graphics_queue := app_create_device(instance, physical_device, queue_index)
	swapchain, images, format, extent := app_create_swapchain(physical_device, device, surface)
	image_views := app_create_image_views(device, images, format)
	render_pass := app_create_render_pass(device, format)
	pipeline, pipeline_layout := app_create_graphics_pipeline(device, render_pass)
	framebuffers := app_create_framebuffers(device, render_pass, extent, image_views)
	command_pool := app_create_command_pool(device, queue_index)
	command_buffers := app_create_command_buffers(device, command_pool)
	image_available_semaphores, render_finished_semaphores, in_flight_fences := app_create_sync_objects(device)

	SDL.ShowWindow(window)

	app = App {
		window                     = window,
		instance                   = instance,
		debug_messenger            = debug_messenger,
		physical_device            = physical_device,
		device                     = device,
		graphics_queue             = graphics_queue,
		surface                    = surface,
		swapchain                  = swapchain,
		swapchain_format           = format,
		swapchain_extent           = extent,
		swapchain_images           = images,
		swapchain_image_views      = image_views,
		render_pass                = render_pass,
		pipeline_layout            = pipeline_layout,
		pipeline                   = pipeline,
		framebuffers               = framebuffers,
		command_pool               = command_pool,
		command_buffers            = command_buffers,
		image_available_semaphores = image_available_semaphores,
		render_finished_semaphores = render_finished_semaphores,
		in_flight_fences           = in_flight_fences,
		quit                       = false,
	}
	return
}

app_quit :: proc(app: ^App) {
	using app
	for fence in in_flight_fences {
		vk.DestroyFence(device, fence, nil)
	}
	for semaphore in render_finished_semaphores {
		vk.DestroySemaphore(device, semaphore, nil)
	}
	for semaphore in image_available_semaphores {
		vk.DestroySemaphore(device, semaphore, nil)
	}
	vk.DestroyCommandPool(device, command_pool, nil)
	for frame in framebuffers {
		vk.DestroyFramebuffer(device, frame, nil)
	}
	delete(framebuffers)
	vk.DestroyPipeline(device, pipeline, nil)
	vk.DestroyPipelineLayout(device, pipeline_layout, nil)
	vk.DestroyRenderPass(device, render_pass, nil)
	for view in swapchain_image_views {
		vk.DestroyImageView(device, view, nil)
	}
	delete(swapchain_image_views)
	delete(swapchain_images)
	vk.DestroySwapchainKHR(device, swapchain, nil)
	vk.DestroyDevice(device, nil)
	vk.DestroySurfaceKHR(instance, surface, nil)
	vk.DestroyDebugUtilsMessengerEXT(instance, debug_messenger, nil)
	vk.DestroyInstance(instance, nil)
	SDL.DestroyWindow(app.window)
	SDL.Quit()
}

app_event :: proc(app: ^App, event: SDL.Event) {
	#partial switch event.type {
	case .QUIT:
		app.quit = true
	case .KEYDOWN:
		if event.key.keysym.sym == .ESCAPE {
			app.quit = true
		}
	}
}

app_draw_frame :: proc(app: ^App) {
	using app

	vk.WaitForFences(device, 1, &in_flight_fences[current_frame], true, max(u64))
	vk.ResetFences(device, 1, &in_flight_fences[current_frame])

	image_index: u32
	vk.AcquireNextImageKHR(
		device,
		swapchain,
		max(u64),
		image_available_semaphores[current_frame],
		vk.Fence(0),
		&image_index,
	)
	vk.ResetCommandBuffer(command_buffers[current_frame], {})
	app_record_command_buffer(
		device,
		framebuffers,
		pipeline,
		render_pass,
		swapchain_extent,
		image_index,
		command_buffers[current_frame],
	)

	wait_semaphores := [1]vk.Semaphore{image_available_semaphores[current_frame]}
	wait_stages := [1]vk.PipelineStageFlags{{.COLOR_ATTACHMENT_OUTPUT}}
	signal_semaphores := [1]vk.Semaphore{render_finished_semaphores[current_frame]}
	submit_info := vk.SubmitInfo {
		sType                = .SUBMIT_INFO,
		waitSemaphoreCount   = 1,
		pWaitSemaphores      = &wait_semaphores[0],
		pWaitDstStageMask    = &wait_stages[0],
		commandBufferCount   = 1,
		pCommandBuffers      = &command_buffers[current_frame],
		signalSemaphoreCount = 1,
		pSignalSemaphores    = &signal_semaphores[0],
	}

	result := vk.QueueSubmit(graphics_queue, 1, &submit_info, in_flight_fences[current_frame])
	if result != .SUCCESS {
		panic("Queue submission failed")
	}

	swapchains := [1]vk.SwapchainKHR{swapchain}
	present_info := vk.PresentInfoKHR {
		sType              = .PRESENT_INFO_KHR,
		waitSemaphoreCount = 1,
		pWaitSemaphores    = &signal_semaphores[0],
		swapchainCount     = 1,
		pSwapchains        = &swapchains[0],
		pImageIndices      = &image_index,
	}

	vk.QueuePresentKHR(graphics_queue, &present_info)
	current_frame = (current_frame + 1) % 2
}

app_run :: proc(app: ^App) {
	for e: SDL.Event; SDL.PollEvent(&e); {
		app_event(app, e)
		app_draw_frame(app)
	}
}

main :: proc() {
	app := app_init()
	defer app_quit(&app)

	for !app.quit {
		app_run(&app)
	}

        vk.DeviceWaitIdle(app.device)
}

// SHADER
//
// COMPILED WITH DXC
//
// struct v2f {
//     float4 position : SV_Position;
//     float3 color : COLOR;
// };
//
// void vertex_main(in uint id: SV_VertexID, out v2f vert_out) {
//     float2 positions[3] = {
//         float2(0.0, -0.5),
//         float2(0.5, 0.5),
//         float2(-0.5, 0.5)
//     };
//
//     float3 colors[3] = {
//         float3(1.0, 0.0, 0.0),
//         float3(0.0, 1.0, 0.0),
//         float3(0.0, 0.0, 1.0)
//     };
//
//     vert_out.position = float4(positions[id], 0.0, 1.0);
//     vert_out.color = colors[id];
// };
//
// float4 pixel_main(in v2f vert_in) : SV_Target {
//     return float4(vert_in.color, 1.0);   
// };
//

pixel_main_src := [?]u8{
	0x03, 0x02, 0x23, 0x07, 0x00, 0x06, 0x01, 0x00, 0x00, 0x00, 0x0e, 0x00,
	0x12, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x11, 0x00, 0x02, 0x00,
	0x01, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x01, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x08, 0x00, 0x04, 0x00, 0x00, 0x00,
	0x01, 0x00, 0x00, 0x00, 0x70, 0x69, 0x78, 0x65, 0x6c, 0x5f, 0x6d, 0x61,
	0x69, 0x6e, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
	0x10, 0x00, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00,
	0x03, 0x00, 0x03, 0x00, 0x05, 0x00, 0x00, 0x00, 0x9e, 0x02, 0x00, 0x00,
	0x05, 0x00, 0x06, 0x00, 0x02, 0x00, 0x00, 0x00, 0x69, 0x6e, 0x2e, 0x76,
	0x61, 0x72, 0x2e, 0x43, 0x4f, 0x4c, 0x4f, 0x52, 0x00, 0x00, 0x00, 0x00,
	0x05, 0x00, 0x07, 0x00, 0x03, 0x00, 0x00, 0x00, 0x6f, 0x75, 0x74, 0x2e,
	0x76, 0x61, 0x72, 0x2e, 0x53, 0x56, 0x5f, 0x54, 0x61, 0x72, 0x67, 0x65,
	0x74, 0x00, 0x00, 0x00, 0x05, 0x00, 0x05, 0x00, 0x01, 0x00, 0x00, 0x00,
	0x70, 0x69, 0x78, 0x65, 0x6c, 0x5f, 0x6d, 0x61, 0x69, 0x6e, 0x00, 0x00,
	0x47, 0x00, 0x04, 0x00, 0x02, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x47, 0x00, 0x04, 0x00, 0x03, 0x00, 0x00, 0x00,
	0x1e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x16, 0x00, 0x03, 0x00,
	0x04, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x2b, 0x00, 0x04, 0x00,
	0x04, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f,
	0x17, 0x00, 0x04, 0x00, 0x06, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
	0x04, 0x00, 0x00, 0x00, 0x17, 0x00, 0x04, 0x00, 0x07, 0x00, 0x00, 0x00,
	0x04, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x20, 0x00, 0x04, 0x00,
	0x08, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00,
	0x20, 0x00, 0x04, 0x00, 0x09, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
	0x06, 0x00, 0x00, 0x00, 0x13, 0x00, 0x02, 0x00, 0x0a, 0x00, 0x00, 0x00,
	0x21, 0x00, 0x03, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00,
	0x3b, 0x00, 0x04, 0x00, 0x08, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
	0x01, 0x00, 0x00, 0x00, 0x3b, 0x00, 0x04, 0x00, 0x09, 0x00, 0x00, 0x00,
	0x03, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x36, 0x00, 0x05, 0x00,
	0x0a, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x0b, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x02, 0x00, 0x0c, 0x00, 0x00, 0x00,
	0x3d, 0x00, 0x04, 0x00, 0x07, 0x00, 0x00, 0x00, 0x0d, 0x00, 0x00, 0x00,
	0x02, 0x00, 0x00, 0x00, 0x51, 0x00, 0x05, 0x00, 0x04, 0x00, 0x00, 0x00,
	0x0e, 0x00, 0x00, 0x00, 0x0d, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x51, 0x00, 0x05, 0x00, 0x04, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x00,
	0x0d, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x51, 0x00, 0x05, 0x00,
	0x04, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x0d, 0x00, 0x00, 0x00,
	0x02, 0x00, 0x00, 0x00, 0x50, 0x00, 0x07, 0x00, 0x06, 0x00, 0x00, 0x00,
	0x11, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x00, 0x00,
	0x10, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x3e, 0x00, 0x03, 0x00,
	0x03, 0x00, 0x00, 0x00, 0x11, 0x00, 0x00, 0x00, 0xfd, 0x00, 0x01, 0x00,
	0x38, 0x00, 0x01, 0x00}

vertex_main_src := [?]u8{
	0x03, 0x02, 0x23, 0x07, 0x00, 0x06, 0x01, 0x00, 0x00, 0x00, 0x0e, 0x00,
	0x2d, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x11, 0x00, 0x02, 0x00,
	0x01, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x01, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x09, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x01, 0x00, 0x00, 0x00, 0x76, 0x65, 0x72, 0x74, 0x65, 0x78, 0x5f, 0x6d,
	0x61, 0x69, 0x6e, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
	0x04, 0x00, 0x00, 0x00, 0x03, 0x00, 0x03, 0x00, 0x05, 0x00, 0x00, 0x00,
	0x9e, 0x02, 0x00, 0x00, 0x05, 0x00, 0x06, 0x00, 0x04, 0x00, 0x00, 0x00,
	0x6f, 0x75, 0x74, 0x2e, 0x76, 0x61, 0x72, 0x2e, 0x43, 0x4f, 0x4c, 0x4f,
	0x52, 0x00, 0x00, 0x00, 0x05, 0x00, 0x05, 0x00, 0x01, 0x00, 0x00, 0x00,
	0x76, 0x65, 0x72, 0x74, 0x65, 0x78, 0x5f, 0x6d, 0x61, 0x69, 0x6e, 0x00,
	0x47, 0x00, 0x04, 0x00, 0x02, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00,
	0x2a, 0x00, 0x00, 0x00, 0x47, 0x00, 0x04, 0x00, 0x03, 0x00, 0x00, 0x00,
	0x0b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x47, 0x00, 0x04, 0x00,
	0x04, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x16, 0x00, 0x03, 0x00, 0x05, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00,
	0x2b, 0x00, 0x04, 0x00, 0x05, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x2b, 0x00, 0x04, 0x00, 0x05, 0x00, 0x00, 0x00,
	0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xbf, 0x2b, 0x00, 0x04, 0x00,
	0x05, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3f,
	0x2b, 0x00, 0x04, 0x00, 0x05, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00,
	0x00, 0x00, 0x80, 0x3f, 0x15, 0x00, 0x04, 0x00, 0x0a, 0x00, 0x00, 0x00,
	0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20, 0x00, 0x04, 0x00,
	0x0b, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00,
	0x17, 0x00, 0x04, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00,
	0x04, 0x00, 0x00, 0x00, 0x20, 0x00, 0x04, 0x00, 0x0d, 0x00, 0x00, 0x00,
	0x03, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x17, 0x00, 0x04, 0x00,
	0x0e, 0x00, 0x00, 0x00, 0x05, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
	0x20, 0x00, 0x04, 0x00, 0x0f, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
	0x0e, 0x00, 0x00, 0x00, 0x13, 0x00, 0x02, 0x00, 0x10, 0x00, 0x00, 0x00,
	0x21, 0x00, 0x03, 0x00, 0x11, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
	0x2b, 0x00, 0x04, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x12, 0x00, 0x00, 0x00,
	0x03, 0x00, 0x00, 0x00, 0x17, 0x00, 0x04, 0x00, 0x13, 0x00, 0x00, 0x00,
	0x05, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x04, 0x00,
	0x14, 0x00, 0x00, 0x00, 0x13, 0x00, 0x00, 0x00, 0x12, 0x00, 0x00, 0x00,
	0x20, 0x00, 0x04, 0x00, 0x15, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00,
	0x14, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x04, 0x00, 0x16, 0x00, 0x00, 0x00,
	0x0e, 0x00, 0x00, 0x00, 0x12, 0x00, 0x00, 0x00, 0x20, 0x00, 0x04, 0x00,
	0x17, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0x16, 0x00, 0x00, 0x00,
	0x20, 0x00, 0x04, 0x00, 0x18, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00,
	0x13, 0x00, 0x00, 0x00, 0x20, 0x00, 0x04, 0x00, 0x19, 0x00, 0x00, 0x00,
	0x07, 0x00, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00, 0x3b, 0x00, 0x04, 0x00,
	0x0b, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
	0x3b, 0x00, 0x04, 0x00, 0x0d, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
	0x03, 0x00, 0x00, 0x00, 0x3b, 0x00, 0x04, 0x00, 0x0f, 0x00, 0x00, 0x00,
	0x04, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x2c, 0x00, 0x05, 0x00,
	0x13, 0x00, 0x00, 0x00, 0x1a, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00,
	0x07, 0x00, 0x00, 0x00, 0x2c, 0x00, 0x05, 0x00, 0x13, 0x00, 0x00, 0x00,
	0x1b, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00,
	0x2c, 0x00, 0x05, 0x00, 0x13, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00,
	0x07, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x2c, 0x00, 0x06, 0x00,
	0x14, 0x00, 0x00, 0x00, 0x1d, 0x00, 0x00, 0x00, 0x1a, 0x00, 0x00, 0x00,
	0x1b, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00, 0x2c, 0x00, 0x06, 0x00,
	0x0e, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00,
	0x06, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x2c, 0x00, 0x06, 0x00,
	0x0e, 0x00, 0x00, 0x00, 0x1f, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00,
	0x09, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, 0x2c, 0x00, 0x06, 0x00,
	0x0e, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00,
	0x06, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x2c, 0x00, 0x06, 0x00,
	0x16, 0x00, 0x00, 0x00, 0x21, 0x00, 0x00, 0x00, 0x1e, 0x00, 0x00, 0x00,
	0x1f, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x36, 0x00, 0x05, 0x00,
	0x10, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	0x11, 0x00, 0x00, 0x00, 0xf8, 0x00, 0x02, 0x00, 0x22, 0x00, 0x00, 0x00,
	0x3b, 0x00, 0x04, 0x00, 0x15, 0x00, 0x00, 0x00, 0x23, 0x00, 0x00, 0x00,
	0x07, 0x00, 0x00, 0x00, 0x3b, 0x00, 0x04, 0x00, 0x17, 0x00, 0x00, 0x00,
	0x24, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0x3d, 0x00, 0x04, 0x00,
	0x0a, 0x00, 0x00, 0x00, 0x25, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
	0x3e, 0x00, 0x03, 0x00, 0x23, 0x00, 0x00, 0x00, 0x1d, 0x00, 0x00, 0x00,
	0x3e, 0x00, 0x03, 0x00, 0x24, 0x00, 0x00, 0x00, 0x21, 0x00, 0x00, 0x00,
	0x41, 0x00, 0x05, 0x00, 0x18, 0x00, 0x00, 0x00, 0x26, 0x00, 0x00, 0x00,
	0x23, 0x00, 0x00, 0x00, 0x25, 0x00, 0x00, 0x00, 0x3d, 0x00, 0x04, 0x00,
	0x13, 0x00, 0x00, 0x00, 0x27, 0x00, 0x00, 0x00, 0x26, 0x00, 0x00, 0x00,
	0x51, 0x00, 0x05, 0x00, 0x05, 0x00, 0x00, 0x00, 0x28, 0x00, 0x00, 0x00,
	0x27, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x51, 0x00, 0x05, 0x00,
	0x05, 0x00, 0x00, 0x00, 0x29, 0x00, 0x00, 0x00, 0x27, 0x00, 0x00, 0x00,
	0x01, 0x00, 0x00, 0x00, 0x50, 0x00, 0x07, 0x00, 0x0c, 0x00, 0x00, 0x00,
	0x2a, 0x00, 0x00, 0x00, 0x28, 0x00, 0x00, 0x00, 0x29, 0x00, 0x00, 0x00,
	0x06, 0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x41, 0x00, 0x05, 0x00,
	0x19, 0x00, 0x00, 0x00, 0x2b, 0x00, 0x00, 0x00, 0x24, 0x00, 0x00, 0x00,
	0x25, 0x00, 0x00, 0x00, 0x3d, 0x00, 0x04, 0x00, 0x0e, 0x00, 0x00, 0x00,
	0x2c, 0x00, 0x00, 0x00, 0x2b, 0x00, 0x00, 0x00, 0x3e, 0x00, 0x03, 0x00,
	0x03, 0x00, 0x00, 0x00, 0x2a, 0x00, 0x00, 0x00, 0x3e, 0x00, 0x03, 0x00,
	0x04, 0x00, 0x00, 0x00, 0x2c, 0x00, 0x00, 0x00, 0xfd, 0x00, 0x01, 0x00,
	0x38, 0x00, 0x01, 0x00}