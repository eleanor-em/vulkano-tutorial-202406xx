use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use tracing::info;
use vulkano::command_buffer::allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;

use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags};
use vulkano::image::{Image, ImageUsage};
use vulkano::image::view::ImageView;
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass};
use vulkano::swapchain::{Surface, Swapchain, SwapchainCreateInfo};
use vulkano::VulkanLibrary;

use winit::event_loop::EventLoop;
use winit::window::{Window, WindowBuilder};

pub struct WindowContext {
    event_loop: EventLoop<()>,
    window: Arc<Window>,
}

impl WindowContext {
    pub fn new() -> Result<Self> {
        let event_loop = EventLoop::new();
        let window = Arc::new(WindowBuilder::new().build(&event_loop)?);
        Ok(Self { event_loop, window })
    }

    // pub fn run_event_loop<F>(self, action: F) -> Result<()> {
    //     self.event_loop.run(action);
    // }
    pub fn event_loop(&self) -> &EventLoop<()> { &self.event_loop }
    pub fn window(&self) -> Arc<Window> { self.window.clone() }

    pub fn consume(self) -> (EventLoop<()>, Arc<Window>) { (self.event_loop, self.window) }
}

#[derive(Clone)]
pub struct TestContext {
    surface: Arc<Surface>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    swapchain: Arc<Swapchain>,
    images: Vec<Arc<Image>>,
    render_pass: Arc<RenderPass>,
    framebuffers: Vec<Arc<Framebuffer>>,
}

fn device_extensions() -> DeviceExtensions {
    DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    }
}

impl TestContext {
    pub fn new(window_ctx: &WindowContext) -> Result<Self> {
        let start = Instant::now();
        let library = VulkanLibrary::new()
            .context("vulkano: no local Vulkan library/DLL")?;
        let instance = macos_instance(window_ctx.event_loop(), library)?;
        let surface = Surface::from_window(instance.clone(), window_ctx.window())?;
        let physical_device = any_physical_device(instance.clone(), surface.clone())?;
        let (device, queue) = any_graphical_queue_family(physical_device.clone())?;
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        ));
        let descriptor_set_allocator = Arc::new(
            StandardDescriptorSetAllocator::new(device.clone(), Default::default()));
        let (swapchain, images) = create_swapchain(window_ctx.window(), surface.clone(), physical_device.clone(), device.clone())?;
        let render_pass = create_render_pass(device.clone(), swapchain.clone())?;
        let framebuffers = create_framebuffers(&images, render_pass.clone())?;

        info!("created vulkano context in: {} ms", start.elapsed().as_millis());
        Ok(Self {
            surface, device, queue,
            memory_allocator, command_buffer_allocator, descriptor_set_allocator,
            swapchain, images, render_pass, framebuffers,
        })
    }

    pub fn device(&self) -> Arc<Device> { self.device.clone() }
    pub fn queue(&self) -> Arc<Queue> { self.queue.clone() }
    pub fn memory_allocator(&self) -> Arc<StandardMemoryAllocator> { self.memory_allocator.clone() }
    pub fn command_buffer_allocator(&self) -> Arc<StandardCommandBufferAllocator> { self.command_buffer_allocator.clone() }
    pub fn descriptor_set_allocator(&self) -> Arc<StandardDescriptorSetAllocator> { self.descriptor_set_allocator.clone() }
    pub fn surface(&self) -> Arc<Surface> { self.surface.clone() }
    pub fn swapchain(&self) -> Arc<Swapchain> { self.swapchain.clone() }
    pub fn images(&self) -> Vec<Arc<Image>> { self.images.clone() }
    pub fn render_pass(&self) -> Arc<RenderPass> { self.render_pass.clone() }
    pub fn framebuffers(&self) -> Vec<Arc<Framebuffer>> { self.framebuffers.clone() }

    pub fn recreate_swapchain(&mut self, window: Arc<Window>) -> Result<()>{
        let new_dimensions = window.inner_size();

        let (new_swapchain, new_images) = self.swapchain
            .recreate(SwapchainCreateInfo {
                // Here, `image_extend` will correspond to the window dimensions.
                image_extent: new_dimensions.into(),
                ..self.swapchain.create_info()
            })?;
        self.swapchain = new_swapchain;
        self.framebuffers = create_framebuffers(&new_images, self.render_pass.clone())?;
        Ok(())
    }
}

fn macos_instance<T>(event_loop: &EventLoop<T>, library: Arc<VulkanLibrary>) -> Result<Arc<Instance>> {
    let required_extensions = Surface::required_extensions(event_loop);
    let instance_create_info = InstanceCreateInfo {
        flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
        enabled_extensions: required_extensions,
        ..Default::default()
    };
    Instance::new(library, instance_create_info).context("vulkano: failed to create instance")
}
fn any_physical_device(instance: Arc<Instance>, surface: Arc<Surface>) -> Result<Arc<PhysicalDevice>> {
    Ok(instance.enumerate_physical_devices()?
        .filter(|p| p.supported_extensions().contains(&device_extensions()))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    q.queue_flags.contains(QueueFlags::GRAPHICS)
                        && p.surface_support(i as u32, &surface).unwrap_or(false)
                })
                .map(|q| (p, q as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            _ => 4,
        }).context("vulkano: no appropriate physical device available")?.0)
}
fn any_graphical_queue_family(physical_device: Arc<PhysicalDevice>) -> Result<(Arc<Device>, Arc<Queue>)> {
    let queue_family_index = physical_device.queue_family_properties()
        .iter().enumerate()
        .position(|(_queue_family_index, queue_family_properties)| {
            queue_family_properties.queue_flags.contains(QueueFlags::GRAPHICS)
        })
        .context("vulkano: couldn't find a graphical queue family")? as u32;
    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            enabled_extensions: device_extensions(),
            ..Default::default()
        },
    ).context("vulkano: failed to create device")?;
    info!("found {} queue(s), using first", queues.len());
    let queue = queues.next().context("vulkano: UNEXPECTED: zero queues?")?;
    Ok((device, queue))
}

fn create_swapchain(window: Arc<Window>,
                    surface: Arc<Surface>,
                    physical_device: Arc<PhysicalDevice>,
                    device: Arc<Device>) -> Result<(Arc<Swapchain>, Vec<Arc<Image>>)> {
    let caps = physical_device
        .surface_capabilities(&surface, Default::default())?;
    let dimensions = window.inner_size();
    let composite_alpha = caps.supported_composite_alpha.into_iter().next().unwrap();
    let image_format =  physical_device
        .surface_formats(&surface, Default::default())?
        .first().context("vulkano: no surface formats found")?
        .0;
    Ok(Swapchain::new(
        device.clone(),
        surface.clone(),
        SwapchainCreateInfo {
            min_image_count: caps.min_image_count + 1, // How many buffers to use in the swapchain
            image_format,
            image_extent: dimensions.into(),
            image_usage: ImageUsage::COLOR_ATTACHMENT, // What the images are going to be used for
            composite_alpha,
            ..Default::default()
        },
    )?)
}

fn create_render_pass(device: Arc<Device>, swapchain: Arc<Swapchain>) -> Result<Arc<RenderPass>> {
    Ok(vulkano::single_pass_renderpass!(
        device.clone(),
        attachments: {
            color: {
                // Set the format the same as the swapchain.
                format: swapchain.image_format(),
                samples: 1,
                load_op: Clear,
                store_op: Store,
            },
        },
        pass: {
            color: [color],
            depth_stencil: {},
        },
    )?)
}

fn create_framebuffers(images: &[Arc<Image>], render_pass: Arc<RenderPass>) -> Result<Vec<Arc<Framebuffer>>> {
    Ok(images.iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )
        }).try_collect()?)
}
