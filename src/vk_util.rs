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
use vulkano::pipeline::graphics::viewport::Viewport;
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass};
use vulkano::swapchain::{PresentFuture, Surface, Swapchain, SwapchainAcquireFuture, SwapchainCreateInfo, SwapchainPresentInfo};
use vulkano::{swapchain, Validated, VulkanError, VulkanLibrary};
use vulkano::command_buffer::{CommandBufferExecFuture, PrimaryAutoCommandBuffer};
use vulkano::sync::future::{FenceSignalFuture, JoinFuture};
use vulkano::sync::GpuFuture;
use winit::event::{Event, WindowEvent};

use winit::event_loop::{ControlFlow, EventLoop};
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

    pub fn event_loop(&self) -> &EventLoop<()> { &self.event_loop }
    pub fn window(&self) -> Arc<Window> { self.window.clone() }

    pub fn consume(self) -> (EventLoop<()>, Arc<Window>) { (self.event_loop, self.window) }

    pub fn create_default_viewport(&self) -> Viewport {
        Viewport {
            offset: [0.0, 0.0],
            extent: self.window.inner_size().into(),
            depth_range: 0.0..=1.0,
        }
    }
}

#[derive(Clone)]
pub struct VulkanoContext {
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
    DeviceExtensions { khr_swapchain: true, ..DeviceExtensions::empty() }
}

impl VulkanoContext {
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
            device.clone(), StandardCommandBufferAllocatorCreateInfo::default()));
        let descriptor_set_allocator = Arc::new(
            StandardDescriptorSetAllocator::new(device.clone(), Default::default()));
        let (swapchain, images) = create_swapchain(
            window_ctx.window(), surface.clone(), physical_device.clone(), device.clone())?;
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
        let (new_swapchain, new_images) = self.swapchain
            .recreate(SwapchainCreateInfo {
                image_extent: window.inner_size().into(),
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
        }).context("vulkano: couldn't find a graphical queue family")? as u32;
    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo { queue_family_index, ..Default::default() }],
            enabled_extensions: device_extensions(),
            ..Default::default()
        })?;
    info!("found {} queue(s), using first", queues.len());
    Ok((device, queues.next().context("vulkano: UNEXPECTED: zero queues?")?))
}

fn create_swapchain(window: Arc<Window>,
                    surface: Arc<Surface>,
                    physical_device: Arc<PhysicalDevice>,
                    device: Arc<Device>) -> Result<(Arc<Swapchain>, Vec<Arc<Image>>)> {
    let caps = physical_device.surface_capabilities(&surface, Default::default())?;
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

pub type CommandBuffer = PrimaryAutoCommandBuffer<Arc<StandardCommandBufferAllocator>>;

pub trait RenderEventHandler {
    fn on_resize(&mut self, ctx: &VulkanoContext, window: Arc<Window>) -> Result<()>;
    fn on_render(&mut self) ->  Result<Vec<Arc<CommandBuffer>>>;
}

pub struct WindowEventHandler<RenderHandler: RenderEventHandler> {
    window: Arc<Window>,
    ctx: VulkanoContext,
    render_handler: RenderHandler,

    window_was_resized: bool,
    should_recreate_swapchain: bool,
    // XXX: FenceSignalFuture is not Send + Sync, so Arc is cheating. However, there is no
    // `impl GpuFuture for Rc<FenceSignalFuture<...>>`, so we can't use Rc. We don't use threads yet
    // so this is safe anyway.
    fences: Vec<Option<Arc<FenceSignalFuture<FenceFuture>>>>,
    last_fence_idx: u32,
}

type FenceFuture = PresentFuture<CommandBufferExecFuture<JoinFuture<Box<dyn GpuFuture>, SwapchainAcquireFuture>>>;
impl<RenderHandler: RenderEventHandler + 'static> WindowEventHandler<RenderHandler> {
    pub fn new(window: Arc<Window>,
               ctx: VulkanoContext,
               handler: RenderHandler) -> Self {
        let frames_in_flight = ctx.images().len();
        Self {
            window, ctx,
            render_handler: handler,
            window_was_resized: false, should_recreate_swapchain: false,
            fences: vec![None; frames_in_flight], last_fence_idx: 0,
        }
    }

    pub fn run(mut self, event_loop: EventLoop<()>) {
        event_loop.run(move |event, _, control_flow| self.run_inner(event, control_flow).unwrap());
    }

    fn maybe_recreate_swapchain(&mut self) -> Result<()> {
        if self.window_was_resized || self.should_recreate_swapchain {
            self.should_recreate_swapchain = false;
            self.ctx.recreate_swapchain(self.window.clone())
                .context("could not recreate swapchain")?;
        }
        if self.window_was_resized {
            self.window_was_resized = false;
            self.render_handler.on_resize(&self.ctx, self.window.clone())?;
        }
        Ok(())
    }

    fn handle_acquired_image(&mut self, image_idx: u32, acquire_future: SwapchainAcquireFuture) -> Result<()> {
        if let Some(image_fence) = &self.fences[image_idx as usize] {
            image_fence.wait(None)?;
        }
        let previous_future = match self.fences[self.last_fence_idx as usize].clone() {
            None => {
                let mut now = vulkano::sync::now(self.ctx.device());
                now.cleanup_finished();
                now.boxed()
            }
            Some(fence) => fence.boxed(),
        };

        let command_buffers = self.render_handler.on_render()?;
        let future = previous_future.join(acquire_future)
            .then_execute(self.ctx.queue(), command_buffers[image_idx as usize].clone())?
            .then_swapchain_present(
                self.ctx.queue(),
                SwapchainPresentInfo::swapchain_image_index(self.ctx.swapchain(), image_idx),
            )
            .then_signal_fence_and_flush()
            .map_err(Validated::unwrap);

        self.fences[image_idx as usize] = match future {
            Ok(value) => Some(Arc::new(value)),
            Err(VulkanError::OutOfDate) => {
                self.should_recreate_swapchain = true;
                None
            },
            Err(e) => return Err(e.into()),
        };
        self.last_fence_idx = image_idx;
        Ok(())
    }

    fn run_inner(&mut self, event: Event<()>, control_flow: &mut ControlFlow) -> Result<()> {
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested, ..
            } => {
                *control_flow = ControlFlow::Exit;
                Ok(())
            },
            Event::WindowEvent {
                event: WindowEvent::Resized(_), ..
            } => {
                self.window_was_resized = true;
                Ok(())
            },
            Event::MainEventsCleared => {
                self.maybe_recreate_swapchain()?;
                match swapchain::acquire_next_image(self.ctx.swapchain(), None).map_err(Validated::unwrap) {
                    Ok((image_idx, suboptimal, acquire_future)) => {
                        if suboptimal { self.should_recreate_swapchain = true; }
                        self.handle_acquired_image(image_idx, acquire_future)
                    },
                    Err(VulkanError::OutOfDate) => {
                        self.should_recreate_swapchain = true;
                        Ok(())
                    },
                    Err(e) => Err(e.into()),
                }
            },
            _ => Ok(()),
        }
    }
}
