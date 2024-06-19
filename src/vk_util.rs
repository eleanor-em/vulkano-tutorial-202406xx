use std::sync::Arc;

use anyhow::{Context, Result};
use tracing::info;
use vulkano::command_buffer::allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;

use vulkano::device::physical::PhysicalDevice;
use vulkano::device::{Device, DeviceCreateInfo, Queue, QueueCreateInfo, QueueFlags};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::VulkanLibrary;
use crate::vk_util;

#[derive(Clone)]
pub struct TestContext {
    device: Arc<Device>,
    queue: Arc<Queue>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
}

impl TestContext {
    pub fn new() -> Result<Self> {
        let library = VulkanLibrary::new()
            .context("vulkano: no local Vulkan library/DLL")?;
        let instance = vk_util::macos_instance(library)?;
        let physical_device = vk_util::any_physical_device(instance)?;
        let (device, queue) = vk_util::any_graphical_queue_family(physical_device)?;
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        ));
        let descriptor_set_allocator = Arc::new(
            StandardDescriptorSetAllocator::new(device.clone(), Default::default()));
        Ok(Self { device, queue, memory_allocator, command_buffer_allocator, descriptor_set_allocator })
    }

    pub fn device(&self) -> Arc<Device> { self.device.clone() }
    pub fn queue(&self) -> Arc<Queue> { self.queue.clone() }
    pub fn memory_allocator(&self) -> Arc<StandardMemoryAllocator> { self.memory_allocator.clone() }
    pub fn command_buffer_allocator(&self) -> Arc<StandardCommandBufferAllocator> { self.command_buffer_allocator.clone() }
    pub fn descriptor_set_allocator(&self) -> Arc<StandardDescriptorSetAllocator> { self.descriptor_set_allocator.clone() }
}

pub fn macos_instance(library: Arc<VulkanLibrary>) -> Result<Arc<Instance>> {
    let instance_create_info = InstanceCreateInfo {
        flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
        ..Default::default()
    };
    Instance::new(library, instance_create_info).context("vulkano: failed to create instance")
}
pub fn any_physical_device(instance: Arc<Instance>) -> anyhow::Result<Arc<PhysicalDevice>> {
    let mut all_physical_devices = instance.enumerate_physical_devices()
            .context("vulkano: could not enumerate physical devices")?;
    info!("found {} physical device(s), using first", all_physical_devices.len());
    all_physical_devices.next().context("vulkano: no physical devices available")
}
pub fn any_graphical_queue_family(physical_device: Arc<PhysicalDevice>) -> Result<(Arc<Device>, Arc<Queue>)> {
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
            ..Default::default()
        },
    ).context("vulkano: failed to create device")?;
    info!("found {} queue(s), using first", queues.len());
    let queue = queues.next().context("vulkano: UNEXPECTED: zero queues?")?;
    Ok((device, queue))
}
