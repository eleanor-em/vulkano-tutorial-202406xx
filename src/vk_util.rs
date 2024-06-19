use std::sync::Arc;

use anyhow::{Context, Result};
use tracing::info;

use vulkano::device::physical::PhysicalDevice;
use vulkano::device::{Device, DeviceCreateInfo, Queue, QueueCreateInfo, QueueFlags};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::VulkanLibrary;

pub fn macos_instance(library: Arc<VulkanLibrary>) -> Result<Arc<Instance>> {
    let instance_create_info = InstanceCreateInfo {
        flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
        ..Default::default()
    };
    Instance::new(library, instance_create_info)
        .context("vulkano: failed to create instance")
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
            // here we pass the desired queue family to use by index
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