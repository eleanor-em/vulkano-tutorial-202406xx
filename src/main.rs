use anyhow::{Context, Result};
use vulkano::VulkanLibrary;

mod vk_util;
mod vk_test;

fn main() -> Result<()>{
    // install global collector configured based on RUST_LOG env var
    tracing_subscriber::fmt()
        .event_format(
            tracing_subscriber::fmt::format()
                .with_target(false)
                .with_file(true)
                .with_line_number(true)
        )
        .init();

    // vulkano setup
    let library = VulkanLibrary::new()
        .context("vulkano: no local Vulkan library/DLL")?;
    let instance = vk_util::macos_instance(library)?;
    let physical_device = vk_util::any_physical_device(instance)?;
    let (device, queue) = vk_util::any_graphical_queue_family(physical_device)?;

    // run tests
    vk_test::s3_buffer_creation(device, queue)?;

    Ok(())
}
