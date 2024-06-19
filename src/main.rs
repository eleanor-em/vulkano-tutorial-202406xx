mod vk_util;

use anyhow::{Context, Result};

use vulkano::VulkanLibrary;
use crate::vk_util::any_graphical_queue_family;

fn main() -> Result<()>{
    // install global collector configured based on RUST_LOG env var
    tracing_subscriber::fmt()
        .event_format(
            tracing_subscriber::fmt::format()
                .with_file(true)
                .with_line_number(true)
        )
        .init();

    // vulkano setup
    let library = VulkanLibrary::new()
        .context("vulkano: no local Vulkan library/DLL")?;
    let instance = vk_util::macos_instance(library)?;
    let physical_device = vk_util::any_physical_device(instance)?;
    let (device, queue) = any_graphical_queue_family(physical_device)?;

    Ok(())
}

