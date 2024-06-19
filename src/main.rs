use std::time::Duration;
use anyhow::Result;

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

    let ctx = vk_util::TestContext::new()?;
    vk_test::s3_buffer_creation(ctx.clone())?;
    vk_test::s4_compute_operations(ctx.clone())?;

    // give everything time to stabilise as needed (e.g. to open images from tests)
    std::thread::sleep(Duration::from_millis(100));
    Ok(())
}
