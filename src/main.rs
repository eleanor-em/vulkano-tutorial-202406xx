#![feature(iterator_try_collect)]

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

    let window_ctx = vk_util::WindowContext::new()?;
    let ctx = vk_util::TestContext::new(&window_ctx)?;
    vk_test::s3_buffer_creation(ctx.clone())?;
    vk_test::s4_compute_operations(ctx.clone())?;
    vk_test::s5_image_creation(ctx.clone())?;
    vk_test::s6_graphics_pipeline(ctx.clone())?;
    vk_test::s7_windowing(window_ctx, ctx.clone())
}
