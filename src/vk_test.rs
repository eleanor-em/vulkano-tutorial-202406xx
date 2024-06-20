use std::process::Command;
use anyhow::{Context, Result};
use tracing::info;

use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, CopyImageToBufferInfo};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::image::view::ImageView;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::sync::GpuFuture;

use crate::vk_util;

pub fn s3_buffer_creation(ctx: vk_util::TestContext) -> Result<()> {
    let src_content: Vec<i32> = (0..64).collect();
    let src = Buffer::from_iter(
        ctx.memory_allocator(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        src_content,
    ).context("failed to create source buffer")?;

    let dest_content: Vec<i32> = (0..64).map(|_| 0).collect();
    let dest = Buffer::from_iter(
        ctx.memory_allocator(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        dest_content,
    ).context("failed to create destination buffer")?;

    let mut builder = AutoCommandBufferBuilder::primary(
        &ctx.command_buffer_allocator(),
        ctx.queue().queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )?;
    builder.copy_buffer(CopyBufferInfo::buffers(src.clone(), dest.clone()))?;
    let command_buffer = builder.build()?;

    vulkano::sync::now(ctx.device())
        .then_execute(ctx.queue(), command_buffer)?
        .then_signal_fence_and_flush()?
        .wait(None)?;

    let src_content = src.read()?;
    let destination_content = dest.read()?;
    assert_eq!(&*src_content, &*destination_content);

    info!("[s3_buffer_creation] succeeded!");
    Ok(())
}

mod s4_compute_shader {
    vulkano_shaders::shader!{
        ty: "compute",
        src: r"
            #version 460

            layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

            layout(set = 0, binding = 0) buffer Data {
                uint data[];
            } buf;

            void main() {
                uint idx = gl_GlobalInvocationID.x;
                buf.data[idx] *= 12;
            }
        ",
    }
}

pub fn s4_compute_operations(ctx: vk_util::TestContext) -> Result<()> {
    // create buffers
    let data_iter = 0..65536u32;
    let data_buffer = Buffer::from_iter(
        ctx.memory_allocator(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        data_iter,
    ).context("failed to create buffer")?;

    // load shader and compute pipeline
    let shader = s4_compute_shader::load(ctx.device()).context("failed to create shader module")?;
    let cs = shader.entry_point("main").context("did not find shader entry point")?;
    let stage = PipelineShaderStageCreateInfo::new(cs);
    let layout = PipelineLayout::new(
        ctx.device(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
            .into_pipeline_layout_create_info(ctx.device())?
    )?;
    let compute_pipeline = ComputePipeline::new(
        ctx.device(),
        /* cache= */ None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    ).context("failed to create compute pipeline")?;

    // load descriptor set
    let pipeline_layout = compute_pipeline.layout();
    let descriptor_set_layouts = pipeline_layout.set_layouts();
    let descriptor_set_layout_index = 0;
    let descriptor_set_layout = descriptor_set_layouts
        .get(descriptor_set_layout_index)
        .context("no descriptor sets found in shader")?;
    let set = PersistentDescriptorSet::new(
        &ctx.descriptor_set_allocator(),
        descriptor_set_layout.clone(),
        [WriteDescriptorSet::buffer(0, data_buffer.clone())],
        [],
    )?;

    // create command buffer
    let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
        &ctx.command_buffer_allocator(),
        ctx.queue().queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )?;
    let work_group_counts = [1024, 1, 1];
    command_buffer_builder
        .bind_pipeline_compute(compute_pipeline.clone())?
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            compute_pipeline.layout().clone(),
            descriptor_set_layout_index as u32,
            set,
        )?
        .dispatch(work_group_counts)?;
    let command_buffer = command_buffer_builder.build()?;

    // execute command buffer
    vulkano::sync::now(ctx.device())
        .then_execute(ctx.queue(), command_buffer)?
        .then_signal_fence_and_flush()?
        .wait(None)?;

    let content = data_buffer.read().unwrap();
    for (n, val) in content.iter().enumerate() {
        assert_eq!(*val, n as u32 * 12);
    }

    info!("[s4_compute_operations] succeeded!");
    Ok(())
}

mod s5_compute_shader {
    vulkano_shaders::shader!{
        ty: "compute",
        src: r"
            #version 460

            layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

            layout(set = 0, binding = 0, rgba8) uniform writeonly image2D img;

            void main() {
                vec2 norm_coordinates = (gl_GlobalInvocationID.xy + vec2(0.5)) / vec2(imageSize(img));
                vec2 c = (norm_coordinates - vec2(0.5)) * 2.0 - vec2(1.0, 0.0);

                vec2 z = vec2(0.0, 0.0);
                float i;
                for (i = 0.0; i < 1.0; i += 0.005) {
                    z = vec2(
                        z.x * z.x - z.y * z.y + c.x,
                        z.y * z.x + z.x * z.y + c.y
                    );

                    if (length(z) > 4.0) {
                        break;
                    }
                }

                vec4 to_write = vec4(vec3(i), 1.0);
                imageStore(img, ivec2(gl_GlobalInvocationID.xy), to_write);
            }
        ",
    }
}

pub fn s5_image_creation(ctx: vk_util::TestContext) -> Result<()> {
    // create image and buffer
    let image = Image::new(
        ctx.memory_allocator(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: vulkano::format::Format::R8G8B8A8_UNORM,
            extent: [1024, 1024, 1],
            usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
    )?;
    let view = ImageView::new_default(image.clone())?;
    let shader = s5_compute_shader::load(ctx.device()).context("failed to create shader module")?;
    let cs = shader.entry_point("main").context("did not find shader entry point")?;
    let stage = PipelineShaderStageCreateInfo::new(cs);
    let pipeline_layout = PipelineLayout::new(
        ctx.device(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
            .into_pipeline_layout_create_info(ctx.device())?
    )?;
    let compute_pipeline = ComputePipeline::new(
        ctx.device(),
        /* cache= */ None,
        ComputePipelineCreateInfo::stage_layout(stage.clone(), pipeline_layout),
    ).context("failed to create compute pipeline")?;
    let pipeline_layout = compute_pipeline.layout();
    let descriptor_set_layouts = pipeline_layout.set_layouts();
    let descriptor_set_layout_index = 0;
    let descriptor_set_layout = descriptor_set_layouts
        .get(descriptor_set_layout_index)
        .context("no descriptor sets found in shader")?;
    let set = PersistentDescriptorSet::new(
        &ctx.descriptor_set_allocator(),
        descriptor_set_layout.clone(),
        [WriteDescriptorSet::image_view(0, view.clone())],
        [],
    )?;

    let buf = Buffer::from_iter(
        ctx.memory_allocator(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        (0..1024 * 1024 * 4).map(|_| 0u8),
    ).context("failed to create buffer")?;

    // create command buffer
    let mut builder = AutoCommandBufferBuilder::primary(
        &ctx.command_buffer_allocator(),
        ctx.queue().queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )?;
    builder.bind_pipeline_compute(compute_pipeline.clone())?
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            compute_pipeline.layout().clone(),
            0,
            set,
        )?
        .dispatch([1024 / 8, 1024 / 8, 1])?
        .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
            image.clone(),
            buf.clone(),
        ))?;
    let command_buffer = builder.build()?;

    vulkano::sync::now(ctx.device())
        .then_execute(ctx.queue(), command_buffer)?
        .then_signal_fence_and_flush()?
        .wait(None)?;

    let buffer_content = buf.read()?;
    std::fs::create_dir_all("output")?;
    let target_path = "output/s5_image.png";
    let image = image::ImageBuffer::<image::Rgba<u8>, _>
        ::from_raw(1024, 1024, &buffer_content[..])
        .context("could not create image")?;
    image.save(target_path)?;
    // XXX: macOS specific
    Command::new("open")
        .arg(target_path)
        .spawn()?;

    info!("[s5_image_creation] succeeded!");
    Ok(())
}
