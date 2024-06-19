use anyhow::{Context, Result};
use tracing::info;

use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::sync::GpuFuture;

use crate::vk_util;
use crate::vk_util::TestContext;

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

    let command_buffer_allocator = StandardCommandBufferAllocator::new(
        ctx.device(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    );
    let mut builder = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator,
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

pub fn s4_compute_operations(ctx: TestContext) -> Result<()> {
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
    let descriptor_set_allocator =
        StandardDescriptorSetAllocator::new(ctx.device(), Default::default());
    let pipeline_layout = compute_pipeline.layout();
    let descriptor_set_layouts = pipeline_layout.set_layouts();

    let descriptor_set_layout_index = 0;
    let descriptor_set_layout = descriptor_set_layouts
        .get(descriptor_set_layout_index)
        .context("no descriptor sets found in shader")?;
    let descriptor_set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        descriptor_set_layout.clone(),
        [WriteDescriptorSet::buffer(0, data_buffer.clone())], // 0 is the binding
        [],
    )?;

    // create command buffer
    let command_buffer_allocator = StandardCommandBufferAllocator::new(
        ctx.device(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    );
    let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator,
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
            descriptor_set,
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
