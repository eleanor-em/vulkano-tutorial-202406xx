use std::sync::Arc;
use anyhow::{Context, Result};
use tracing::info;
use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo};
use vulkano::device::{Device, Queue};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::sync::GpuFuture;

pub fn s3_buffer_creation(device: Arc<Device>, queue: Arc<Queue>) -> Result<()> {
    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    let src_content: Vec<i32> = (0..64).collect();
    let src = Buffer::from_iter(
        memory_allocator.clone(),
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
        memory_allocator.clone(),
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
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    );
    let mut builder = AutoCommandBufferBuilder::primary(
        &command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )?;
    builder.copy_buffer(CopyBufferInfo::buffers(src.clone(), dest.clone()))?;
    let command_buffer = builder.build()?;

    vulkano::sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)?
        .then_signal_fence_and_flush()?
        .wait(None)?;

    let src_content = src.read()?;
    let destination_content = dest.read()?;
    assert_eq!(&*src_content, &*destination_content);
    info!("Everything succeeded!");

    Ok(())
}
