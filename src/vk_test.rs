use std::sync::Arc;
use anyhow::{Context, Result};
use tracing::info;

use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, CopyImageToBufferInfo, PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents, SubpassEndInfo};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::image::view::ImageView;
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::pipeline::{ComputePipeline, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, Subpass};
use vulkano::shader::ShaderModule;
use vulkano::{swapchain, Validated, VulkanError};
use vulkano::swapchain::SwapchainPresentInfo;
use vulkano::sync::future::FenceSignalFuture;
use vulkano::sync::GpuFuture;
use winit::event::{Event, WindowEvent};
use winit::event_loop::ControlFlow;

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
    // create image and destination buffer (to copy the image into)
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

    // load shader and compute pipeline
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

    // load descriptor set
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
    // Command::new("open")
    //     .arg(target_path)
    //     .spawn()?;

    info!("[s5_image_creation] succeeded!");
    Ok(())
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
struct MyVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}

mod s6_vertex_shader {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 460

            layout(location = 0) in vec2 position;

            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
            }
        ",
    }
}
mod s6_fragment_shader {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
            #version 460

            layout(location = 0) out vec4 f_color;

            void main() {
                f_color = vec4(1.0, 0.0, 0.0, 1.0);
            }
        ",
    }
}
pub fn s6_graphics_pipeline(ctx: vk_util::TestContext) -> Result<()> {
    // create vertex buffer
    let vertex1 = MyVertex { position: [-0.5, -0.5] };
    let vertex2 = MyVertex { position: [ 0.0,  0.5] };
    let vertex3 = MyVertex { position: [ 0.5, -0.25] };
    let vertex_buffer = Buffer::from_iter(
        ctx.memory_allocator(),
        BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        vec![vertex1, vertex2, vertex3],
    )?;

    // create render pass
    let render_pass = vulkano::single_pass_renderpass!(
        ctx.device(),
        attachments: {
            color: {
                format: vulkano::format::Format::R8G8B8A8_UNORM,
                samples: 1,
                load_op: Clear,
                store_op: Store,
            },
        },
        pass: {
            color: [color],
            depth_stencil: {},
        },
    )?;

    // create image, output buffer, viewport, and framebuffer
    let image = Image::new(
        ctx.memory_allocator(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format: vulkano::format::Format::R8G8B8A8_UNORM,
            extent: [1024, 1024, 1],
            usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
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
    )?;
    let view = ImageView::new_default(image.clone()).unwrap();
    let viewport = Viewport {
        offset: [0.0, 0.0],
        extent: [1024.0, 1024.0],
        depth_range: 0.0..=1.0,
    };
    let framebuffer = Framebuffer::new(
        render_pass.clone(),
        FramebufferCreateInfo {
            attachments: vec![view],
            ..Default::default()
        },
    )?;

    // load shaders
    let vs = s6_vertex_shader::load(ctx.device()).context("failed to create shader module")?;
    let fs = s6_fragment_shader::load(ctx.device()).context("failed to create shader module")?;

    // create pipeline
    let pipeline = {
        let vs = vs.entry_point("main").context("vertex shader: entry point missing")?;
        let fs = fs.entry_point("main").context("fragment shader: entry point missing")?;

        let vertex_input_state = MyVertex::per_vertex()
            .definition(&vs.info().input_interface)?;

        let stages = [
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs),
        ];

        let layout = PipelineLayout::new(
            ctx.device(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(ctx.device())?,
        )?;

        let subpass = Subpass::from(render_pass.clone(), 0).context("failed to create subpass")?;

        GraphicsPipeline::new(
            ctx.device(),
            /* cache= */ None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                // default is a list of triangles
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState {
                    viewports: [viewport].into_iter().collect(),
                    ..Default::default()
                }),
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.num_color_attachments(),
                    ColorBlendAttachmentState::default(),
                )),
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )?
    };

    // create command buffer
    let mut builder = AutoCommandBufferBuilder::primary(
        &ctx.command_buffer_allocator(),
        ctx.queue().queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )?;
    builder.begin_render_pass(
            RenderPassBeginInfo {
                clear_values: vec![Some([0.0, 0.0, 1.0, 1.0].into())],
                ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
            },
            SubpassBeginInfo {
                contents: SubpassContents::Inline,
                ..Default::default()
            },
        )?
        .bind_pipeline_graphics(pipeline.clone())?
        .bind_vertex_buffers(0, vertex_buffer.clone())?
        .draw(3, 1, 0, 0)?
        .end_render_pass(SubpassEndInfo::default())?
        .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(image, buf.clone()))?;
    let command_buffer = builder.build()?;
    vulkano::sync::now(ctx.device())
        .then_execute(ctx.queue(), command_buffer)?
        .then_signal_fence_and_flush()?
        .wait(None)?;

    let target_path = "output/s6_image.png";
    let buffer_content = buf.read().unwrap();
    let image = image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(1024, 1024, &buffer_content[..])
        .context("could not create image")?;
    image.save(target_path)?;
    // XXX: macOS specific
    // Command::new("open")
    //     .arg(target_path)
    //     .spawn()?;

    info!("[s6_graphics_pipeline] succeeded!");
    Ok(())
}

fn s7_create_pipeline(ctx: &vk_util::TestContext,
                      vs: Arc<ShaderModule>,
                      fs: Arc<ShaderModule>,
                      viewport: Viewport) -> Result<Arc<GraphicsPipeline>> {
    let vs = vs.entry_point("main").context("vertex shader: entry point missing")?;
    let fs = fs.entry_point("main").context("fragment shader: entry point missing")?;

    let vertex_input_state = MyVertex::per_vertex()
        .definition(&vs.info().input_interface)?;
    let stages = [
        PipelineShaderStageCreateInfo::new(vs),
        PipelineShaderStageCreateInfo::new(fs),
    ];

    let layout = PipelineLayout::new(
        ctx.device(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(ctx.device())?,
    )?;

    let subpass = Subpass::from(ctx.render_pass(), 0).context("failed to create subpass")?;

    Ok(GraphicsPipeline::new(
        ctx.device(),
        None,
        GraphicsPipelineCreateInfo {
            stages: stages.into_iter().collect(),
            vertex_input_state: Some(vertex_input_state),
            input_assembly_state: Some(InputAssemblyState::default()),
            viewport_state: Some(ViewportState {
                viewports: [viewport].into_iter().collect(),
                ..Default::default()
            }),
            rasterization_state: Some(RasterizationState::default()),
            multisample_state: Some(MultisampleState::default()),
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                subpass.num_color_attachments(),
                ColorBlendAttachmentState::default(),
            )),
            subpass: Some(subpass.into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )?)
}

fn s7_create_command_buffers(ctx: &vk_util::TestContext,
                             pipeline: Arc<GraphicsPipeline>,
                             vertex_buffer: &Subbuffer<[MyVertex]>)
        -> Result<Vec<Arc<PrimaryAutoCommandBuffer<Arc<StandardCommandBufferAllocator>>>>> {
    Ok(ctx.framebuffers()
        .iter()
        .map(|framebuffer| {
            let mut builder = AutoCommandBufferBuilder::primary(
                &ctx.command_buffer_allocator(),
                ctx.queue().queue_family_index(),
                CommandBufferUsage::MultipleSubmit,
            )?;

            builder.begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some([0.1, 0.1, 0.1, 1.0].into())],
                    ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                },
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )?
                .bind_pipeline_graphics(pipeline.clone())?
                .bind_vertex_buffers(0, vertex_buffer.clone())?
                .draw(vertex_buffer.len() as u32, 1, 0, 0)?
                .end_render_pass(SubpassEndInfo::default())?;
            builder.build()
        }).try_collect()?)
}

pub fn s7_windowing(window_ctx: vk_util::WindowContext, mut ctx: vk_util::TestContext) -> Result<()> {
    let (event_loop, window) = window_ctx.consume();

    // create vertex buffer
    let vertex1 = MyVertex { position: [-0.5, -0.5] };
    let vertex2 = MyVertex { position: [ 0.0,  0.5] };
    let vertex3 = MyVertex { position: [ 0.5, -0.25] };
    let vertex_buffer = Buffer::from_iter(
        ctx.memory_allocator(),
        BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        vec![vertex1, vertex2, vertex3],
    )?;

    // load shaders
    let vs = s6_vertex_shader::load(ctx.device()).context("failed to create shader module")?;
    let fs = s6_fragment_shader::load(ctx.device()).context("failed to create shader module")?;

    let mut viewport = Viewport {
        offset: [0.0, 0.0],
        extent: window.inner_size().into(),
        depth_range: 0.0..=1.0,
    };

    let pipeline = s7_create_pipeline(&ctx, vs.clone(), fs.clone(), viewport.clone())?;
    let mut command_buffers = s7_create_command_buffers(&ctx, pipeline, &vertex_buffer)?;
    let frames_in_flight = ctx.images().len();
    // XXX: FenceSignalFuture is not Send + Sync, so Arc is cheating. However, there is no
    // `impl GpuFuture for Rc<FenceSignalFuture<...>>`, so we can't use Rc. We don't use threads yet
    // so this is safe anyway.
    let mut fences: Vec<Option<Arc<FenceSignalFuture<_>>>> = vec![None; frames_in_flight];
    let mut previous_fence_i = 0;

    let mut window_resized = false;
    let mut recreate_swapchain = false;
    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            },
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                window_resized = true;
            },
            Event::MainEventsCleared => {
                if window_resized || recreate_swapchain {
                    recreate_swapchain = false;
                    ctx.recreate_swapchain(window.clone()).expect("could not recreate swapchain");
                }
                if window_resized {
                    window_resized = false;
                    viewport.extent = window.inner_size().into();
                    command_buffers = s7_create_pipeline(&ctx, vs.clone(), fs.clone(), viewport.clone())
                        .and_then(|new_pipeline| {
                            s7_create_command_buffers(&ctx, new_pipeline, &vertex_buffer)
                        }).expect("failed to recreate command buffers after resize");
                }
                let (image_i, suboptimal, acquire_future) =
                    match swapchain::acquire_next_image(ctx.swapchain(), None)
                            .map_err(Validated::unwrap) {
                        Ok(r) => r,
                        Err(VulkanError::OutOfDate) => {
                            recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("failed to acquire next image: {e}"),
                    };
                if suboptimal {
                    recreate_swapchain = true;
                }
                if let Some(image_fence) = &fences[image_i as usize] {
                    image_fence.wait(None).unwrap();
                }
                let previous_future = match fences[previous_fence_i as usize].clone() {
                    None => {
                        let mut now = vulkano::sync::now(ctx.device());
                        now.cleanup_finished();
                        now.boxed()
                    }
                    Some(fence) => fence.boxed(),
                };
                let future = previous_future
                    .join(acquire_future)
                    .then_execute(ctx.queue(), command_buffers[image_i as usize].clone())
                    .unwrap()
                    .then_swapchain_present(
                        ctx.queue(),
                        SwapchainPresentInfo::swapchain_image_index(ctx.swapchain(), image_i),
                    )
                    .then_signal_fence_and_flush();
                fences[image_i as usize] = match future.map_err(Validated::unwrap) {
                    Ok(value) => Some(Arc::new(value)),
                    Err(VulkanError::OutOfDate) => {
                        recreate_swapchain = true;
                        None
                    }
                    Err(e) => {
                        println!("failed to flush future: {e}");
                        None
                    }
                };
                previous_fence_i = image_i;
            },
            _ => (),
        }
    });
}
