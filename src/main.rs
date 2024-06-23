#![feature(iterator_try_collect)]

use std::sync::Arc;
use anyhow::{Context, Result};
use num_traits::{Float, One};

use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents, SubpassEndInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter};
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::{GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::render_pass::Subpass;
use vulkano::shader::ShaderModule;
use winit::window::Window;
use linalg::{Mat3x3, Vec2};

mod vk_util;
mod vk_test;
mod linalg;

use vk_util::VulkanoContext;

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
    let ctx = vk_util::VulkanoContext::new(&window_ctx)?;
    vk_test::s3_buffer_creation(ctx.clone())?;
    vk_test::s4_compute_operations(ctx.clone())?;
    vk_test::s5_image_creation(ctx.clone())?;
    vk_test::s6_graphics_pipeline(ctx.clone())?;
    s7_windowing(window_ctx, ctx.clone())
}

// This is the "main" test (i.e. used for active dev).
#[derive(BufferContents, Vertex, Clone, Copy)]
#[repr(C)]
struct S7Vertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}
mod s7_vertex_shader {
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
mod s7_fragment_shader {
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
fn s7_windowing(window_ctx: vk_util::WindowContext, ctx: vk_util::VulkanoContext) -> Result<()> {
    let vertex1 = S7Vertex { position: [-0.5, -0.5] };
    let vertex2 = S7Vertex { position: [ 0.0,  0.5] };
    let vertex3 = S7Vertex { position: [ 0.5, -0.25] };
    let vertex_buffer = Buffer::from_iter(
        ctx.memory_allocator(),
        BufferCreateInfo { usage: BufferUsage::VERTEX_BUFFER, ..Default::default() },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE, ..Default::default()
        },
        vec![vertex1, vertex2, vertex3],
    )?;

    let viewport = window_ctx.create_default_viewport();
    let vs = s7_vertex_shader::load(ctx.device()).context("failed to create shader module")?;
    let fs = s7_fragment_shader::load(ctx.device()).context("failed to create shader module")?;
    let command_buffers = s7_create_pipeline(&ctx, vs.clone(), fs.clone(), viewport.clone())
        .and_then(|pipeline| s7_create_command_buffers(&ctx, pipeline, &vertex_buffer))?;
    let handler = S7Handler::new(vs, fs, vertex_buffer, viewport, command_buffers);

    // TODO: proper test cases...
    let a = Vec2 { x: 1.0, y: 1.0 };
    assert!((a * 2.0).almost_eq(Vec2 { x: 2.0, y: 2.0 }));
    assert!((2.0 * a).almost_eq(Vec2 { x: 2.0, y: 2.0 }));
    assert!(f64::abs((a * 2.0 - a).x - 1.0) < f64::epsilon());
    assert!(f64::abs((a * 2.0 - a).y - 1.0) < f64::epsilon());
    assert!((Mat3x3::rotation(-1.0)
        * Mat3x3::rotation(0.5)
        * Mat3x3::rotation(0.5)).almost_eq(Mat3x3::one()));

    let (event_loop, window) = window_ctx.consume();
    vk_util::WindowEventHandler::new(window, ctx, handler).run(event_loop);


    Ok(())
}

struct S7Handler {
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    vertex_buffer: Subbuffer<[S7Vertex]>,
    viewport: Viewport,
    command_buffers: Vec<Arc<PrimaryAutoCommandBuffer>>,
    top_left: [f32; 2],
    t: usize,
}

impl S7Handler {
    fn new(vs: Arc<ShaderModule>,
           fs: Arc<ShaderModule>,
           vertex_buffer: Subbuffer<[S7Vertex]>,
           viewport: Viewport,
           command_buffers: Vec<Arc<PrimaryAutoCommandBuffer>>) -> Self {
        Self {
            vs, fs, vertex_buffer, viewport, command_buffers,
            top_left: [-0.5, -0.5],
            t: 0,
        }
    }

    fn recreate_command_buffers(&mut self, ctx: &VulkanoContext) -> Result<()> {
        self.command_buffers = s7_create_pipeline(ctx, self.vs.clone(), self.fs.clone(), self.viewport.clone())
            .and_then(|pipeline| s7_create_command_buffers(ctx, pipeline, &self.vertex_buffer))?;
        Ok(())
    }
}
impl vk_util::RenderEventHandler<PrimaryAutoCommandBuffer> for S7Handler {
    fn on_resize(&mut self, ctx: &VulkanoContext, window: Arc<Window>) -> Result<()> {
        self.viewport.extent = window.inner_size().into();
        self.recreate_command_buffers(ctx)?;
        Ok(())
    }

    fn on_update(&mut self, ctx: &vk_util::VulkanoContext) -> Result<()> {
        self.t = (self.t + 1) % 62831853;
        const PERIOD: f64 = 70.0;
        let radians = self.t as f64 / PERIOD;
        // TODO: impl From<[f32; 2]> for Vec2
        let mut top_left_vec: Vec2 = self.top_left.into();
        let mut bottom_right_vec = Vec2 { x: top_left_vec.x + 0.5, y: top_left_vec.y + 0.5 };
        let mut bottom_left_vec = Vec2 { x: top_left_vec.x, y: top_left_vec.y + 0.5 };
        let transform = Mat3x3::translation_vec2(-bottom_right_vec) *
            Mat3x3::rotation(-radians) *
            Mat3x3::translation_vec2(bottom_right_vec);
        top_left_vec *= transform;
        bottom_right_vec *= transform;
        bottom_left_vec *= transform;
        let next_vertices = [
            S7Vertex { position: top_left_vec.into() },
            S7Vertex { position: bottom_right_vec.into() },
            S7Vertex { position: bottom_left_vec.into() },
        ];
        self.vertex_buffer = Buffer::from_iter(
            ctx.memory_allocator(),
            BufferCreateInfo { usage: BufferUsage::VERTEX_BUFFER, ..Default::default() },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE, ..Default::default()
            },
            next_vertices,
        )?;
        self.recreate_command_buffers(ctx)?;
        Ok(())
    }

    fn on_render(&mut self, _ctx: &vk_util::VulkanoContext) -> Result<Vec<Arc<PrimaryAutoCommandBuffer>>> {
        Ok(self.command_buffers.clone())
    }
}

fn s7_create_pipeline(ctx: &vk_util::VulkanoContext,
                      vs: Arc<ShaderModule>,
                      fs: Arc<ShaderModule>,
                      viewport: Viewport) -> Result<Arc<GraphicsPipeline>> {
    let vs = vs.entry_point("main").context("vertex shader: entry point missing")?;
    let fs = fs.entry_point("main").context("fragment shader: entry point missing")?;
    let vertex_input_state = S7Vertex::per_vertex()
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

fn s7_create_command_buffers(ctx: &vk_util::VulkanoContext,
                             pipeline: Arc<GraphicsPipeline>,
                             vertex_buffer: &Subbuffer<[S7Vertex]>)
                             -> Result<Vec<Arc<PrimaryAutoCommandBuffer>>> {
    Ok(ctx.framebuffers()
        .iter()
        .map(|framebuffer| {
            let mut builder = AutoCommandBufferBuilder::primary(
                ctx.command_buffer_allocator(),
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
                })?
                .bind_pipeline_graphics(pipeline.clone())?
                .bind_vertex_buffers(0, vertex_buffer.clone())?
                .draw(vertex_buffer.len() as u32, 1, 0, 0)?
                .end_render_pass(SubpassEndInfo::default())?;
            builder.build()
        }).try_collect()?)
}
