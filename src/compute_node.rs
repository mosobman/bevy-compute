use std::{
	borrow::Cow,
	time::{Duration, Instant},
};

use bevy::{
	ecs::system::SystemState,
	prelude::*,
	render::{
		render_graph::{Node, NodeRunError, RenderGraphContext},
		render_resource::{
			CachedComputePipelineId, CachedPipelineState, ComputePassDescriptor, ComputePipelineDescriptor, PipelineCache,
		},
		renderer::{RenderContext, RenderDevice, RenderQueue},
	},
};

use super::{
	compute_bind_groups::ComputeBindGroups,
	compute_data_transmission::ComputeMessage,
	compute_sequence::{ComputeAction, ComputeSequence, ComputeStep},
	ComputeTaskDoneEvent, CopyBufferEvent,
};
use crate::shader_buffer_set::{ShaderBufferRenderSet, ShaderBufferSet};

pub struct ComputeNode {
	sequence: ComputeSequence,
	current_task: usize,
	current_pipelines_loaded: bool,
	step_states: Vec<ComputeStepState>,
	iterations: u32,
	group_start_time: Instant,
}

struct ComputeStepState {
	step: ComputeStep,
	id: Option<CachedComputePipelineId>,
	last_run_time: Instant,
	run_this_time: bool,
	copy_buffer_ready: bool,
}

impl ComputeNode {
	pub fn new(sequence: &ComputeSequence) -> Self {
		Self {
			sequence: sequence.clone(),
			current_task: 0,
			current_pipelines_loaded: false,
			step_states: Vec::new(),
			iterations: 0,
			group_start_time: Instant::now(),
		}
	}

	fn run_shader(
		&self, pipeline_id: CachedComputePipelineId, x_workgroup_size: u32, y_workgroup_size: u32, z_workgroup_size: u32,
		world: &World, render_context: &mut RenderContext,
	) {
		let pipeline_cache = world.resource::<PipelineCache>();
		let bind_groups = world.resource::<ComputeBindGroups>();
		let Some(pipeline) = pipeline_cache.get_compute_pipeline(pipeline_id) else {
			panic!("Somehow running the shader without all the shader pipelines being loaded");
		};
		let encoder = render_context.command_encoder();
		{
			let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
			pass.set_pipeline(pipeline);
			for (i, bind_group) in bind_groups.0.iter().enumerate() {
				pass.set_bind_group(i as u32, bind_group, &[]);
			}
			pass.dispatch_workgroups(x_workgroup_size, y_workgroup_size, z_workgroup_size);
		}
	}
}

impl Node for ComputeNode {
	fn update(&mut self, world: &mut World) {
		// All the tasks have been completed, so there's nothing to do.
		if self.current_task >= self.sequence.tasks.len() {
			return;
		}

		let mut system_state: SystemState<(
			ResMut<ShaderBufferSet>,
			ResMut<ShaderBufferRenderSet>,
			Res<RenderDevice>,
			Res<RenderQueue>,
			Res<ComputeSequence>,
			ResMut<PipelineCache>,
			Res<AssetServer>,
		)> = SystemState::new(world);
		let (mut buffers, mut render_buffers, device, render_queue, sequence, mut pipeline_cache, asset_server) =
			system_state.get_mut(world);

		let group = &self.sequence.tasks[self.current_task];

		// If there's a maximum number of iterations, check if it's been reached.
		// If it has, clean up after this task and move on to the next.
		// This is an assignment, as it has to update the extracted group if the
		// group is complete.
		let group = if let Some(max_iterations) = group.iterations {
			if self.iterations >= max_iterations.get() {
				for step in self.step_states.iter() {
					if let ComputeAction::CopyBuffer { src } = step.step.action {
						render_buffers.remove_copy_buffer(src);
					}
				}
				let now = Instant::now();
				self.current_task += 1;
				self.current_pipelines_loaded = false;
				self.step_states.clear();
				self.iterations = 0;
				self
					.sequence
					.sender
					.send(ComputeMessage::GroupDone(ComputeTaskDoneEvent {
						group_finished: self.current_task - 1,
						group_finished_label: group.label.clone(),
						time_in_group: now - self.group_start_time,
						final_group: self.current_task == self.sequence.tasks.len(),
					}))
					.unwrap();
				self.group_start_time = now;
				// All the tasks have been completed, so there's nothing to do.
				if self.current_task >= self.sequence.tasks.len() {
					return;
				}
				&self.sequence.tasks[self.current_task]
			} else {
				group
			}
		} else {
			group
		};

		// If step_states is empty, this must be the first iteration on a new group,
		// so it's time to initialize the step_states, which includes setting up all
		// the pipelines in the PipelineCache.
		if self.step_states.len() == 0 {
			for step in group.steps.iter() {
				if let ComputeAction::CopyBuffer { src } = step.action {
					render_buffers.create_copy_buffer(src, &buffers, &device);
				}
				let id = if let ComputeAction::RunShader { shader, entry_point, .. } = &step.action {
					let bind_group_layouts = buffers.bind_group_layouts(&device);
					let shader = asset_server.load(shader);
					Some(pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
						label: None,
						layout: bind_group_layouts.clone(),
						push_constant_ranges: Vec::new(),
						shader: shader,
						shader_defs: vec![],
						entry_point: Some(Cow::Owned(entry_point.clone())),
						zero_initialize_workgroup_memory: true,
					}))
				} else {
					None
				};
				self.step_states.push(ComputeStepState {
					step: step.clone(),
					id,
					last_run_time: if let Some(max_frequency) = step.max_frequency {
						Instant::now() - Duration::from_secs_f32(2.0 / max_frequency.get() as f32)
					} else {
						Instant::now()
					},
					run_this_time: true,
					copy_buffer_ready: true,
				});
			}
			pipeline_cache.process_queue();
		}

		// If the pipelines have not been marked as loaded, check them.
		// If they're loaded, mark them as such. Otherwise we can't continue yet.
		if !self.current_pipelines_loaded {
			let step_states = self.step_states.iter().flat_map(|step| {
				if let Some(id) = step.id {
					Some(pipeline_cache.get_compute_pipeline_state(id))
				} else {
					None
				}
			});
			let state = step_states.fold(Some(Ok(())), |acc, x| match (acc, x) {
				(None, _) => None,
				(Some(Err(e)), _) => Some(Err(e)),
				(Some(Ok(_)), CachedPipelineState::Ok(_)) => Some(Ok(())),
				(Some(Ok(_)), CachedPipelineState::Err(e)) => Some(Err(e)),
				(Some(Ok(_)), _) => None,
			});
			self.current_pipelines_loaded = match state {
				Some(Ok(_)) => true,
				Some(Err(e)) => panic!("{}", e),
				None => false,
			}
		}

		// If the pipelines are actually loaded now, then:
		// - update the iteration buffer, if there is one
		// - for every step:
		//   - if it has a frequency limit, check if it should run this frame
		//   - if it's a buffer copy, alternate whether it copies into or out of the
		//     copy buffer
		if self.current_pipelines_loaded {
			if let Some(buffer) = sequence.iteration_buffer {
				buffers.set_buffer(buffer, self.iterations, &render_queue);
			}
			self.iterations += 1;

			for step in self.step_states.iter_mut() {
				step.run_this_time = if let Some(max_frequency) = step.step.max_frequency {
					let now = Instant::now();
					if now - step.last_run_time > Duration::from_secs_f32(1.0 / max_frequency.get() as f32) {
						step.last_run_time = now;
						true
					} else {
						false
					}
				} else {
					true
				};

				if step.run_this_time {
					step.copy_buffer_ready = !step.copy_buffer_ready;
				}
			}
		}
	}

	fn run(
		&self, _graph: &mut RenderGraphContext, context: &mut RenderContext, world: &World,
	) -> Result<(), NodeRunError> {
		// All the tasks have been completed, so there's nothing to do.
		if self.current_task >= self.sequence.tasks.len() {
			return Ok(());
		}

		// If the current pipelines aren't loaded yet, then we can't do anything
		// this frame.
		if !self.current_pipelines_loaded {
			return Ok(());
		}

		let device = world.resource::<RenderDevice>();
		let buffers = world.resource::<ShaderBufferSet>();
		let render_buffers = world.resource::<ShaderBufferRenderSet>();

		// Iterate over all the steps and run them.
		for step in self.step_states.iter() {
			if !step.run_this_time {
				continue;
			}

			match step.step.action {
				ComputeAction::CopyBuffer { src } => {
					if step.copy_buffer_ready {
						let data = render_buffers.copy_from_copy_buffer_to_vec(src, device);
						self.sequence.sender.send(ComputeMessage::CopyBuffer(CopyBufferEvent { buffer: src, data })).unwrap();
					} else {
						render_buffers.copy_to_copy_buffer(src, buffers, context);
					}
				}
				ComputeAction::RunShader { x_workgroup_count, y_workgroup_count, z_workgroup_count, .. } => {
					if let Some(id) = step.id {
						self.run_shader(id, x_workgroup_count, y_workgroup_count, z_workgroup_count, world, context);
					} else {
						panic!("Somehow got to trying to run a RunShader action step with no pipeline ID");
					}
				}
				ComputeAction::SwapBuffers { buffer } => {
					self.sequence.sender.send(ComputeMessage::SwapBuffers(buffer)).unwrap();
				}
			}
		}

		Ok(())
	}
}
