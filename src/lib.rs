#![warn(missing_docs)]

//! This crate is a plugin for the Bevy game engine to simplify the use of compute shaders.

//! It provides a pretty simple API. First, add the [BevyComputePlugin] to your Bevy app. To initiate the compute shaders, first set up all the needed buffers in the [ShaderBufferSet]. Then, send a [StartComputeEvent] with a [Vec] of [ComputeTask]s that will define the sequence of shaders to run. If relevant, be prepared to recieve [CopyBufferEvent]s, which will have buffer data returned from the computer shaders back to the CPU, and [ComputeTaskDoneEvent]s, which will tell you that a given compute task has completed.
//!
//! And that's really it. But let's cover these steps in a big more detail.
//!
//! # Add the Plugin
//!
//! This is done in the standard way. Just add this call to your Bevy app initialization:
//!
//! ```Rust
//! app.add_plugins((BevyComputePlugin));
//! ```
//!
//! # Making Buffers
//!
//! The [ShaderBufferSet] provides a simple API for managing GPU buffers. This is added as a resource by the [BevyComputePlugin], so you can request `Res<ShaderBufferSet>` in any system to manage your buffers.
//!
//! It provides the following functions for creating buffers:
//!
//! - [add_storage_uninit](ShaderBufferSet::add_storage_uninit) - Add an uninitialized storage buffer.
//! - [add_storage_zeroed](ShaderBufferSet::add_storage_zeroed) - Add a storage buffer filled with 0 bytes.
//! - [add_storage_init](ShaderBufferSet::add_storage_init) - Add a storage buffer with initial data provided.
//! - [add_uniform_init](ShaderBufferSet::add_uniform_init) - Add a uniform buffer with initial data provided.
//! - [add_texture_fill](ShaderBufferSet::add_texture_fill) - Add a texture buffer filled with a solid color.
//!
//! All of these return a [ShaderBufferHandle], which you can store and treat like an opaque reference to access the buffer in the future. Except for [add_texture_fill](ShaderBufferSet::add_texture_fill), which returns a tuple of two such handles.
//!
//! Every one of these functions takes a [Binding], which determines how it's bound to the shaders. WGSL shaders require that each buffer have a group and a binding, which are numeric identifiers used to match the buffers specified on the CPU to those that exist in the shaders. The [Binding] is an enum, which can come in three types:
//!
//! - [SingleBound(u32, u32)](Binding::SingleBound) - This is the standard binding. The first value is the group and the second the binding.
//! - [Double(u32, (u32, u32))](Binding::Double) - This is a double buffer. There's actually two buffers. One is considered the front buffer, and one the back buffer, and they can be swapped. The first value the group both buffers will be in, and the tuple is the bindings of the front and back buffers, respectively. This is discussed in more detail in the "Double Buffering" section below.
//! - [SingleUnbound](Binding::SingleUnbound) - This buffer is not bound, and is thus inaccessible in shaders. While there are unbound buffers used in the background for data transmission purposes, it's rarely if ever useful to specify this at this level.
//!
//! The [ShaderBufferSet] also provides a few more functions for managing buffers:
//!
//! - [delete_buffer](ShaderBufferSet::delete_buffer) - Predictably, this deletes a buffer.
//! - [image_handle](ShaderBufferSet::image_handle) - Extracts the Bevy `Handle<Image>` associated with a texture buffer, so it can be displayed.
//! - [set_buffer](ShaderBufferSet::set_buffer) - Sets the contents of a buffer.
//!
//! ## Setting Buffer Contents
//!
//! Buffer contents are internally just arrays of bytes, but they can be converted from more complicated data structures. This API uses the [ShaderType](bevy::render::render_resource::ShaderType) trait to do that, which comes from the Encase crate that is included with Bevy. You can put `#[derive(ShaderType)]` in front of any data type, as long as all fields in that data type also implement [ShaderType](bevy::render::render_resource::ShaderType). All basic numeric types already do, along with any array, tuple or [Vec] of types that implement [ShaderType](bevy::render::render_resource::ShaderType). Which makes it very easy to pass whatever structured data you want into your shaders. Just be careful, because the shader has to specify the structure of the data independently, and if there's a mismatch it will only throw an error if they're a different size.
//!
//! # Starting the Compute Shader
//!
//! To start running the compute shaders, you need to throw a [StartComputeEvent]. This contains a [Vec] of [ComputeTask]s, which details all the compute tasks to complete, and a optional [ShaderBufferHandle], for the optional iteration buffer.
//!
//! ## ComputeTask
//!
//! A compute task represents one stage of your compute shader program. The compute task is optionally provided a number of iterations, and it will run for that many ticks before moving on to the next task. If that's not provided, it'll run forever. A compute task is also given a list of [ComputeStep]s, each of which is a specific shader to run, or other compute-related action to take, in order, each iteration. It can also be given an optional label, which is used to identify the task in the [ComputeTaskDoneEvent] that's thrown when the task completes.
//!
//! Each [ComputeStep] contains just two fields.
//!
//! The first is an optional maximum frequency. If provided, this means this step won't necessarily run every iteration, but only if it's been long enough since the last time it ran. The frequency is in Hz, or iterations per second. So if a max frequency of 30 is provided, that means if it's been less than 1000/30=16.67 ms since the last time it ran, then it won't run this iteration. This is often useful if you have a long running computation, and want to display the results in real time. You can potentially speed things up by only updating the display at a set framerate, even if the computation is running at a much faster rate.
//!
//! The second field of the [ComputeStep] is a [ComputeAction], which is an enum which describes what to actually do. It has the following options:
//!
//! - [RunShader](ComputeAction::RunShader) - The meat of the compute shaders. This runs an actual shader. You must provide the Bevy asset path to the shader file, the name of the entry point function in that shader file, and the workgroup count in the x, y and z dimensions.
//! - [CopyBuffer](ComputeAction::CopyBuffer) - Copy the data from a buffer to the CPU. Will be returned as a `Vec<u8>` via a [CopyBufferEvent].
//! - [SwapBuffers](ComputeAction::SwapBuffers) - Swap double buffers. See the "Double Buffering" section below.
//!
//! # Double Buffering
//!
//! It can sometimes be useful to have double buffers, where one buffer is the front buffer, and one the back buffer, and you read from the front buffer while writing to the back buffer, and then swap them for the next frame. This allows you to avoid reading from and writing to the same buffer, which can result in weird behavior when some of the data you're reading was written last frame, and some was written earlier this frame.
//!
//! So this plugin supports this directly. When you declare a buffer with the [Double](Binding::Double) binding type, it will actually create two buffers internally. One of them is considered the front buffer, which will be bound to the first binding provided, and the back buffer will be bound to the second binding. When the [SwapBuffers](ComputeAction::SwapBuffers) compute action happens, it will swap which buffer is considered the front buffer.
//!
//! When you do a [CopyBuffer](ComputeAction::CopyBuffer) compute action on a double buffer, it will always copy out of the front buffer. Also, if you call the [image_handle](ShaderBufferSet::image_handle) function on a double buffer texture, it will return the handle for the front buffer.
//!
//! There's also a special accommodation for using a double buffered texture on a Bevy sprite. The [DoubleBufferedSprite] component requires a [Sprite] component, and it will automatically update that image handle on that sprite every frame to contain the new front buffer.

mod compute_bind_groups;
mod compute_data_transmission;
mod compute_main_setup;
mod compute_node;
mod compute_render_setup;
mod compute_sequence;
mod extract_resources;
mod parse_render_messages;
mod queue_bind_group;
mod shader_buffer_set;
mod swap_sprite_buffers;

use std::{sync::mpsc::sync_channel, time::Duration};

use bevy::{
	prelude::*,
	render::{Render, RenderApp, RenderSystems},
};
use compute_data_transmission::ComputeDataTransmission;
use compute_main_setup::compute_main_setup;
use compute_render_setup::compute_render_setup;
use compute_sequence::ComputeSequence;
pub use compute_sequence::*;
use extract_resources::extract_resources;
use parse_render_messages::parse_render_messages;
use queue_bind_group::queue_bind_group;
use shader_buffer_set::ShaderBufferSetPlugin;
pub use shader_buffer_set::*;
use swap_sprite_buffers::swap_sprite_buffers;

/// This plugin adds all the systems, resources and events necessary for bevy_compute to function. Please add it to your
/// bevy app with:
///
/// ```Rust
/// app.add_plugins((BevyComputePlugin));
/// ```
pub struct BevyComputePlugin;

impl Plugin for BevyComputePlugin {
	fn build(&self, app: &mut App) {
		let (sender, receiver) = sync_channel(16);

		app
			.add_plugins(ShaderBufferSetPlugin)
			.insert_non_send_resource(ComputeDataTransmission { sender, receiver })
			.add_systems(Update, compute_main_setup)
			.add_systems(First, parse_render_messages.run_if(resource_exists::<ComputeSequence>))
			.add_systems(Update, swap_sprite_buffers.run_if(resource_exists::<ComputeSequence>))
			.add_message::<StartComputeEvent>()
			.add_message::<CopyBufferEvent>()
			.add_message::<ComputeTaskDoneEvent>();

		let render_app = app.sub_app_mut(RenderApp);
		render_app
			.add_systems(ExtractSchedule, extract_resources)
			.add_systems(Render, queue_bind_group.in_set(RenderSystems::Queue).run_if(resource_exists::<ComputeSequence>))
			.add_systems(Render, compute_render_setup.run_if(resource_added::<ComputeSequence>));
	}
}

/// This event is how you start the compute shaders. Specify the details of how they're going to run with the [tasks](StartComputeEvent::tasks), and optionally provide a buffer to store the current iteration count with [iteration_buffer](StartComputeEvent::iteration_buffer).
#[derive(Event, Message)]
pub struct StartComputeEvent {
	/// Ths list of compute tasks to complete. It will run each task in sequence, and throw a [ComputeTaskDoneEvent] when they're done.
	pub tasks: Vec<ComputeTask>,

	/// An optional iteration buffer. This buffer should be a 4-byte uniform buffer, that stores a single u32. If provided, then every tick, it will be set to the current iteration count within the current compute task. It will reset to zero every time a new compute task starts.
	pub iteration_buffer: Option<ShaderBufferHandle>,
}

/// This event is thrown every time a [CopyBuffer][ComputeAction::CopyBuffer] compute action is executed. It contains the handle of the buffer that was copied, and a `Vec<u8>` with all the data. This is how you get data back out of the compute shader to the CPU.
#[derive(Event, Message)]
pub struct CopyBufferEvent {
	/// This is the handle of the buffer that was copied.
	pub buffer: ShaderBufferHandle,

	/// This is the data the buffer contained, as a raw sequence of bytes.
	pub data: Vec<u8>,
}

/// This event is thrown every time a compute task is completed.
#[derive(Event, Message)]
pub struct ComputeTaskDoneEvent {
	/// The number of the completed task, as in, the index into the `Vec<ComputeTask>` that was provided in the [StartComputeEvent].
	pub group_finished: usize,

	/// The label of the completed task, if one was provided.
	pub group_finished_label: Option<String>,

	/// The time spent on the task. A timestamp is taken on start and completion of the task, and the difference provided here.
	pub time_in_group: Duration,

	/// Whether this is the final task. If all you care about is whether the entire compute sequence is done, then check this.
	pub final_group: bool,
}

/// This component should be placed on any sprite entity that is intended to display a double buffered texture. It requires a [Sprite]. There is an internal system that will update the image handle on that [Sprite] to be the current front buffer.
#[derive(Component)]
#[require(Sprite)]
pub struct DoubleBufferedSprite(pub ShaderBufferHandle);
