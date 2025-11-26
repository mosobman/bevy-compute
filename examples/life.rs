extern crate bevy_compute;

use std::num::NonZeroU32;

use bevy::{
	prelude::*,
	render::render_resource::{StorageTextureAccess, TextureFormat},
};
use bevy_compute::{
	BevyComputePlugin, Binding, ComputeAction, ComputeStep, ComputeTask, DoubleBufferedSprite, ShaderBufferSet,
	StartComputeEvent,
};

/// This example uses a shader source file from the assets subdirectory
const SHADER_ASSET_PATH: &str = "shaders/game_of_life.wgsl";

const DISPLAY_FACTOR: u32 = 4;
const SIZE: (u32, u32) = (1280 / DISPLAY_FACTOR, 720 / DISPLAY_FACTOR);
const WORKGROUP_SIZE: u32 = 8;

fn main() {
	App::new()
		.insert_resource(ClearColor(Color::BLACK))
		.add_plugins((
			DefaultPlugins
				.set(WindowPlugin {
					primary_window: Some(Window {
						resolution: ((SIZE.0 * DISPLAY_FACTOR) as u32, (SIZE.1 * DISPLAY_FACTOR) as u32).into(),
						// uncomment for unthrottled FPS
						present_mode: bevy::window::PresentMode::AutoNoVsync,
						..default()
					}),
					..default()
				})
				.set(ImagePlugin::default_nearest()),
			BevyComputePlugin,
		))
		.add_systems(Startup, setup)
		.run();
}

fn setup(
	mut commands: Commands, mut buffer_set: ResMut<ShaderBufferSet>, mut images: ResMut<Assets<Image>>,
	mut start_compute_events: MessageWriter<StartComputeEvent>,
) {
	let image = buffer_set.add_texture_fill(
		&mut images,
		SIZE.0,
		SIZE.1,
		TextureFormat::R32Float,
		&0.0f32.to_ne_bytes(),
		StorageTextureAccess::ReadOnly,
		Binding::Double(0, (0, 1)),
	);

	commands.spawn((
		Sprite {
			image: buffer_set.image_handle(image).unwrap(),
			custom_size: Some(Vec2::new(SIZE.0 as f32, SIZE.1 as f32)),
			..default()
		},
		Transform::from_scale(Vec3::splat(DISPLAY_FACTOR as f32)),
		DoubleBufferedSprite(image),
	));
	commands.spawn(Camera2d);

	start_compute_events.write(StartComputeEvent {
		tasks: vec![
			ComputeTask {
				label: Some("Init".to_owned()),
				iterations: NonZeroU32::new(1),
				steps: vec![
					ComputeStep {
						max_frequency: None,
						action: ComputeAction::RunShader {
							shader: SHADER_ASSET_PATH.to_owned(),
							entry_point: "init".to_owned(),
							x_workgroup_count: SIZE.0 / WORKGROUP_SIZE,
							y_workgroup_count: SIZE.1 / WORKGROUP_SIZE,
							z_workgroup_count: 1,
						},
					},
					ComputeStep { max_frequency: None, action: ComputeAction::SwapBuffers { buffer: image } },
				],
			},
			ComputeTask {
				label: Some("Update".to_owned()),
				iterations: None,
				steps: vec![
					ComputeStep {
						max_frequency: NonZeroU32::new(10),
						action: ComputeAction::RunShader {
							shader: SHADER_ASSET_PATH.to_owned(),
							entry_point: "update".to_owned(),
							x_workgroup_count: SIZE.0 / WORKGROUP_SIZE,
							y_workgroup_count: SIZE.1 / WORKGROUP_SIZE,
							z_workgroup_count: 1,
						},
					},
					ComputeStep { max_frequency: NonZeroU32::new(10), action: ComputeAction::SwapBuffers { buffer: image } },
				],
			},
		],
		iteration_buffer: None,
	});
}
