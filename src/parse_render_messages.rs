use bevy::prelude::*;

use super::{
	compute_data_transmission::{ComputeDataTransmission, ComputeMessage},
	ComputeTaskDoneEvent, CopyBufferEvent,
};
use crate::shader_buffer_set::ShaderBufferSet;

pub fn parse_render_messages(
	mut copy_buffer_events: MessageWriter<CopyBufferEvent>, mut group_done_events: MessageWriter<ComputeTaskDoneEvent>,
	mut buffer_set: ResMut<ShaderBufferSet>, transmission: NonSend<ComputeDataTransmission>,
) {
	while let Ok(data) = transmission.receiver.try_recv() {
		match data {
			ComputeMessage::CopyBuffer(event) => {
				copy_buffer_events.write(event);
			}
			ComputeMessage::GroupDone(event) => {
				group_done_events.write(event);
			}
			ComputeMessage::SwapBuffers(handle) => {
				buffer_set.swap_front_buffer(handle);
			}
		}
	}
}
