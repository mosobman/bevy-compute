use bevy::prelude::*;

use super::{compute_data_transmission::ComputeDataTransmission, compute_sequence::ComputeSequence, StartComputeEvent};

pub fn compute_main_setup(
	mut commands: Commands, mut start_events: MessageReader<StartComputeEvent>,
	transmission: NonSend<ComputeDataTransmission>,
) {
	if let Some(event) = start_events.read().next() {
		commands.insert_resource(ComputeSequence {
			sender: transmission.sender.clone(),
			tasks: event.tasks.clone(),
			iteration_buffer: event.iteration_buffer,
		});
		if let Some(_) = start_events.read().next() {
			panic!("Attempted to start multiple compute sequences at once");
		}
	}
}
