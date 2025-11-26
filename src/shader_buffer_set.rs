use std::{
	fmt::{Display, Formatter},
	sync::mpsc::channel,
};

use std::collections::HashMap;
use bevy::asset::RenderAssetUsages;
use bevy::{
	prelude::*,
	render::{
		extract_resource::ExtractResource,
		render_asset::{RenderAssets}, // RenderAssetUsages
		render_resource::{
			encase::private::{WriteInto, Writer},
			BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutEntry, BindingResource, BindingType, Buffer,
			BufferBindingType, BufferDescriptor, BufferInitDescriptor, BufferUsages, Extent3d, MapMode, // Maintain
			ShaderStages, ShaderType, StorageBuffer, StorageTextureAccess, TextureDimension, TextureFormat, TextureUsages,
			TextureViewDimension,
		},
		renderer::{RenderContext, RenderDevice, RenderQueue},
		texture::GpuImage,
		Extract, RenderApp,
	},
	// utils::HashMap,
};
use wgpu::PollType;

#[derive(Clone)]
enum ShaderBufferStorage {
	Storage { buffer: Buffer, readonly: bool },
	Uniform(Buffer),
	StorageTexture { format: TextureFormat, access: StorageTextureAccess, image: Handle<Image> },
}

impl ShaderBufferStorage {
	fn bind_group_entry<'a>(&'a self, binding: u32, gpu_images: &'a RenderAssets<GpuImage>) -> BindGroupEntry<'a> {
		match self {
			ShaderBufferStorage::Storage { buffer, readonly: _ } => {
				BindGroupEntry { binding, resource: buffer.as_entire_binding() }
			}
			ShaderBufferStorage::Uniform(buffer) => BindGroupEntry { binding, resource: buffer.as_entire_binding() },
			ShaderBufferStorage::StorageTexture { image, .. } => {
				let image = gpu_images.get(image).unwrap();
				BindGroupEntry { binding, resource: BindingResource::TextureView(&image.texture_view) }
			}
		}
	}

	fn bind_group_layout_entry_binding_type(&self, access_override: Option<StorageTextureAccess>) -> BindingType {
		match &self {
			ShaderBufferStorage::Storage { buffer: _, readonly } => BindingType::Buffer {
				ty: BufferBindingType::Storage { read_only: *readonly },
				has_dynamic_offset: false,
				min_binding_size: None,
			},
			ShaderBufferStorage::Uniform(_) => {
				BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }
			}
			ShaderBufferStorage::StorageTexture { format, access, .. } => BindingType::StorageTexture {
				access: access_override.unwrap_or(*access),
				format: *format,
				view_dimension: TextureViewDimension::D2,
			},
		}
	}

	fn set<T: ShaderType + WriteInto>(&self, data: T, render_queue: &RenderQueue) {
		fn set_buffer<T: ShaderType + WriteInto>(data: T, buffer: &Buffer, render_queue: &RenderQueue) {
			let mut bytes = Vec::new();
			let mut writer = Writer::new(&data, &mut bytes, 0).unwrap();
			data.write_into(&mut writer);
			render_queue.write_buffer(buffer, 0, bytes.as_ref());
		}

		if let ShaderBufferStorage::Storage { buffer, readonly: _ } = &self {
			set_buffer(data, buffer, render_queue);
		} else if let ShaderBufferStorage::Uniform(buffer) = &self {
			set_buffer(data, buffer, render_queue);
		} else {
			panic!("Tried to set data on a buffer that isn't a storage or uniform buffer");
		}
	}

	pub fn delete(&mut self, images: &mut Assets<Image>) {
		match &self {
			ShaderBufferStorage::Storage { buffer, .. } => buffer.destroy(),
			ShaderBufferStorage::Uniform(buffer) => buffer.destroy(),
			ShaderBufferStorage::StorageTexture { image, .. } => {
				images.remove(image);
			}
		}
	}

	pub fn image_handle(&self) -> Option<Handle<Image>> {
		match self {
			ShaderBufferStorage::StorageTexture { image, .. } => Some(image.clone()),
			_ => None,
		}
	}

	pub fn gpu_buffer(&self) -> Option<Buffer> {
		match self {
			ShaderBufferStorage::Storage { buffer, .. } => Some(buffer.clone()),
			ShaderBufferStorage::Uniform(buffer) => Some(buffer.clone()),
			_ => None,
		}
	}
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum FrontBuffer {
	First,
	Second,
}

#[derive(Clone)]
enum ShaderBufferInfo {
	SingleBound { binding: (u32, u32), storage: ShaderBufferStorage },
	SingleUnbound { storage: ShaderBufferStorage },
	Double { binding: (u32, (u32, u32)), front: FrontBuffer, storage: (ShaderBufferStorage, ShaderBufferStorage) },
}

/// Specifies how a given buffer will be bound to the shaders.
#[derive(Clone, Copy)]
pub enum Binding {
	/// This will be a single buffer accessible in shaders. The first number is the group, and the second the binding.
	SingleBound(u32, u32),

	/// This buffer will not be accessible in shaders. While there are absolutely uses for unbound buffers, it's rare that it'll be useful to specify an unbound buffer at this layer.
	SingleUnbound,

	/// This will actually be two buffers, of identical size, type and format. One will the front buffer, that is read from, and the other the back buffer, that is written to. Which buffers is which can be swapped with the [SwapBuffers](crate::ComputeAction::SwapBuffers) compute action. The first number is the group they will be both be bound in, and the second tuple is the bindings of the front and back buffers, respectively. If this binding is used for a texture buffer, then the front buffer will always be `ReadOnly` and the back buffer `WriteOnly`, overriding the provided access specifier.
	Double(u32, (u32, u32)),
}

impl ShaderBufferInfo {
	fn new<F: FnMut() -> ShaderBufferStorage>(binding: Binding, mut make_storage: F) -> Self {
		match binding {
			Binding::SingleBound(group, binding) => Self::SingleBound { binding: (group, binding), storage: make_storage() },
			Binding::SingleUnbound => Self::SingleUnbound { storage: make_storage() },
			Binding::Double(group, bindings) => Self::Double {
				binding: (group, bindings),
				front: FrontBuffer::First,
				storage: (make_storage(), make_storage()),
			},
		}
	}

	fn new_storage_uninit(
		render_device: &RenderDevice, size: u32, usage: BufferUsages, binding: Binding, readonly: bool,
	) -> Self {
		Self::new(binding, || ShaderBufferStorage::Storage {
			buffer: render_device.create_buffer(&BufferDescriptor {
				label: None,
				size: size as u64,
				usage,
				mapped_at_creation: false,
			}),
			readonly,
		})
	}

	fn new_storage_zeroed(
		render_device: &RenderDevice, size: u32, usage: BufferUsages, binding: Binding, readonly: bool,
	) -> Self {
		Self::new(binding, || ShaderBufferStorage::Storage {
			buffer: render_device.create_buffer_with_data(&BufferInitDescriptor {
				label: None,
				contents: &vec![0u8; size as usize],
				usage,
			}),
			readonly,
		})
	}

	fn new_storage_init<T: ShaderType + WriteInto + Default + Clone>(
		render_device: &RenderDevice, render_queue: &RenderQueue, data: T, usage: BufferUsages, binding: Binding,
		readonly: bool,
	) -> Self {
		Self::new(binding, || ShaderBufferStorage::Storage {
			buffer: {
				let mut buffer = StorageBuffer::default();
				buffer.set(data.clone());
				buffer.add_usages(usage);
				buffer.write_buffer(&render_device, &render_queue);
				buffer.buffer().unwrap().clone()
			},
			readonly,
		})
	}

	fn new_uniform_init<T: ShaderType + WriteInto + Default + Clone>(
		render_device: &RenderDevice, render_queue: &RenderQueue, data: T, usage: BufferUsages, binding: Binding,
	) -> Self {
		Self::new(binding, || {
			ShaderBufferStorage::Uniform({
				let mut buffer = StorageBuffer::default();
				buffer.set(data.clone());
				buffer.add_usages(usage);
				buffer.write_buffer(&render_device, &render_queue);
				buffer.buffer().unwrap().clone()
			})
		})
	}

	fn new_write_texture(
		images: &mut Assets<Image>, width: u32, height: u32, format: TextureFormat, fill: &[u8],
		access: StorageTextureAccess, binding: Binding,
	) -> Self {
		Self::new(binding, || {
			let mut image = Image::new_fill(
				Extent3d { width: width, height: height, depth_or_array_layers: 1 },
				TextureDimension::D2,
				fill,
				format,
				RenderAssetUsages::RENDER_WORLD,
			);
			image.texture_descriptor.usage =
				TextureUsages::COPY_DST | TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
			let image = images.add(image);
			ShaderBufferStorage::StorageTexture { format, access, image }
		})
	}

	fn bind_group_entries<'a>(&'a self, gpu_images: &'a RenderAssets<GpuImage>) -> Vec<BindGroupEntry<'a>> {
		match self {
			Self::SingleBound { binding: (_, binding), storage } => vec![storage.bind_group_entry(*binding, gpu_images)],
			Self::SingleUnbound { .. } => vec![],
			Self::Double { binding: (_, (binding1, binding2)), storage: (storage1, storage2), front } => {
				let (storage1, storage2) =
					if *front == FrontBuffer::First { (storage2, storage1) } else { (storage1, storage2) };
				vec![storage1.bind_group_entry(*binding1, gpu_images), storage2.bind_group_entry(*binding2, gpu_images)]
			}
		}
	}

	fn bind_group_layout_entry(&self) -> Vec<BindGroupLayoutEntry> {
		match &self {
			&ShaderBufferInfo::SingleBound { binding: (_, binding), storage } => vec![BindGroupLayoutEntry {
				binding: *binding,
				visibility: ShaderStages::COMPUTE,
				ty: storage.bind_group_layout_entry_binding_type(None),
				count: None,
			}],
			ShaderBufferInfo::SingleUnbound { .. } => vec![],
			ShaderBufferInfo::Double { binding: (_, (binding1, binding2)), storage: (storage1, storage2), front } => {
				let (storage1, storage2) =
					if *front == FrontBuffer::First { (storage2, storage1) } else { (storage1, storage2) };
				vec![
					BindGroupLayoutEntry {
						binding: *binding1,
						visibility: ShaderStages::COMPUTE,
						ty: storage1.bind_group_layout_entry_binding_type(Some(StorageTextureAccess::ReadOnly)),
						count: None,
					},
					BindGroupLayoutEntry {
						binding: *binding2,
						visibility: ShaderStages::COMPUTE,
						ty: storage2.bind_group_layout_entry_binding_type(Some(StorageTextureAccess::WriteOnly)),
						count: None,
					},
				]
			}
		}
	}

	fn image_handle(&self) -> Option<Handle<Image>> {
		match &self {
			ShaderBufferInfo::SingleBound { storage, .. } | ShaderBufferInfo::SingleUnbound { storage } => {
				storage.image_handle()
			}
			ShaderBufferInfo::Double { storage: (storage1, storage2), front, .. } => {
				let storage = match front {
					FrontBuffer::First => storage1,
					FrontBuffer::Second => storage2,
				};
				storage.image_handle()
			}
		}
	}

	fn gpu_buffer(&self) -> Option<Buffer> {
		match &self {
			ShaderBufferInfo::SingleBound { storage, .. } | ShaderBufferInfo::SingleUnbound { storage } => {
				storage.gpu_buffer()
			}
			ShaderBufferInfo::Double { storage: (storage1, storage2), front, .. } => {
				let storage = match front {
					FrontBuffer::First => storage1,
					FrontBuffer::Second => storage2,
				};
				storage.gpu_buffer()
			}
		}
	}

	fn set<T: ShaderType + WriteInto + Clone>(&self, data: T, render_queue: &RenderQueue) {
		match &self {
			ShaderBufferInfo::SingleBound { storage, .. } => storage.set(data, render_queue),
			ShaderBufferInfo::SingleUnbound { storage, .. } => storage.set(data, render_queue),
			ShaderBufferInfo::Double { storage: (storage1, storage2), .. } => {
				storage1.set(data.clone(), render_queue);
				storage2.set(data, render_queue);
			}
		};
	}

	pub fn delete(&mut self, images: &mut Assets<Image>) {
		match self {
			ShaderBufferInfo::SingleBound { storage, .. } | ShaderBufferInfo::SingleUnbound { storage } => {
				storage.delete(images)
			}
			ShaderBufferInfo::Double { storage: (storage1, storage2), .. } => {
				storage1.delete(images);
				storage2.delete(images);
			}
		}
	}
}

/// Provides a system for managing all the buffers used by your shaders. This gives you the functions to add buffers, delete buffers, set the contents of buffers, and for texture buffers, to extract their image handle for display.
#[derive(Resource, Clone, ExtractResource)]
pub struct ShaderBufferSet {
	buffers: HashMap<u32, ShaderBufferInfo>,
	groups: Vec<Vec<u32>>,
	next_id: u32,
}

/// This is an opaque identifier you can store to reference a buffer again in the future.
#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub enum ShaderBufferHandle {
	#[doc(hidden)]
	Bound { group: u32, id: u32 },
	#[doc(hidden)]
	Unbound { id: u32 },
}

impl Display for ShaderBufferHandle {
	fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
		match self {
			ShaderBufferHandle::Bound { group, id } => {
				write!(f, "{{ group({}), id({}) }}", group, id)
			}
			ShaderBufferHandle::Unbound { id } => write!(f, "{{ id({}) }}", id),
		}
	}
}

fn bind_group_layout(buffers: &Vec<&ShaderBufferInfo>, device: &RenderDevice) -> BindGroupLayout {
	device.create_bind_group_layout(
		None,
		buffers.iter().flat_map(|buffer| buffer.bind_group_layout_entry()).collect::<Vec<_>>().as_slice(),
	)
}

impl ShaderBufferSet {
	pub(crate) fn new() -> Self { Self { buffers: HashMap::new(), groups: Vec::new(), next_id: 0 } }

	/// Add a new uninitialized storage buffer.
	/// - render_device: The [RenderDevice] resouce from Bevy.
	/// - size: The size of the buffer in bytes.
	/// - usage: See Bevy's [BufferUsages].
	/// - binding: How the buffer will be bound for access from the shader. See [Binding] for details. Specifying [Binding::Double] makes this a double buffer.
	/// - readonly: If true, then this buffer can only be read in the shader, and can't be written to.
	pub fn add_storage_uninit(
		&mut self, render_device: &RenderDevice, size: u32, usage: BufferUsages, binding: Binding, readonly: bool,
	) -> ShaderBufferHandle {
		self.store_buffer(binding, ShaderBufferInfo::new_storage_uninit(render_device, size, usage, binding, readonly))
	}

	/// Add a new storage buffer initialized to all zero bytes.
	/// - render_device: The [RenderDevice] resouce from Bevy.
	/// - size: The size of the buffer in bytes.
	/// - usage: See Bevy's [BufferUsages].
	/// - binding: How the buffer will be bound for access from the shader. See [Binding] for details. Specifying [Binding::Double] makes this a double buffer.
	/// - readonly: If true, then this buffer can only be read in the shader, and can't be written to.
	pub fn add_storage_zeroed(
		&mut self, render_device: &RenderDevice, size: u32, usage: BufferUsages, binding: Binding, readonly: bool,
	) -> ShaderBufferHandle {
		self.store_buffer(binding, ShaderBufferInfo::new_storage_zeroed(render_device, size, usage, binding, readonly))
	}

	/// Add a new storage buffer initialized with the provided data.
	/// - render_device: The [RenderDevice] resouce from Bevy.
	/// - render_queue: The [RenderQueue] resource from Bevy.
	/// - data: The data. Must implement the [ShaderType] trait. The buffer's size will be determined by the size of this data.
	/// - usage: See Bevy's [BufferUsages].
	/// - binding: How the buffer will be bound for access from the shader. See [Binding] for details. Specifying [Binding::Double] makes this a double buffer, in which case both buffers will be initialized with the provided data.
	/// - readonly: If true, then this buffer can only be read in the shader, and can't be written to.
	pub fn add_storage_init<T: ShaderType + WriteInto + Clone + Default>(
		&mut self, render_device: &RenderDevice, render_queue: &RenderQueue, data: T, usage: BufferUsages,
		binding: Binding, readonly: bool,
	) -> ShaderBufferHandle {
		self.store_buffer(
			binding,
			ShaderBufferInfo::new_storage_init(render_device, render_queue, data, usage, binding, readonly),
		)
	}

	/// Add a new uniform buffer initialized with the provided data.
	/// - render_device: The [RenderDevice] resouce from Bevy.
	/// - render_queue: The [RenderQueue] resource from Bevy.
	/// - data: The data. Must implement the [ShaderType] trait. The buffer's size will be determined by the size of this data.
	/// - usage: See Bevy's [BufferUsages].
	/// - binding: How the buffer will be bound for access from the shader. See [Binding] for details. Specifying [Binding::Double] makes this a double buffer, but given that uniform buffers are always read-only, there's little point to double buffering them.
	pub fn add_uniform_init<T: ShaderType + WriteInto + Clone + Default>(
		&mut self, render_device: &RenderDevice, render_queue: &RenderQueue, data: T, usage: BufferUsages, binding: Binding,
	) -> ShaderBufferHandle {
		self.store_buffer(binding, ShaderBufferInfo::new_uniform_init(render_device, render_queue, data, usage, binding))
	}

	/// Add a new texture buffer initialized with the provided solid color.
	/// - images: The `Assets<Image>` resource from Bevy.
	/// - width: The width of the texture in pixels.
	/// - height: The height of the texture in pixels.
	/// - format: The pixel format of the texture.
	/// - fill: One pixel's worth of data, provided as a byte array. The entire texture will be filled with this.
	/// - access: Whether this texture is read-only, write-only or read-write. This is ignored if the texture is double buffered.
	/// - binding: How the buffer will be bound for access from the shader. See [Binding] for details. Specifying [Binding::Double] makes this a double buffer, in which case the access mode specified in the previous argument is ignored.
	pub fn add_texture_fill(
		&mut self, images: &mut Assets<Image>, width: u32, height: u32, format: TextureFormat, fill: &[u8],
		access: StorageTextureAccess, binding: Binding,
	) -> ShaderBufferHandle {
		self
			.store_buffer(binding, ShaderBufferInfo::new_write_texture(images, width, height, format, fill, access, binding))
	}

	pub(crate) fn bind_groups(&self, device: &RenderDevice, gpu_images: &RenderAssets<GpuImage>) -> Vec<BindGroup> {
		self
			.groups
			.iter()
			.map(|buffer_ids| {
				let buffers = buffer_ids.iter().map(|id| self.buffers.get(id).unwrap()).collect::<Vec<_>>();
				device.create_bind_group(
					None,
					&bind_group_layout(&buffers, &device),
					buffers.iter().flat_map(|buffer| buffer.bind_group_entries(gpu_images)).collect::<Vec<_>>().as_slice(),
				)
			})
			.collect()
	}

	pub(crate) fn bind_group_layouts(&self, device: &RenderDevice) -> Vec<BindGroupLayout> {
		self
			.groups
			.iter()
			.map(|buffer_ids| {
				let buffers = buffer_ids.iter().map(|id| self.buffers.get(id).unwrap()).collect::<Vec<_>>();
				bind_group_layout(&buffers, device)
			})
			.collect()
	}

	/// Delete a buffer.
	/// - handle: The handle to the buffer to be deleted.
	/// - images: The `Assets<Image>` resource from Bevy.
	pub fn delete_buffer(&mut self, handle: ShaderBufferHandle, images: &mut Assets<Image>) {
		let buffer = match handle {
			ShaderBufferHandle::Bound { group, id, .. } => {
				let buffer = self.buffers.remove(&id);
				if let Some(buffers) = self.groups.get_mut(group as usize) {
					if let Some(index) = buffers.iter().position(|buffer_id| *buffer_id == id) {
						buffers.remove(index);
					}
				}
				buffer
			}
			ShaderBufferHandle::Unbound { id } => self.buffers.remove(&id),
		};
		if let Some(mut buffer) = buffer {
			buffer.delete(images);
		}
	}

	/// Get the image handle for a texture buffer. If the provided buffer isn't a texture buffer, it will just return `None`. If the provided buffer is a double buffer, it will return the image handle for the current front buffer.
	pub fn image_handle(&self, handle: ShaderBufferHandle) -> Option<Handle<Image>> {
		if let Some(buffer) = self.get_buffer(handle) {
			buffer.image_handle()
		} else {
			None
		}
	}

	/// Get the GPU buffer, as a [bevy_render::render_resource::buffer], for a storage or uniform buffer. If the provided buffer isn't a storage or uniform buffer, it will just return `None`. If the provided buffer is a double buffer, it will return the GPU buffer for the current front buffer.
	pub fn gpu_buffer(&self, handle: ShaderBufferHandle) -> Option<Buffer> {
		if let Some(buffer) = self.get_buffer(handle) {
			buffer.gpu_buffer()
		} else {
			None
		}
	}

	pub(crate) fn swap_front_buffer(&mut self, handle: ShaderBufferHandle) {
		let buffer = self.get_mut_buffer(handle);
		let Some(buffer) = buffer else {
			panic!("Attempted to set the front buffer of {}, but it doesn't exist", handle);
		};
		let ShaderBufferInfo::Double { front, .. } = buffer else {
			panic!("Attempt to set the front buffer of {}, which isn't a double buffer", handle);
		};
		*front = match front {
			FrontBuffer::First => FrontBuffer::Second,
			FrontBuffer::Second => FrontBuffer::First,
		}
	}

	/// Set the contents of a buffer. The data must be a type that implements [ShaderType], and it must match the size of the buffer. If this is a double buffer, the both buffers will be set.
	pub fn set_buffer<T: ShaderType + WriteInto + Clone>(
		&mut self, handle: ShaderBufferHandle, data: T, render_queue: &RenderQueue,
	) {
		if let Some(buffer) = self.get_buffer(handle) {
			buffer.set(data, render_queue);
		} else {
			panic!("Tried to set data on a non-existent buffer");
		}
	}

	fn store_buffer(&mut self, binding: Binding, buffer: ShaderBufferInfo) -> ShaderBufferHandle {
		let id = self.next_id;
		self.next_id += 1;
		self.buffers.insert(id, buffer);
		match binding {
			Binding::SingleBound(group, _) | Binding::Double(group, _) => {
				if group as usize >= self.groups.len() {
					self.groups.resize(group as usize + 1, Vec::new())
				}
				self.groups[group as usize].push(id);
				ShaderBufferHandle::Bound { group, id }
			}
			Binding::SingleUnbound => ShaderBufferHandle::Unbound { id },
		}
	}

	fn get_buffer(&self, handle: ShaderBufferHandle) -> Option<ShaderBufferInfo> {
		match handle {
			ShaderBufferHandle::Bound { id, .. } | ShaderBufferHandle::Unbound { id } => self.buffers.get(&id).cloned(),
		}
	}

	fn get_mut_buffer(&mut self, handle: ShaderBufferHandle) -> Option<&mut ShaderBufferInfo> {
		match handle {
			ShaderBufferHandle::Bound { id, .. } | ShaderBufferHandle::Unbound { id } => self.buffers.get_mut(&id),
		}
	}
}

fn extract_resources(mut commands: Commands, buffers: Extract<Option<Res<ShaderBufferSet>>>) {
	if let Some(buffers) = &*buffers {
		commands.insert_resource(ShaderBufferSet::extract_resource(&buffers));
	}
}

#[derive(Resource)]
pub(crate) struct ShaderBufferRenderSet {
	copy_buffers: HashMap<ShaderBufferHandle, Buffer>,
}

impl ShaderBufferRenderSet {
	fn new() -> Self { Self { copy_buffers: HashMap::new() } }

	pub fn create_copy_buffer(&mut self, handle: ShaderBufferHandle, buffers: &ShaderBufferSet, device: &RenderDevice) {
		if self.copy_buffers.contains_key(&handle) {
			panic!("Tried to create a copy buffer for {}, which already has one", handle);
		}
		let Some(src) = buffers.get_buffer(handle) else {
			panic!("Tried to create a copy buffer for {}, which does not exist", handle);
		};
		let storage = match &src {
			ShaderBufferInfo::SingleBound { storage, .. } | ShaderBufferInfo::SingleUnbound { storage } => storage,
			_ => panic!("Tried to create a copy buffer for {}, which is a double buffer", handle),
		};
		let ShaderBufferStorage::Storage { buffer: src, .. } = storage else {
			panic!("Tried to create a copy buffer for {}, which is not a storage buffer", handle);
		};
		let dst = ShaderBufferInfo::new_storage_uninit(
			device,
			src.size() as u32,
			BufferUsages::COPY_DST | BufferUsages::MAP_READ,
			Binding::SingleUnbound,
			false,
		);
		let ShaderBufferInfo::SingleUnbound { storage: dst_storage } = dst else {
			panic!("Tried to create a copy buffer for {}, but somehow it ended up not unbound", handle);
		};
		let ShaderBufferStorage::Storage { buffer: dst, .. } = dst_storage else {
			panic!("Tried to create a copy buffer for {}, but somehow it ended up as a non-storage buffer", handle);
		};
		self.copy_buffers.insert(handle, dst);
	}

	pub fn remove_copy_buffer(&mut self, handle: ShaderBufferHandle) {
		let Some(buffer) = self.copy_buffers.get(&handle) else {
			panic!("Tried to remove copy buffer for {}, but it doesn't have one", handle);
		};
		buffer.destroy();
		self.copy_buffers.remove(&handle);
	}

	pub fn copy_to_copy_buffer(
		&self, handle: ShaderBufferHandle, buffers: &ShaderBufferSet, context: &mut RenderContext,
	) {
		let Some(src) = buffers.get_buffer(handle) else {
			panic!("Tried to copy from buffer {}, which doesn't exist", handle);
		};
		let src_storage = match &src {
			ShaderBufferInfo::SingleBound { storage, .. } | ShaderBufferInfo::SingleUnbound { storage } => storage,
			_ => panic!("Tried to copy from buffer {}, which is a double buffer", handle),
		};
		let ShaderBufferStorage::Storage { buffer: src, .. } = src_storage else {
			panic!("Tried to copy from buffer {}, which is not a storage buffer", handle);
		};
		let Some(dst) = self.copy_buffers.get(&handle) else {
			panic!("Tried to copy {} to it's copy buffer, but it doesn't yet have one", handle);
		};
		let encoder = context.command_encoder();
		encoder.copy_buffer_to_buffer(&src, 0, &dst, 0, src.size());
	}

	pub fn copy_from_copy_buffer_to_vec(&self, handle: ShaderBufferHandle, device: &RenderDevice) -> Vec<u8> {
		if let Some(buffer) = self.copy_buffers.get(&handle) {
			let buffer_slice = buffer.slice(..);
			let (sender, receiver) = channel();
			buffer_slice.map_async(MapMode::Read, move |result| {
				sender.send(result).unwrap();
			});
			device.poll(PollType::Wait).unwrap();
			receiver.recv().unwrap().unwrap();
			let result = buffer_slice.get_mapped_range().to_vec();
			buffer.unmap();
			result
		} else {
			panic!("Tried to copy from buffer {} to vec when it has not yet been copied to a copy buffer", handle);
		}
	}
}

pub(crate) struct ShaderBufferSetPlugin;

impl Plugin for ShaderBufferSetPlugin {
	fn build(&self, app: &mut App) {
		app.insert_resource(ShaderBufferSet::new());
		app
			.sub_app_mut(RenderApp)
			.add_systems(ExtractSchedule, extract_resources)
			.insert_resource(ShaderBufferRenderSet::new());
	}
}
