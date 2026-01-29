//! WASM bindings for backend types.

use crate::backend::{Backend, DeviceCapabilities, NullBackend};
use wasm_bindgen::prelude::*;

/// Device capabilities information.
#[wasm_bindgen]
pub struct JsDeviceCapabilities {
    inner: DeviceCapabilities,
}

#[wasm_bindgen]
impl JsDeviceCapabilities {
    /// Check if the device supports f16 (half precision).
    #[wasm_bindgen(getter, js_name = supportsF16)]
    pub fn supports_f16(&self) -> bool {
        self.inner.supports_f16
    }

    /// Check if the device supports bf16 (bfloat16).
    #[wasm_bindgen(getter, js_name = supportsBf16)]
    pub fn supports_bf16(&self) -> bool {
        self.inner.supports_bf16
    }

    /// Check if the device supports tf32 (TensorFloat-32).
    #[wasm_bindgen(getter, js_name = supportsTf32)]
    pub fn supports_tf32(&self) -> bool {
        self.inner.supports_tf32
    }

    /// Get maximum memory in bytes, if known.
    #[wasm_bindgen(getter, js_name = maxMemoryBytes)]
    pub fn max_memory_bytes(&self) -> Option<u64> {
        self.inner.max_memory_bytes
    }

    /// Get number of compute units, if known.
    #[wasm_bindgen(getter, js_name = computeUnits)]
    pub fn compute_units(&self) -> Option<u32> {
        self.inner.compute_units
    }
}

impl JsDeviceCapabilities {
    pub(crate) fn from_inner(inner: DeviceCapabilities) -> Self {
        Self { inner }
    }
}

/// A null/mock backend for testing and development.
///
/// This backend doesn't perform actual computation but provides
/// a reference implementation of the Backend trait.
#[wasm_bindgen]
pub struct JsNullBackend {
    inner: NullBackend,
}

#[wasm_bindgen]
impl JsNullBackend {
    /// Create a new NullBackend for the specified device.
    #[wasm_bindgen(constructor)]
    pub fn new(device: &str) -> Self {
        Self {
            inner: NullBackend::new(device.to_string()),
        }
    }

    /// Create a NullBackend configured for CPU.
    #[wasm_bindgen]
    pub fn cpu() -> Self {
        Self {
            inner: NullBackend::cpu(),
        }
    }

    /// Get the backend name.
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.inner.name().to_string()
    }

    /// Get the device identifier.
    #[wasm_bindgen(getter)]
    pub fn device(&self) -> String {
        self.inner.device().to_string()
    }

    /// Check if the backend is available.
    #[wasm_bindgen(js_name = isAvailable)]
    pub fn is_available(&self) -> bool {
        self.inner.is_available()
    }

    /// Get device capabilities.
    #[wasm_bindgen]
    pub fn capabilities(&self) -> JsDeviceCapabilities {
        JsDeviceCapabilities::from_inner(self.inner.capabilities())
    }
}

impl JsNullBackend {
    pub(crate) fn inner(&self) -> &NullBackend {
        &self.inner
    }
}
