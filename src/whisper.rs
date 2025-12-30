// Backend dispatcher. Select exactly one Whisper backend via cargo features.

// Enforce one-and-only-one backend feature.
#[cfg(not(any(feature = "whisper_cpp_cpu", feature = "whisper_cpp_cuda",)))]
compile_error!(
    "No Whisper backend selected. Enable exactly one of: whisper_cpp_cpu, whisper_cpp_cuda."
);

#[cfg(any(all(feature = "whisper_cpp_cpu", feature = "whisper_cpp_cuda"),))]
compile_error!("Multiple Whisper backends enabled. Choose exactly one backend feature.");

// Android limitation: only whisper_cpp_cpu is supported.
#[cfg(target_os = "android")]
#[cfg(any(feature = "whisper_cpp_cuda",))]
compile_error!("Android supports only the whisper_cpp_cpu backend.");

#[cfg(any(feature = "whisper_cpp_cpu", feature = "whisper_cpp_cuda"))]
#[path = "whisper_cpp.rs"]
mod whisper_cpp;

#[cfg(any(feature = "whisper_cpp_cpu", feature = "whisper_cpp_cuda"))]
pub use whisper_cpp::{WhisperConfig, WhisperDeviceNotice, WhisperRunner};
