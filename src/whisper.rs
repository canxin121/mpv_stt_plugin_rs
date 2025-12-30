// Backend dispatcher. Select exactly one Whisper backend via cargo features.

// Enforce one-and-only-one backend feature.
#[cfg(not(any(
    feature = "whisper_cpp_cpu",
    feature = "whisper_cpp_cuda",
    feature = "fast_whisper_cpu",
    feature = "fast_whisper_cuda",
)))]
compile_error!("No Whisper backend selected. Enable exactly one of: whisper_cpp_cpu, whisper_cpp_cuda, fast_whisper_cpu, fast_whisper_cuda.");

#[cfg(any(
    all(feature = "whisper_cpp_cpu", feature = "whisper_cpp_cuda"),
    all(feature = "whisper_cpp_cpu", feature = "fast_whisper_cpu"),
    all(feature = "whisper_cpp_cpu", feature = "fast_whisper_cuda"),
    all(feature = "whisper_cpp_cuda", feature = "fast_whisper_cpu"),
    all(feature = "whisper_cpp_cuda", feature = "fast_whisper_cuda"),
    all(feature = "fast_whisper_cpu", feature = "fast_whisper_cuda"),
))]
compile_error!("Multiple Whisper backends enabled. Choose exactly one backend feature.");

// Android limitation: only whisper_cpp_cpu is supported.
#[cfg(target_os = "android")]
#[cfg(any(
    feature = "whisper_cpp_cuda",
    feature = "fast_whisper_cpu",
    feature = "fast_whisper_cuda",
))]
compile_error!("Android supports only the whisper_cpp_cpu backend.");

// fast_whisper CUDA backend is Linux-only.
#[cfg(feature = "fast_whisper_cuda")]
#[cfg(not(target_os = "linux"))]
compile_error!("fast_whisper_cuda is supported only on Linux.");

#[cfg(any(feature = "whisper_cpp_cpu", feature = "whisper_cpp_cuda"))]
mod whisper_cpp;
#[cfg(any(feature = "fast_whisper_cpu", feature = "fast_whisper_cuda"))]
mod fast_whisper;

#[cfg(any(feature = "whisper_cpp_cpu", feature = "whisper_cpp_cuda"))]
pub use whisper_cpp::{WhisperConfig, WhisperDeviceNotice, WhisperRunner};

#[cfg(any(feature = "fast_whisper_cpu", feature = "fast_whisper_cuda"))]
pub use fast_whisper::{WhisperConfig, WhisperDeviceNotice, WhisperRunner};
