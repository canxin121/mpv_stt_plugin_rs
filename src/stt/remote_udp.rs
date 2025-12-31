use super::{BackendKind, SttBackend, SttDeviceNotice};
use crate::crypto::{AuthToken, EncryptionKey};
use crate::error::{Result, MpvSttPluginRsError};
use crate::srt::SrtFile;
use log::{debug, info, trace};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::{SocketAddr, UdpSocket};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::time::{Duration, Instant, SystemTime};
#[cfg(feature = "stt_remote_udp")]
use libopusenc_static_sys as opusenc;
#[cfg(feature = "stt_remote_udp")]
use std::ffi::CStr;
#[cfg(feature = "stt_remote_udp")]
use std::os::raw::{c_int, c_void};
#[cfg(feature = "stt_remote_udp")]
use std::ptr::NonNull;

#[cfg(feature = "stt_remote_udp")]
const OPUS_SET_APPLICATION_REQUEST: c_int = 4000;
#[cfg(feature = "stt_remote_udp")]
const OPUS_APPLICATION_VOIP: c_int = 2048;

#[cfg(feature = "stt_remote_udp")]
fn ope_error_message(code: c_int) -> String {
    unsafe {
        let ptr = opusenc::ope_strerror(code);
        if ptr.is_null() {
            format!("OpusEnc error {code}")
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}

#[cfg(feature = "stt_remote_udp")]
struct PacketSink {
    packets: Vec<Vec<u8>>,
    seen_packets: usize,
}

#[cfg(feature = "stt_remote_udp")]
impl PacketSink {
    fn new() -> Self {
        Self {
            packets: Vec::new(),
            seen_packets: 0,
        }
    }
}

#[cfg(feature = "stt_remote_udp")]
unsafe extern "C" fn opusenc_write_cb(
    _user_data: *mut c_void,
    _ptr: *const ::core::ffi::c_uchar,
    _len: opusenc::opus_int32,
) -> c_int {
    0
}

#[cfg(feature = "stt_remote_udp")]
unsafe extern "C" fn opusenc_close_cb(_user_data: *mut c_void) -> c_int {
    0
}

#[cfg(feature = "stt_remote_udp")]
unsafe extern "C" fn opusenc_packet_cb(
    user_data: *mut c_void,
    packet_ptr: *const ::core::ffi::c_uchar,
    packet_len: opusenc::opus_int32,
    _flags: opusenc::opus_uint32,
) {
    if user_data.is_null() || packet_ptr.is_null() || packet_len <= 0 {
        return;
    }
    let sink = unsafe { &mut *(user_data as *mut PacketSink) };
    sink.seen_packets += 1;
    if sink.seen_packets <= 2 {
        return;
    }
    let data = unsafe { std::slice::from_raw_parts(packet_ptr, packet_len as usize) };
    sink.packets.push(data.to_vec());
}

#[cfg(feature = "stt_remote_udp")]
struct OpusEncHandle {
    ptr: NonNull<opusenc::OggOpusEnc>,
}

#[cfg(feature = "stt_remote_udp")]
impl OpusEncHandle {
    fn new(sample_rate: i32, channels: i32) -> Result<Self> {
        let comments = unsafe { opusenc::ope_comments_create() };
        if comments.is_null() {
            return Err(MpvSttPluginRsError::SttFailed(
                "OpusEnc comments init failed".to_string(),
            ));
        }
        let callbacks = opusenc::OpusEncCallbacks {
            write: Some(opusenc_write_cb),
            close: Some(opusenc_close_cb),
        };
        let mut error: c_int = 0;
        let ptr = unsafe {
            opusenc::ope_encoder_create_callbacks(
                &callbacks,
                std::ptr::null_mut(),
                comments,
                sample_rate as opusenc::opus_int32,
                channels,
                0,
                &mut error,
            )
        };
        unsafe {
            opusenc::ope_comments_destroy(comments);
        }
        let ptr = NonNull::new(ptr).ok_or_else(|| {
            MpvSttPluginRsError::SttFailed(format!(
                "OpusEnc encoder init failed: {}",
                ope_error_message(error)
            ))
        })?;
        if error != 0 {
            return Err(MpvSttPluginRsError::SttFailed(format!(
                "OpusEnc encoder init failed: {}",
                ope_error_message(error)
            )));
        }
        Ok(Self { ptr })
    }

    fn set_packet_callback(&mut self, sink: *mut PacketSink) -> Result<()> {
        let ret = unsafe {
            opusenc::ope_encoder_ctl(
                self.ptr.as_ptr(),
                opusenc::OPE_SET_PACKET_CALLBACK_REQUEST as c_int,
                Some(opusenc_packet_cb),
                sink as *mut c_void,
            )
        };
        if ret != 0 {
            return Err(MpvSttPluginRsError::SttFailed(format!(
                "OpusEnc packet callback setup failed: {}",
                ope_error_message(ret)
            )));
        }
        Ok(())
    }

    fn set_application(&mut self, application: c_int) -> Result<()> {
        let ret = unsafe {
            opusenc::ope_encoder_ctl(
                self.ptr.as_ptr(),
                OPUS_SET_APPLICATION_REQUEST,
                application as opusenc::opus_int32,
            )
        };
        if ret != 0 {
            return Err(MpvSttPluginRsError::SttFailed(format!(
                "OpusEnc set application failed: {}",
                ope_error_message(ret)
            )));
        }
        Ok(())
    }

    fn write_samples(&mut self, samples: &[i16]) -> Result<()> {
        let ret = unsafe {
            opusenc::ope_encoder_write(
                self.ptr.as_ptr(),
                samples.as_ptr() as *const opusenc::opus_int16,
                samples.len() as c_int,
            )
        };
        if ret != 0 {
            return Err(MpvSttPluginRsError::SttFailed(format!(
                "OpusEnc write failed: {}",
                ope_error_message(ret)
            )));
        }
        Ok(())
    }

    fn drain(&mut self) -> Result<()> {
        let ret = unsafe { opusenc::ope_encoder_drain(self.ptr.as_ptr()) };
        if ret != 0 {
            return Err(MpvSttPluginRsError::SttFailed(format!(
                "OpusEnc drain failed: {}",
                ope_error_message(ret)
            )));
        }
        Ok(())
    }
}

#[cfg(feature = "stt_remote_udp")]
impl Drop for OpusEncHandle {
    fn drop(&mut self) {
        unsafe {
            opusenc::ope_encoder_destroy(self.ptr.as_ptr());
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum CompressionFormat {
    Opus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum Message {
    AudioChunk {
        request_id: u64,
        chunk_index: u32,
        total_chunks: u32,
        duration_ms: u64,
        data: Vec<u8>,
        auth_token: [u8; 32],
        compression: CompressionFormat,
    },
    Cancel {
        request_id: u64,
        auth_token: [u8; 32],
    },
    Result {
        request_id: u64,
        chunk_index: u32,
        total_chunks: u32,
        data: Vec<u8>,
    },
    Error {
        request_id: u64,
        message: String,
    },
}

impl Message {
    fn encode(&self, encryption_key: Option<&EncryptionKey>) -> Result<Vec<u8>> {
        let serialized = postcard::to_allocvec(self)
            .map_err(|e| MpvSttPluginRsError::SttFailed(format!("encode error: {}", e)))?;

        if let Some(key) = encryption_key {
            key.encrypt(&serialized)
        } else {
            Ok(serialized)
        }
    }

    fn decode(data: &[u8], encryption_key: Option<&EncryptionKey>) -> Result<Self> {
        let decrypted = if let Some(key) = encryption_key {
            key.decrypt(data)?
        } else {
            data.to_vec()
        };

        postcard::from_bytes(&decrypted)
            .map_err(|e| MpvSttPluginRsError::SttFailed(format!("decode error: {}", e)))
    }
}

pub struct RemoteSttConfig {
    pub server_addr: String,
    pub timeout_ms: u64,
    pub max_retry: usize,
    pub enable_encryption: bool,
    pub encryption_key: String,
    pub auth_secret: String,
    // Compression is always enabled; config is kept for future tuning
}

impl Default for RemoteSttConfig {
    fn default() -> Self {
        Self {
            server_addr: "127.0.0.1:9000".to_string(),
            timeout_ms: 120_000,
            max_retry: 3,
            enable_encryption: false,
            encryption_key: String::new(),
            auth_secret: String::new(),
        }
    }
}

pub struct RemoteUdpBackend {
    config: RemoteSttConfig,
    socket: UdpSocket,
    server_addr: SocketAddr,
    cancel_generation: Arc<AtomicU64>,
    encryption_key: Option<EncryptionKey>,
    auth_token: AuthToken,
}

impl RemoteUdpBackend {
    pub fn new(config: RemoteSttConfig) -> Result<Self> {
        let socket = UdpSocket::bind("0.0.0.0:0")?;
        let server_addr = config
            .server_addr
            .parse()
            .map_err(|e| MpvSttPluginRsError::SttFailed(format!("invalid server address: {}", e)))?;

        socket.set_read_timeout(Some(Duration::from_millis(5000)))?;

        let encryption_key = if config.enable_encryption {
            if config.encryption_key.is_empty() {
                return Err(MpvSttPluginRsError::SttFailed(
                    "Encryption enabled but encryption_key is empty".to_string(),
                ));
            }
            Some(EncryptionKey::from_passphrase(&config.encryption_key))
        } else {
            None
        };

        let auth_token = if !config.auth_secret.is_empty() {
            AuthToken::from_secret(&config.auth_secret)
        } else {
            AuthToken::from_secret("")
        };

        Ok(Self {
            config,
            socket,
            server_addr,
            cancel_generation: Arc::new(AtomicU64::new(0)),
            encryption_key,
            auth_token,
        })
    }

    fn transcribe_impl<P: AsRef<Path>>(
        &mut self,
        audio_path: P,
        output_prefix: P,
        duration_ms: u64,
    ) -> Result<()> {
        let audio_str = audio_path
            .as_ref()
            .to_str()
            .ok_or_else(|| MpvSttPluginRsError::InvalidPath("Invalid audio path".to_string()))?;

        trace!(
            "Remote UDP STT: {} (duration: {}ms, compression: Opus)",
            audio_str, duration_ms
        );

        let run_generation = self.cancel_generation.load(Ordering::Relaxed);

        let audio_data = self.compress_audio(&audio_path)?;

        if audio_data.is_empty() {
            return Err(MpvSttPluginRsError::SttFailed(
                "Audio data is empty".to_string(),
            ));
        }

        let request_id = self.generate_request_id();
        let srt_data =
            self.send_request_with_retry(request_id, &audio_data, duration_ms, run_generation)?;

        if self.cancel_generation.load(Ordering::Relaxed) != run_generation {
            return Err(MpvSttPluginRsError::SttCancelled);
        }

        let srt_file = SrtFile::parse_content(&String::from_utf8_lossy(&srt_data))?;
        let output_path = PathBuf::from(output_prefix.as_ref()).with_extension("srt");
        srt_file.save(&output_path)?;

        debug!("Remote UDP STT completed successfully");
        Ok(())
    }

    fn generate_request_id(&self) -> u64 {
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }

    fn send_request_with_retry(
        &self,
        request_id: u64,
        audio: &[u8],
        duration_ms: u64,
        run_generation: u64,
    ) -> Result<Vec<u8>> {
        let mut last_error = None;

        for attempt in 0..self.config.max_retry {
            if self.cancel_generation.load(Ordering::Relaxed) != run_generation {
                self.send_cancel(request_id)?;
                return Err(MpvSttPluginRsError::SttCancelled);
            }

            match self.send_request(request_id, audio, duration_ms, run_generation) {
                Ok(result) => return Ok(result),
                Err(e) => {
                    last_error = Some(e);
                    if attempt + 1 < self.config.max_retry {
                        debug!("UDP request attempt {} failed, retrying...", attempt + 1);
                        std::thread::sleep(Duration::from_millis(500));
                    }
                }
            }
        }

        Err(last_error.unwrap())
    }

    fn send_request(
        &self,
        request_id: u64,
        audio: &[u8],
        duration_ms: u64,
        run_generation: u64,
    ) -> Result<Vec<u8>> {
        self.send_audio_chunks(request_id, audio, duration_ms)?;
        self.receive_response(request_id, run_generation)
    }

    fn compress_audio<P: AsRef<Path>>(&self, audio_path: P) -> Result<Vec<u8>> {
        #[cfg(feature = "stt_remote_udp")]
        {
            use hound::WavReader;

            let reader = WavReader::open(audio_path)
                .map_err(|e| MpvSttPluginRsError::SttFailed(format!("Failed to read WAV: {}", e)))?;

            let spec = reader.spec();
            if spec.channels != 1 || spec.sample_rate != 16000 {
                return Err(MpvSttPluginRsError::SttFailed(format!(
                    "Unsupported WAV format for compression: {}ch {}Hz",
                    spec.channels, spec.sample_rate
                )));
            }

            let samples: Vec<i16> = reader
                .into_samples::<i16>()
                .collect::<std::result::Result<Vec<i16>, _>>()
                .map_err(|e| {
                    MpvSttPluginRsError::SttFailed(format!("Failed to read samples: {}", e))
                })?;

            let mut sink = PacketSink::new();
            let mut encoder = OpusEncHandle::new(16000, 1)?;
            encoder.set_packet_callback(&mut sink)?;
            encoder.set_application(OPUS_APPLICATION_VOIP)?;
            encoder.write_samples(&samples)?;
            encoder.drain()?;

            let mut compressed = Vec::new();
            for packet in sink.packets {
                compressed.extend_from_slice(&(packet.len() as u32).to_le_bytes());
                compressed.extend_from_slice(&packet);
            }

            info!(
                "Opus compression: {} samples → {} bytes",
                samples.len(),
                compressed.len()
            );
            Ok(compressed)
        }
        #[cfg(not(feature = "stt_remote_udp"))]
        {
            Err(MpvSttPluginRsError::SttFailed(
                "Compression not available".to_string(),
            ))
        }
    }

    fn send_audio_chunks(&self, request_id: u64, audio: &[u8], duration_ms: u64) -> Result<()> {
        const MAX_PAYLOAD: usize = 60000;

        let chunks: Vec<Vec<u8>> = audio.chunks(MAX_PAYLOAD).map(|c| c.to_vec()).collect();
        let total_chunks = chunks.len() as u32;

        let compression = CompressionFormat::Opus;

        for (index, chunk_data) in chunks.into_iter().enumerate() {
            let message = Message::AudioChunk {
                request_id,
                chunk_index: index as u32,
                total_chunks,
                duration_ms,
                data: chunk_data,
                auth_token: *self.auth_token.as_bytes(),
                compression,
            };

            let packet = message.encode(self.encryption_key.as_ref())?;
            self.socket.send_to(&packet, self.server_addr)?;
        }

        info!(
            "Sent {} audio chunks ({} bytes, compression: {:?}) for request {}",
            total_chunks,
            audio.len(),
            compression,
            request_id
        );
        Ok(())
    }

    fn receive_response(&self, request_id: u64, run_generation: u64) -> Result<Vec<u8>> {
        let deadline = Instant::now() + Duration::from_millis(self.config.timeout_ms);
        let mut result_chunks: HashMap<u32, Vec<u8>> = HashMap::new();
        let mut buf = vec![0u8; 65507];

        loop {
            if self.cancel_generation.load(Ordering::Relaxed) != run_generation {
                self.send_cancel(request_id)?;
                return Err(MpvSttPluginRsError::SttCancelled);
            }

            if Instant::now() > deadline {
                self.send_cancel(request_id)?;
                return Err(MpvSttPluginRsError::SttFailed(
                    "UDP request timed out".to_string(),
                ));
            }

            match self.socket.recv(&mut buf) {
                Ok(len) => {
                    let message = Message::decode(&buf[..len], self.encryption_key.as_ref())?;

                    match message {
                        Message::Result {
                            request_id: resp_id,
                            chunk_index,
                            total_chunks,
                            data,
                        } if resp_id == request_id => {
                            result_chunks.insert(chunk_index, data);

                            if result_chunks.len() == total_chunks as usize {
                                let mut srt_data = Vec::new();
                                for i in 0..total_chunks {
                                    if let Some(chunk) = result_chunks.get(&i) {
                                        srt_data.extend_from_slice(chunk);
                                    } else {
                                        return Err(MpvSttPluginRsError::SttFailed(format!(
                                            "Missing result chunk {}",
                                            i
                                        )));
                                    }
                                }
                                return Ok(srt_data);
                            }
                        }
                        Message::Error {
                            request_id: resp_id,
                            message: msg,
                        } if resp_id == request_id => {
                            return Err(MpvSttPluginRsError::SttFailed(format!(
                                "Server error: {}",
                                msg
                            )));
                        }
                        _ => {}
                    }
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(50));
                    continue;
                }
                Err(e) => {
                    return Err(MpvSttPluginRsError::SttFailed(format!(
                        "UDP receive error: {}",
                        e
                    )));
                }
            }
        }
    }

    fn send_cancel(&self, request_id: u64) -> Result<()> {
        let message = Message::Cancel {
            request_id,
            auth_token: *self.auth_token.as_bytes(),
        };
        let packet = message.encode(self.encryption_key.as_ref())?;
        self.socket.send_to(&packet, self.server_addr)?;
        debug!("Sent cancel for request {}", request_id);
        Ok(())
    }
}

impl SttBackend for RemoteUdpBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::RemoteUdp
    }

    fn transcribe<P: AsRef<Path>>(
        &mut self,
        audio_path: P,
        output_prefix: P,
        duration_ms: u64,
    ) -> Result<()> {
        self.transcribe_impl(audio_path, output_prefix, duration_ms)
    }

    fn cancel_inflight(&self) {
        self.cancel_generation.fetch_add(1, Ordering::Relaxed);
    }

    fn take_device_notice(&mut self) -> Option<SttDeviceNotice> {
        None
    }
}
