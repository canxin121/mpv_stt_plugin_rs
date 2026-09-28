use crate::common::{MpvSttError, Result};
use std::fmt::Write as _;
use tracing::{debug, trace};
use std::process::{Command, Output, Stdio};
use std::time::Duration;
use wait_timeout::ChildExt;

fn format_cmd_for_error(label: &str) -> String {
    label.to_string()
}

pub fn run_capture_output(mut cmd: Command, label: &str, timeout: Duration) -> Result<Output> {
    cmd.stdin(Stdio::null());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    trace!(command = %describe(&cmd, label), "spawning a child process");
    let mut child = cmd.spawn().map_err(|e| {
        MpvSttError::ProcessFailed(format!(
            "Failed to spawn {}: {}",
            format_cmd_for_error(label),
            e
        ))
    })?;

    match child.wait_timeout(timeout).map_err(|e| {
        MpvSttError::ProcessFailed(format!(
            "Failed waiting for {}: {}",
            format_cmd_for_error(label),
            e
        ))
    })? {
        Some(status) => {
            let output = child.wait_with_output()?;
            debug!(
                label,
                status = %status,
                stdout_bytes = output.stdout.len(),
                stderr_bytes = output.stderr.len(),
                "child process finished"
            );
            Ok(output)
        }
        None => {
            let _ = child.kill();
            let _ = child.wait();
            Err(MpvSttError::ProcessTimeout(format!(
                "{} timed out after {}ms",
                format_cmd_for_error(label),
                timeout.as_millis()
            )))
        }
    }
}

/// The command line, minus the program's own arguments only where they would
/// carry a secret: nothing here is a key today, so the whole line is safe.
fn describe(cmd: &Command, label: &str) -> String {
    let mut text = format!("{label}: {}", cmd.get_program().to_string_lossy());
    for arg in cmd.get_args() {
        let _ = write!(text, " {}", arg.to_string_lossy());
    }
    text
}

pub fn run_capture_output_with_stdin(
    mut cmd: Command,
    label: &str,
    stdin_bytes: &[u8],
    timeout: Duration,
) -> Result<Output> {
    cmd.stdin(Stdio::piped());
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    let mut child = cmd.spawn().map_err(|e| {
        MpvSttError::ProcessFailed(format!(
            "Failed to spawn {}: {}",
            format_cmd_for_error(label),
            e
        ))
    })?;

    trace!(
        command = %describe(&cmd, label),
        stdin_bytes = stdin_bytes.len(),
        "spawning a child process"
    );
    if let Some(mut stdin) = child.stdin.take() {
        use std::io::Write;
        stdin.write_all(stdin_bytes)?;
    }

    match child.wait_timeout(timeout).map_err(|e| {
        MpvSttError::ProcessFailed(format!(
            "Failed waiting for {}: {}",
            format_cmd_for_error(label),
            e
        ))
    })? {
        Some(status) => {
            let output = child.wait_with_output()?;
            debug!(
                label,
                status = %status,
                stdout_bytes = output.stdout.len(),
                stderr_bytes = output.stderr.len(),
                "child process finished"
            );
            Ok(output)
        }
        None => {
            let _ = child.kill();
            let _ = child.wait();
            Err(MpvSttError::ProcessTimeout(format!(
                "{} timed out after {}ms",
                format_cmd_for_error(label),
                timeout.as_millis()
            )))
        }
    }
}
