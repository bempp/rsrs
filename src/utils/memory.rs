//! Lightweight process memory reporting helpers.

use std::sync::{
    atomic::{AtomicU64, Ordering},
    OnceLock,
};

#[derive(Debug, Clone, Copy, Default)]
pub struct ProcessMemoryUsage {
    pub resident_bytes: Option<u64>,
    pub peak_resident_bytes: Option<u64>,
}

static TRACE_LAST_RSS_BYTES: AtomicU64 = AtomicU64::new(0);
static TRACE_ENABLED: OnceLock<bool> = OnceLock::new();
static TRACE_DELTA_BYTES: OnceLock<u64> = OnceLock::new();

pub fn format_bytes(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

    let bytes_f64 = bytes as f64;
    if bytes_f64 >= GIB {
        format!("{:.2} GiB", bytes_f64 / GIB)
    } else if bytes_f64 >= MIB {
        format!("{:.2} MiB", bytes_f64 / MIB)
    } else if bytes_f64 >= KIB {
        format!("{:.2} KiB", bytes_f64 / KIB)
    } else {
        format!("{bytes} B")
    }
}

pub fn process_memory_usage() -> ProcessMemoryUsage {
    platform::process_memory_usage()
}

pub fn matrix_bytes<Item>(rows: usize, cols: usize) -> u64 {
    (rows as u64) * (cols as u64) * (std::mem::size_of::<Item>() as u64)
}

pub fn memory_trace_enabled() -> bool {
    *TRACE_ENABLED.get_or_init(|| {
        std::env::var("RSRS_TRACE_MEMORY")
            .map(|value| value != "0")
            .unwrap_or(false)
    })
}

fn trace_delta_bytes() -> u64 {
    *TRACE_DELTA_BYTES.get_or_init(|| {
        std::env::var("RSRS_TRACE_MEMORY_DELTA_MB")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .map(|mb| mb * 1024 * 1024)
            .unwrap_or(4 * 1024 * 1024)
    })
}

pub fn trace_memory_growth(label: &str, estimated_bytes: Option<u64>) {
    if !memory_trace_enabled() {
        return;
    }

    let usage = process_memory_usage();
    let Some(rss) = usage.resident_bytes else {
        return;
    };

    let last_rss = TRACE_LAST_RSS_BYTES.load(Ordering::Relaxed);
    if last_rss != 0 && rss < last_rss.saturating_add(trace_delta_bytes()) {
        return;
    }

    TRACE_LAST_RSS_BYTES.store(rss, Ordering::Relaxed);

    let peak = usage
        .peak_resident_bytes
        .map(format_bytes)
        .unwrap_or_else(|| "unavailable".to_string());

    match estimated_bytes {
        Some(bytes) => println!(
            "Memory trace [{label}]: rss = {}, peak = {peak}, est tmp ~= {}",
            format_bytes(rss),
            format_bytes(bytes)
        ),
        None => println!(
            "Memory trace [{label}]: rss = {}, peak = {peak}",
            format_bytes(rss)
        ),
    }
}

pub fn trace_memory_event(label: &str, estimated_bytes: Option<u64>) {
    if !memory_trace_enabled() {
        return;
    }

    let usage = process_memory_usage();
    let rss = usage
        .resident_bytes
        .map(format_bytes)
        .unwrap_or_else(|| "unavailable".to_string());
    let peak = usage
        .peak_resident_bytes
        .map(format_bytes)
        .unwrap_or_else(|| "unavailable".to_string());

    match estimated_bytes {
        Some(bytes) => println!(
            "Memory event [{label}]: rss = {rss}, peak = {peak}, est tmp ~= {}",
            format_bytes(bytes)
        ),
        None => println!("Memory event [{label}]: rss = {rss}, peak = {peak}"),
    }
}

#[cfg(target_os = "macos")]
mod platform {
    use super::ProcessMemoryUsage;
    use std::{mem::size_of, os::raw::c_int};

    type KernReturn = c_int;
    type MachPort = u32;
    type TaskFlavor = u32;
    type TaskInfoCount = u32;

    const KERN_SUCCESS: KernReturn = 0;
    const MACH_TASK_BASIC_INFO: TaskFlavor = 20;

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct TimeValue {
        seconds: c_int,
        microseconds: c_int,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct MachTaskBasicInfo {
        virtual_size: u64,
        resident_size: u64,
        resident_size_max: u64,
        user_time: TimeValue,
        system_time: TimeValue,
        policy: c_int,
        suspend_count: c_int,
    }

    unsafe extern "C" {
        fn mach_task_self() -> MachPort;
        fn task_info(
            target_task: MachPort,
            flavor: TaskFlavor,
            task_info_out: *mut c_int,
            task_info_out_count: *mut TaskInfoCount,
        ) -> KernReturn;
    }

    pub fn process_memory_usage() -> ProcessMemoryUsage {
        let mut info = MachTaskBasicInfo::default();
        let mut count = (size_of::<MachTaskBasicInfo>() / size_of::<c_int>()) as TaskInfoCount;

        let result = unsafe {
            task_info(
                mach_task_self(),
                MACH_TASK_BASIC_INFO,
                &mut info as *mut _ as *mut c_int,
                &mut count,
            )
        };

        if result == KERN_SUCCESS {
            ProcessMemoryUsage {
                resident_bytes: Some(info.resident_size),
                peak_resident_bytes: Some(info.resident_size_max),
            }
        } else {
            ProcessMemoryUsage::default()
        }
    }
}

#[cfg(target_os = "linux")]
mod platform {
    use super::ProcessMemoryUsage;

    fn parse_kib(contents: &str, key: &str) -> Option<u64> {
        contents.lines().find_map(|line| {
            let rest = line.strip_prefix(key)?.trim();
            let value = rest.split_whitespace().next()?.parse::<u64>().ok()?;
            Some(value * 1024)
        })
    }

    pub fn process_memory_usage() -> ProcessMemoryUsage {
        let Ok(contents) = std::fs::read_to_string("/proc/self/status") else {
            return ProcessMemoryUsage::default();
        };

        ProcessMemoryUsage {
            resident_bytes: parse_kib(&contents, "VmRSS:"),
            peak_resident_bytes: parse_kib(&contents, "VmHWM:"),
        }
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
mod platform {
    use super::ProcessMemoryUsage;

    pub fn process_memory_usage() -> ProcessMemoryUsage {
        ProcessMemoryUsage::default()
    }
}
