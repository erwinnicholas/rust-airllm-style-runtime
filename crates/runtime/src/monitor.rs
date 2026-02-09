use sysinfo::{Pid, System}; // removed ProcessExt, SystemExt
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use std::thread;
use std::time::Duration;

pub struct SystemMonitor {
    stop_signal: Arc<AtomicBool>,
}

impl SystemMonitor {
    pub fn start(interval_ms: u64) -> Self {
        let stop_signal = Arc::new(AtomicBool::new(false));
        let signal_clone = stop_signal.clone();

        thread::spawn(move || {
            // 1. Create the system object
            let mut sys = System::new_all();
            
            // 2. Get current PID (sysinfo uses its own Pid type)
            let pid = Pid::from(std::process::id() as usize);

            println!("[Monitor] Background thread started. Tracking PID: {}", pid);

            while !signal_clone.load(Ordering::Relaxed) {
                // 3. Refresh process info. 
                // In sysinfo 0.30+, 'refresh_all' is the safest generic way to update everything.
                // (Optimized usage would use refresh_processes_specifics, but this is fine for V1)
                sys.refresh_all();

                if let Some(process) = sys.process(pid) {
                    // sysinfo returns memory in BYTES
                    let memory_bytes = process.memory(); 
                    let memory_mb = memory_bytes as f64 / 1024.0 / 1024.0;
                    let cpu_usage = process.cpu_usage();
                    
                    print!("\r[Monitor] OS RAM: {:.2} MB | CPU: {:.1}% ", 
                        memory_mb, 
                        cpu_usage
                    );
                }
                
                thread::sleep(Duration::from_millis(interval_ms));
            }
            println!("\n[Monitor] Stopped.");
        });

        Self { stop_signal }
    }

    pub fn stop(&self) {
        self.stop_signal.store(true, Ordering::Relaxed);
    }
}