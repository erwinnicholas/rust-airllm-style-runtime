use runtime::scheduler::manager::{Scheduler, SchedulerDecision};
use runtime::monitor::SystemMonitor;
use std::thread;
use std::time::Duration;

fn main() {
    println!("--- Booting EdgeMind Runtime ---");

    // 1. Start the Background Monitor (Polls every 500ms)
    let monitor = SystemMonitor::start(100); 

    // 2. Initialize System (50MB Limit)
    let mut system = Scheduler::boot(50).expect("Failed to boot system");

    // 3. Define the workload (Same as before)
    // We add a `sleep` here so you have time to see the monitor update!
    let layers = vec![
        ("Layer_01", 15 * 1024 * 1024),
        ("Layer_02", 15 * 1024 * 1024),
        ("Layer_03", 15 * 1024 * 1024),
        ("Layer_04", 15 * 1024 * 1024), // This one triggers eviction
        ("Layer_05", 15 * 1024 * 1024),
    ];

    println!("\nStarting Inference Sequence...");
    
    for (_name, size) in layers {
        // Simulate "Processing Time" so the monitor can capture the spike
        thread::sleep(Duration::from_millis(600)); 

        loop {
            match system.request_load(0, size) {
                SchedulerDecision::LoadSuccess { ptr: _ } => {
                    // Note: We don't print here to avoid messing up the Monitor's \r output
                    // Just let the Monitor show the RAM going up!
                    break;
                }
                SchedulerDecision::MustUnload { layer_id: _ } => {
                    system.unload_all(); 
                    // Give the OS time to reclaim memory (if we were actually freeing)
                    thread::sleep(Duration::from_millis(200));
                }
                SchedulerDecision::OOM => panic!("System Crash!"),
            }
        }
    }

    // Stop monitor and exit
    monitor.stop();
    println!("\n--- Inference Complete ---");
}