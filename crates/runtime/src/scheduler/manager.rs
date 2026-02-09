use crate::memory::arena::{ModelArena, ArenaError};

/// The "Kernel" that manages neural network execution.
pub struct Scheduler {
    arena: ModelArena,
    // We track "virtual" layers to know what is currently loaded
    loaded_layers: Vec<usize>, 
}

#[derive(Debug)]
pub enum SchedulerDecision {
    LoadSuccess { ptr: *mut u8 },
    /// The system is full, but we can make space by unloading old layers.
    MustUnload { layer_id: usize },
    /// Critical failure: Even if we unload everything, this layer is too big.
    OOM,
}

impl Scheduler {
    /// Boot the scheduler with a hard RAM limit (e.g., 128MB)
    pub fn boot(memory_limit_mb: usize) -> Result<Self, ArenaError> {
        Ok(Self {
            arena: ModelArena::new(memory_limit_mb)?,
            loaded_layers: Vec::new(),
        })
    }

    /// The core logic: "I want to load Layer X. Can I?"
    pub fn request_load(&mut self, layer_id: usize, size_bytes: usize) -> SchedulerDecision {
        println!("[Scheduler] Request: Load Layer {} ({} MB)", layer_id, size_bytes / 1024 / 1024);

        // 1. Try to allocate directly
        match self.arena.alloc(size_bytes) {
            Ok(ptr) => {
                self.loaded_layers.push(layer_id);
                SchedulerDecision::LoadSuccess { ptr }
            }
            Err(_) => {
                // 2. If full, check if we can unload something
                if let Some(&old_layer) = self.loaded_layers.first() {
                    println!("[Scheduler] Memory Full. Suggesting eviction of Layer {}", old_layer);
                    SchedulerDecision::MustUnload { layer_id: old_layer }
                } else {
                    // 3. If nothing to unload, we are truly OOM
                    SchedulerDecision::OOM
                }
            }
        }
    }

    /// Free up memory (conceptually unloads a layer)
    pub fn unload_all(&mut self) {
        println!("[Scheduler] Resetting Arena (Unloading all layers)...");
        self.arena.reset();
        self.loaded_layers.clear();
    }

    pub fn memory_usage(&self) -> usize {
        self.arena.used_bytes()
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_eviction_policy() {
        // Boot system with 100 bytes (Tiny!)
        // Note: ModelArena usually aligns to 32 bytes, so we use small valid numbers
        let mut scheduler = Scheduler::boot(1).unwrap(); // 1MB
        let mb = 1024 * 1024;

        // 1. Load 0.6MB (Fits)
        match scheduler.request_load(1, (0.6 * mb as f64) as usize) {
            SchedulerDecision::LoadSuccess { .. } => assert!(true),
            _ => panic!("First load should succeed"),
        }

        // 2. Load 0.5MB (Should trigger Eviction, NOT OOM)
        // Because 0.5MB fits in 1MB *if* we empty it.
        match scheduler.request_load(2, (0.5 * mb as f64) as usize) {
            SchedulerDecision::MustUnload { layer_id } => {
                assert_eq!(layer_id, 1); // Should suggest evicting Layer 1
            },
            _ => panic!("Should ask to unload"),
        }
    }

    #[test]
    fn test_scheduler_hard_oom() {
        let mut scheduler = Scheduler::boot(1).unwrap(); // 1MB
        let mb = 1024 * 1024;

        // Try to load 2MB into a 1MB container
        match scheduler.request_load(1, 2 * mb) {
            SchedulerDecision::OOM => assert!(true),
            _ => panic!("Should be impossible to load"),
        }
    }
}