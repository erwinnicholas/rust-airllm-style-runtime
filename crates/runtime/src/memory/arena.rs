use std::alloc::{Layout, alloc, dealloc};
use std::ptr::NonNull;

/// A specialized memory allocator for Model Tensors.
/// It mimics a "Linear Allocator" - incredibly fast, but must be reset often.
pub struct ModelArena {
    memory_start: NonNull<u8>,
    capacity: usize,
    offset: usize, // Current pointer to free memory
}

#[derive(Debug)]
pub enum ArenaError {
    OutOfMemory { requested: usize, available: usize },
    AllocationFailed,
}

impl ModelArena {
    /// Request a large chunk of RAM from the OS once (at startup).
    pub fn new(capacity_mb: usize) -> Result<Self, ArenaError> {
        let capacity_bytes = capacity_mb * 1024 * 1024;
        
        // We use unsafe Rust to allocate raw memory with specific alignment
        // This is pure Systems Engineering.
        let layout = Layout::from_size_align(capacity_bytes, 32)
            .map_err(|_| ArenaError::AllocationFailed)?;
            
        let ptr = unsafe { alloc(layout) };
        let memory_start = NonNull::new(ptr).ok_or(ArenaError::AllocationFailed)?;

        Ok(Self {
            memory_start,
            capacity: capacity_bytes,
            offset: 0,
        })
    }

    /// The "malloc" replacement.
    /// Returns a pointer to the start of the valid memory block.
    pub fn alloc(&mut self, size: usize) -> Result<*mut u8, ArenaError> {
        if self.offset + size > self.capacity {
            return Err(ArenaError::OutOfMemory { 
                requested: size, 
                available: self.capacity - self.offset 
            });
        }

        let ptr = unsafe { 
            self.memory_start.as_ptr().add(self.offset) 
        };

        // --- ADD THIS BLOCK ---
        // Force the OS to allocate physical RAM by writing to it.
        // In a real scenario, this happens when we load weights from disk.
        unsafe {
            std::ptr::write_bytes(ptr, 0, size);
        }
        // ----------------------

        self.offset += size;
        Ok(ptr)
    }

    /// Reset the arena. We don't "free" individual objects.
    /// We just move the pointer back to 0. (Extremely fast).
    pub fn reset(&mut self) {
        self.offset = 0;
    }
    
    /// Metrics: How much memory is currently used?
    pub fn used_bytes(&self) -> usize {
        self.offset
    }
}

// Safety: We must clean up the raw memory when the program exits
impl Drop for ModelArena {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.capacity, 32).unwrap();
        unsafe {
            dealloc(self.memory_start.as_ptr(), layout);
        }
    }
}
// We need to tell Rust this is safe to send across threads
unsafe impl Send for ModelArena {}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_initialization() {
        // Guard Rail: Can we allocate 1MB?
        let arena = ModelArena::new(1).expect("Failed to create arena");
        assert_eq!(arena.capacity, 1024 * 1024);
        assert_eq!(arena.used_bytes(), 0);
    }

    #[test]
    fn test_allocation_success() {
        let mut arena = ModelArena::new(10).unwrap(); // 10MB
        
        // Allocate 1KB
        let ptr = arena.alloc(1024).expect("Should fit");
        assert!(!ptr.is_null());
        assert_eq!(arena.used_bytes(), 1024);
        
        // Allocate another 1KB
        let ptr2 = arena.alloc(1024).expect("Should fit");
        
        // Guard Rail: Pointers must not overlap
        // The distance between pointers should be exactly 1024 bytes
        unsafe {
            assert_eq!(ptr.add(1024), ptr2);
        }
    }

    #[test]
    fn test_out_of_memory() {
        let mut arena = ModelArena::new(1).unwrap(); // 1MB Total
        
        // Allocate 0.6MB (Success)
        let _ = arena.alloc(600 * 1024).unwrap();
        
        // Allocate 0.5MB (Fail: 0.6 + 0.5 > 1.0)
        let result = arena.alloc(500 * 1024);
        
        match result {
            Err(ArenaError::OutOfMemory { requested, available }) => {
                assert_eq!(requested, 500 * 1024);
                // Available should be 1MB - 600KB = 424KB (roughly)
                assert!(available < 500 * 1024);
            }
            _ => panic!("Should have failed with OOM"),
        }
    }

    #[test]
    fn test_reset_behavior() {
        let mut arena = ModelArena::new(1).unwrap();
        let _ = arena.alloc(1024).unwrap();
        
        // Reset
        arena.reset();
        assert_eq!(arena.used_bytes(), 0);
        
        // Should be able to allocate again at the start
        let _ = arena.alloc(1024 * 1024).unwrap(); // Fill entire arena
    }
}