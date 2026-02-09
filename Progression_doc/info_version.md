# EdgeMind Runtime (v0.1.0 - "The Memory Manager")

**Current Status:** Alpha / Prototype
**Focus:** User-Space Virtual Memory Management

## 1. What is Implemented
We have built a custom runtime environment capable of enforcing strict memory limits on simulated workloads.

### Core Modules (`crates/runtime`)
* **Memory Arena (`arena.rs`):** * A custom linear allocator using `unsafe` pointer arithmetic.
    * **Capability:** Allocates a fixed contiguous block of RAM from the OS at startup.
    * **Constraint:** Enforces a hard `capacity` limit. Returns `ArenaError::OutOfMemory` if exceeded.
    * **Physical Commitment:** Forces OS page allocation via `write_bytes` (memset) to ensure limits are tested against real hardware RAM.
    * **Fast Reset:** Supports O(1) memory freeing by resetting the offset pointer (Logic: "Drop Everything").

* **Scheduler (`manager.rs`):**
    * Acts as the "Kernel" for neural network layers.
    * **Logic:** manages a list of `loaded_layers`.
    * **Policy:** 1.  Tries to allocate layer in Arena.
        2.  If full, triggers `MustUnload` decision.
        3.  If layer > total capacity, triggers `OOM`.

* **System Monitor (`monitor.rs`):**
    * Background thread using `sysinfo`.
    * **Capability:** Polls OS-level process metrics (RSS Memory, CPU) independent of the runtime's internal counters.

### Workload (`crates/models`)
* **DeepFeedForward:** A standard 10-layer neural network definition using the `Burn` framework (currently used for size estimation).

## 2. Architecture Constraints
* **No Global Allocator:** The system currently manages *only* the memory explicitly requested through the Arena. Overhead (stack, standard library) is not yet capped.
* **Single Tenant:** The Arena is not thread-safe for concurrent allocations (Single Producer).
* **Eviction Policy:** "Flush All" (The simplest valid strategy).

## 3. Verified Behaviors
* [x] Runtime boots with a specific memory cap (e.g., 50MB).
* [x] Scheduler successfully loads layers until capacity is hit.
* [x] Scheduler correctly identifies "Memory Full" and evicts old data.
* [x] OS Monitor confirms physical RAM usage aligns with Arena allocations.