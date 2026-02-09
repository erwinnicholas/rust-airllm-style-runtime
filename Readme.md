markdown
# EdgeMind Runtime — V1 (AirLLM-style Layer Streaming in Rust)

This repository contains **EdgeMind Runtime V1**, a small Rust library and demo that implements an AirLLM-style runtime for **layer-by-layer model loading with strict, model-only memory limits**.

The goal of V1 is simple and narrow:

> Run a neural network under a fixed memory budget for **model weights and activations only**, and fail or degrade gracefully instead of letting the OS kill the process.

This project does **not** aim to beat existing ML frameworks on accuracy or speed. The focus is **predictable memory usage and controlled execution**.

---

## Scope (V1)

### What this project does
- Loads model layers **on demand** instead of keeping the full model in memory.
- Enforces a **hard memory limit for model allocations only** (weights + activations).
- Fails safely when the model exceeds its memory budget (no OS OOM kills).
- Exposes metrics for:
  - model memory usage
  - layer load/unload time
  - inference latency
  - early-exit / rejection count
- Provides a small demo model and CLI to validate behavior.

### What this project does not do (V1)
- No GPU memory management.
- No thermal or system-wide memory control.
- No production object detection models.
- No kernel changes or privileged operations.
- No automatic import of arbitrary PyTorch models (toy model only).

---

## Design Overview

The runtime is split into three parts:



runtime/
├── allocator/     # Model-only memory arena (hard limit)
├── model/         # Layer graph and weight paging
├── scheduler/     # Simple policy for layer residency / early-exit
examples/
└── toy_model/     # Small model + CLI demo
bench/
└── stress/        # Memory pressure tests
docs/
└── design.md     # Detailed design notes

`

### Model-only Memory Control

All model tensors (weights + activations) must be allocated from a **ModelArena**:

- The arena is created with a fixed capacity (e.g., 50 MB).
- If a layer cannot be loaded into the arena, the runtime:
  - returns `MemoryQuotaExceeded`, or
  - triggers an early-exit (configurable policy).

Frame buffers, logging, and non-model memory use the normal system allocator and are **not** counted toward the model budget.

### Layer Streaming

Each model layer is represented as a `LayerWeights` object:

- Weights are stored on disk.
- During inference:
  1. Load next layer into the arena.
  2. Run forward pass for that layer.
  3. Unload previous layer to free arena memory.
- At most `N` layers are resident at once (configurable).

This ensures peak model memory usage stays within the configured limit.

---

## Configuration

Example policy (`policy.yaml`):

yaml
model:
  arena_capacity_mb: 50
  keep_resident_layers: 1

scheduler:
  on_memory_quota_exceeded: early_exit   # options: early_exit | reject_request

metrics:
  enable_http: true
  http_addr: "127.0.0.1:9090"
`

---

## Build & Run

### Requirements

* Rust (stable)
* Linux (for demo; no kernel features required)

### Build

bash
cargo build --release


### Run the demo

bash
EDGE_MODEL_ARENA_MB=50 cargo run --example toy_model


You should see logs showing:

* layer loads/unloads
* current model arena usage
* early-exit or rejection when memory is insufficient

---

## Metrics

The runtime exports:

* `model_arena_usage_bytes`
* `model_arena_capacity_bytes`
* `layer_load_time_ms`
* `inference_latency_ms`
* `quota_exceeded_count`
* `exit_level`

Metrics are printed to stdout and can be exposed via HTTP if enabled.

---

## Testing

Run all tests:

bash
cargo test


Key tests:

* Arena allocation and free accounting
* Layer load/unload sequence
* Quota exceeded behavior
* No global OOM on forced memory pressure

---

## Roadmap

### V2 — Orchestrator

* Frame capture and request queue
* Backpressure when model arena is saturated

### V3 — Thermal and System Memory Control

* Read CPU temperature and system memory
* Adjust scheduler policy based on telemetry

### V4 — Production Hardening

* Cross-compilation (ARM)
* Multiple models
* Stable storage format for weights
* Containerized deployment
* Optional unikernel demo

---

## Failure Model

* If the model exceeds its memory budget, the runtime **does not crash**.
* The runtime either:

  * exits early with partial output, or
  * rejects the request.
* The OS OOM killer should never be triggered by model allocations in V1.

---

## License

MIT or Apache-2.0 (choose one before first release).

---

## Notes

This project is intentionally minimal in V1.
The goal is to prove **bounded model memory and controlled execution** before adding system-level policies or production features.



root dir/
├── Cargo.lock
├── Cargo.toml              # WORKSPACE ROOT: Defines members [crates/*]
├── config/
│   └── system_limits.toml  # CONFIG: Defines "Artificial RAM Limit" (e.g., 512MB)
│
├── crates/
│   ├── common/             # SHARED: Types used by both Runtime and Model
│   │   ├── Cargo.toml
│   │   └── src/lib.rs      # Defines "TensorId", "MemoryStats", "SystemError"
│   │
│   ├── runtime/            # CORE SYSTEM: The "OS" logic
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── memory/     # THE ARENA: Custom allocator logic
│   │       │   └── arena.rs
│   │       └── scheduler/  # THE BRAIN: Decides which layer runs when
│   │           └── policy.rs
│   │
│   └── models/             # USER SPACE: The Neural Net (Burn Code)
│       ├── Cargo.toml
│       └── src/
│           ├── lib.rs
│           └── feed_forward.rs # The 10-layer FFN definition
│
└── examples/
    └── run_inference.rs    # ENTRY POINT: Boots the Runtime + Loads the Model