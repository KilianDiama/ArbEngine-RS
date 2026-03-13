# 🧮 High-Performance SIMD Engine in Rust

This project is a **production-ready, high-performance SIMD engine** written in Rust.  
It provides aligned memory buffers, parallel SIMD processing, and a flexible strategy pattern, making it ideal for HPC workloads such as financial analytics, physics simulations, or signal processing.

---

## Features

- ✅ Aligned memory buffers for SIMD (`AlignedBuffer<N>`)  
- ✅ Strategy-based SIMD processing (`SIMDStrategy<N>` trait)  
- ✅ Parallel execution using Rayon (`ArbEngine`)  
- ✅ Handles padding and last-vector masks safely  
- ✅ Easily extensible for any SIMD-based algorithm  
- ✅ Example: volatility detection using absolute threshold  

---

## Installation

1. Install Rust (if not already):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
