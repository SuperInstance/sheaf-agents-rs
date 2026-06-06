# Architecture — sheaf-agents-rs

> *Internal design and data flow.*

## Overview

This crate provides functionality for the SuperInstance ecosystem.

## Core Types

- **`SheafEdge`**
- **`CellularSheaf`**
- **`Cohomology`**
- **`AgentState`**
- **`AgentNetwork`**
- **`ConvergenceResult`**
- **`ConsensusQuality`**

## Key Functions

- `new()`
- `h0()`
- `h1()`
- `sheaf_laplacian()`
- `spectral_gap()`
- `cohomology_basis()`
- `add_edge()`
- `total_vertex_dim()`

## Source Structure

1 Rust source file(s) in `src/`.
Language: Rust

## Cross-Repo References

- [ternary-core](https://github.com/SuperInstance/ternary-core) — shared Z₃ traits
- [ternary-types](https://github.com/SuperInstance/ternary-types) — type-level encodings
- [Full SuperInstance fleet](https://github.com/orgs/SuperInstance/repositories?q=ternary)
