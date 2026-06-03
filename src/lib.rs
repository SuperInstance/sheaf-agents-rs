//! # sheaf-agents
//!
//! Cellular sheaves on graphs for multi-agent consensus analysis.
//!
//! **H¹ > 0 means communication can't help.** Your agents disagree not because
//! of latency or bugs, but because the math makes agreement impossible.
//!
//! This library computes exactly *where* multi-agent systems structurally
//! cannot agree, using sheaf cohomology on graphs.

use nalgebra::{DMatrix, DVector};
use std::fmt;

// ── Cellular Sheaf ──────────────────────────────────────────────────────────

/// A directed edge in the sheaf's underlying graph.
///
/// Each edge carries two restriction maps: `r1` maps stalk at `v1` into the
/// edge space, and `r2` maps stalk at `v2` into the edge space.
#[derive(Clone)]
pub struct SheafEdge {
    pub v1: usize,
    pub v2: usize,
    /// Restriction map from stalk at `v1` → edge space.
    pub r1: DMatrix<f64>,
    /// Restriction map from stalk at `v2` → edge space.
    pub r2: DMatrix<f64>,
}

/// A cellular sheaf on a graph.
///
/// Assigns a vector space (stalk) to each vertex and linear restriction maps
/// to each edge. The cohomology of this sheaf tells you whether global
/// agreement is possible.
#[derive(Clone)]
pub struct CellularSheaf {
    /// Dimension of the stalk at each vertex.
    pub stalk_dims: Vec<usize>,
    pub edges: Vec<SheafEdge>,
}

impl CellularSheaf {
    /// Create a sheaf with the given stalk dimensions and capacity for edges.
    pub fn new(stalk_dims: Vec<usize>) -> Self {
        Self {
            stalk_dims,
            edges: Vec::new(),
        }
    }

    /// Dimension of H⁰ (global sections).
    pub fn h0(&self, tol: f64) -> usize {
        self.cohomology(tol).h0_dim
    }

    /// Dimension of H¹ (obstructions to agreement).
    pub fn h1(&self, tol: f64) -> usize {
        self.cohomology(tol).h1_dim
    }

    /// Alias for [`CellularSheaf::laplacian`]; returns the sheaf Laplacian.
    pub fn sheaf_laplacian(&self) -> DMatrix<f64> {
        self.laplacian()
    }

    /// Compute the spectral gap (λ₂ - λ₁) of the sheaf Laplacian,
    /// where λ₁ ≤ λ₂ ≤ … are the eigenvalues of L.
    ///
    /// The spectral gap measures how quickly the disagreement diffusion
    /// converges. A larger gap means faster consensus.
    pub fn spectral_gap(&self) -> f64 {
        let l = self.laplacian();
        if l.nrows() < 2 {
            return 0.0;
        }
        let eigen = l.symmetric_eigenvalues();
        let mut vals: Vec<f64> = eigen.iter().cloned().collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // Find the first eigenvalue that is numerically > 0
        vals.into_iter().find(|&v| v > 1e-12).unwrap_or(0.0)
    }

    /// Return a basis for Hᵈ as a `Vec<Vec<f64>>`.
    ///
    /// Each inner Vec is one basis vector (flattened vertex stalk or edge cochain).
    /// Returns an empty Vec if the cohomology is zero in that dimension
    /// or if `dim` is not 0 or 1.
    pub fn cohomology_basis(&self, dim: usize, tol: f64) -> Vec<Vec<f64>> {
        let coh = self.cohomology(tol);
        let (nrows, ncols, mat) = match dim {
            0 => (coh.h0_basis.nrows(), coh.h0_basis.ncols(), &coh.h0_basis),
            1 => (coh.h1_basis.nrows(), coh.h1_basis.ncols(), &coh.h1_basis),
            _ => return vec![],
        };
        if ncols == 0 {
            return vec![];
        }
        (0..ncols)
            .map(|col| {
                (0..nrows).map(|row| mat[(row, col)]).collect()
            })
            .collect()
    }

    /// Add an edge with restriction maps `r1` (from `v1`) and `r2` (from `v2`).
    pub fn add_edge(
        &mut self,
        v1: usize,
        v2: usize,
        r1: DMatrix<f64>,
        r2: DMatrix<f64>,
    ) {
        self.edges.push(SheafEdge { v1, v2, r1, r2 });
    }

    /// Total dimension of the vertex stalk space.
    pub fn total_vertex_dim(&self) -> usize {
        self.stalk_dims.iter().sum()
    }

    /// Total dimension of the edge space.
    pub fn total_edge_dim(&self) -> usize {
        self.edges.iter().map(|e| e.r1.nrows()).sum()
    }

    /// Offset (in the flattened vertex vector) for vertex `v`.
    pub fn vertex_offset(&self, v: usize) -> usize {
        self.stalk_dims[..v].iter().sum()
    }
}

impl fmt::Debug for CellularSheaf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CellularSheaf")
            .field("stalk_dims", &self.stalk_dims)
            .field("n_edges", &self.edges.len())
            .finish()
    }
}

// ── Coboundary Operator ─────────────────────────────────────────────────────

/// Build the coboundary operator d₀: (total_edge_dim × total_vertex_dim).
///
/// For each edge e=(v1,v2): (d₀ s)_e = r1(s_{v1}) − r2(s_{v2}).
fn build_coboundary(sheaf: &CellularSheaf) -> DMatrix<f64> {
    let n = sheaf.total_vertex_dim();
    let m = sheaf.total_edge_dim();
    let mut d0 = DMatrix::zeros(m, n);

    let mut row_off = 0;
    for edge in &sheaf.edges {
        let de = edge.r1.nrows();
        let off1 = sheaf.vertex_offset(edge.v1);
        let off2 = sheaf.vertex_offset(edge.v2);

        // +r1 block
        for i in 0..de {
            for j in 0..edge.r1.ncols() {
                d0[(row_off + i, off1 + j)] += edge.r1[(i, j)];
            }
        }
        // -r2 block
        for i in 0..de {
            for j in 0..edge.r2.ncols() {
                d0[(row_off + i, off2 + j)] -= edge.r2[(i, j)];
            }
        }
        row_off += de;
    }
    d0
}

// ── Null Space via RREF ─────────────────────────────────────────────────────

/// Compute rank and null-space basis of an m×n matrix.
///
/// Returns `(rank, null_basis)` where `null_basis` has shape `(n, nullity)`.
/// Each column is a null-space basis vector.
fn null_space(mat: &DMatrix<f64>, tol: f64) -> (usize, DMatrix<f64>) {
    let m = mat.nrows();
    let n = mat.ncols();

    if m == 0 || n == 0 {
        return (0, DMatrix::zeros(0, 0));
    }

    // Work on a mutable copy
    let mut w = mat.clone();
    let mut pivot_col: Vec<Option<usize>> = vec![None; m];
    let mut is_pivot = vec![false; n];

    let mut rank = 0;
    for col in 0..n {
        if rank >= m {
            break;
        }
        // Find pivot
        let mut max_row = None;
        let mut max_val = tol;
        for row in rank..m {
            let val = w[(row, col)].abs();
            if val > max_val {
                max_val = val;
                max_row = Some(row);
            }
        }
        let Some(prow) = max_row else { continue };

        // Swap rows
        if prow != rank {
            for j in 0..n {
                let tmp = w[(rank, j)];
                w[(rank, j)] = w[(prow, j)];
                w[(prow, j)] = tmp;
            }
        }

        pivot_col[rank] = Some(col);
        is_pivot[col] = true;

        let piv = w[(rank, col)];
        // Eliminate below
        for row in (rank + 1)..m {
            let factor = w[(row, col)] / piv;
            w[(row, col)] = 0.0;
            for j in (col + 1)..n {
                w[(row, j)] -= factor * w[(rank, j)];
            }
        }
        rank += 1;
    }

    // Back-substitute to RREF
    for i in (0..rank).rev() {
        let pc = pivot_col[i].unwrap();
        let piv = w[(i, pc)];
        for j in pc..n {
            w[(i, j)] /= piv;
        }
        for row in 0..i {
            let factor = w[(row, pc)];
            for j in pc..n {
                w[(row, j)] -= factor * w[(i, j)];
            }
        }
    }

    let nullity = n - rank;
    if nullity == 0 {
        return (rank, DMatrix::zeros(n, 0));
    }

    let mut basis = DMatrix::zeros(n, nullity);
    let mut free_idx = 0;
    for j in 0..n {
        if !is_pivot[j] {
            for i in 0..rank {
                let pc = pivot_col[i].unwrap();
                basis[(pc, free_idx)] = -w[(i, j)];
            }
            basis[(j, free_idx)] = 1.0;
            free_idx += 1;
        }
    }

    (rank, basis)
}

// ── Cohomology ──────────────────────────────────────────────────────────────

/// Sheaf cohomology: H⁰ = ker(d₀), H¹ = coker(d₀).
///
/// H⁰ captures global sections (agreements). H¹ captures structural
/// obstructions to agreement — the dimensions of disagreement that no amount
/// of communication can resolve.
#[derive(Debug)]
pub struct Cohomology {
    /// Dimension of H⁰ (global sections / agreements).
    pub h0_dim: usize,
    /// Dimension of H¹ (obstructions / structural disagreements).
    pub h1_dim: usize,
    /// Basis for H⁰: columns are global section vectors.
    pub h0_basis: DMatrix<f64>,
    /// Basis for H¹: columns are obstruction vectors in the edge cochain space.
    pub h1_basis: DMatrix<f64>,
}

impl CellularSheaf {
    /// Compute sheaf cohomology.
    ///
    /// - H⁰ = ker(d₀): the space of global sections (perfect agreement).
    /// - H¹ = coker(d₀) = ker(d₀ᵀ): obstructions in the edge space that
    ///   are orthogonal to im(d₀). These are disagreements with no upstream cause.
    pub fn cohomology(&self, tol: f64) -> Cohomology {
        let total_v = self.total_vertex_dim();
        let total_e = self.total_edge_dim();

        // Trivial sheaf
        if total_v == 0 && total_e == 0 {
            return Cohomology {
                h0_dim: 0,
                h1_dim: 0,
                h0_basis: DMatrix::zeros(0, 0),
                h1_basis: DMatrix::zeros(0, 0),
            };
        }

        // No edges: H⁰ = full vertex space, H¹ = 0
        if self.edges.is_empty() {
            return Cohomology {
                h0_dim: total_v,
                h1_dim: 0,
                h0_basis: DMatrix::identity(total_v, total_v),
                h1_basis: DMatrix::zeros(0, 0),
            };
        }

        let d0 = build_coboundary(self);

        // H⁰ = ker(d₀)
        let (rank_d0, h0_basis) = null_space(&d0, tol);
        let h0_dim = total_v - rank_d0;

        // H¹ = coker(d₀) = ker(d₀ᵀ)
        // Elements of ker(d₀ᵀ) are 1-cochains orthogonal to im(d₀).
        let d0t = d0.transpose();
        let (_rank_d0t, h1_basis) = null_space(&d0t, tol);
        let h1_dim = total_e - _rank_d0t;

        Cohomology {
            h0_dim,
            h1_dim,
            h0_basis,
            h1_basis,
        }
    }
}

// ── Sheaf Laplacian ─────────────────────────────────────────────────────────

impl CellularSheaf {
    /// Compute the sheaf Laplacian L = d₀ᵀ d₀.
    ///
    /// Generalizes the graph Laplacian. Positive semidefinite.
    /// Kernel = H⁰ (global sections).
    pub fn laplacian(&self) -> DMatrix<f64> {
        let n = self.total_vertex_dim();
        let mut l = DMatrix::zeros(n, n);

        for edge in &self.edges {
            let d1 = self.stalk_dims[edge.v1];
            let d2 = self.stalk_dims[edge.v2];
            let off1 = self.vertex_offset(edge.v1);
            let off2 = self.vertex_offset(edge.v2);

            let r1t_r1 = &edge.r1.transpose() * &edge.r1;
            let r2t_r2 = &edge.r2.transpose() * &edge.r2;
            let r1t_r2 = &edge.r1.transpose() * &edge.r2;
            let r2t_r1 = &edge.r2.transpose() * &edge.r1;

            // L[v1, v1] += r1ᵀ r1
            for i in 0..d1 {
                for j in 0..d1 {
                    l[(off1 + i, off1 + j)] += r1t_r1[(i, j)];
                }
            }
            // L[v2, v2] += r2ᵀ r2
            for i in 0..d2 {
                for j in 0..d2 {
                    l[(off2 + i, off2 + j)] += r2t_r2[(i, j)];
                }
            }
            // L[v1, v2] -= r1ᵀ r2
            for i in 0..d1 {
                for j in 0..d2 {
                    l[(off1 + i, off2 + j)] -= r1t_r2[(i, j)];
                }
            }
            // L[v2, v1] -= r2ᵀ r1
            for i in 0..d2 {
                for j in 0..d1 {
                    l[(off2 + i, off1 + j)] -= r2t_r1[(i, j)];
                }
            }
        }

        l
    }
}

// ── Agent Network ───────────────────────────────────────────────────────────

/// An agent sitting on a vertex with a local belief vector.
#[derive(Clone, Debug)]
pub struct AgentState {
    pub vertex: usize,
    pub belief: DVector<f64>,
}

/// A network of agents, one per vertex, synchronized via a shared sheaf.
#[derive(Clone, Debug)]
pub struct AgentNetwork {
    pub sheaf: CellularSheaf,
    pub agents: Vec<AgentState>,
}

impl AgentNetwork {
    /// Create a network with one agent per vertex, zero-initialized beliefs.
    pub fn new(sheaf: &CellularSheaf) -> Self {
        let agents = sheaf
            .stalk_dims
            .iter()
            .enumerate()
            .map(|(v, &d)| AgentState {
                vertex: v,
                belief: DVector::zeros(d),
            })
            .collect();

        Self {
            sheaf: sheaf.clone(),
            agents,
        }
    }

    /// Set an agent's belief vector.
    pub fn set_belief(&mut self, agent: usize, belief: DVector<f64>) {
        self.agents[agent].belief = belief;
    }

    /// Measure total disagreement across all edges.
    ///
    /// For each edge, computes ‖r1(x₁) − r2(x₂)‖₂ and sums them up.
    pub fn disagreement(&self) -> f64 {
        let mut total = 0.0;
        for edge in &self.sheaf.edges {
            let x1 = &self.agents[edge.v1].belief;
            let x2 = &self.agents[edge.v2].belief;
            let diff = &edge.r1 * x1 - &edge.r2 * x2;
            total += diff.norm();
        }
        total
    }

    /// One gradient descent step on the disagreement energy.
    ///
    /// Returns the combined belief vector and whether the gradient vanished.
    pub fn synchronize(&mut self, step: f64) -> (DVector<f64>, bool) {
        let total_dim = self.sheaf.total_vertex_dim();
        let mut gradient: DVector<f64> = DVector::zeros(total_dim);

        for edge in &self.sheaf.edges {
            let v1 = edge.v1;
            let v2 = edge.v2;
            let off1 = self.sheaf.vertex_offset(v1);
            let off2 = self.sheaf.vertex_offset(v2);

            let x1 = &self.agents[v1].belief;
            let x2 = &self.agents[v2].belief;
            let diff = &edge.r1 * x1 - &edge.r2 * x2;

            let g1 = &edge.r1.transpose() * &diff;
            let g2 = &edge.r2.transpose() * &diff;

            for i in 0..g1.len() {
                gradient[off1 + i] += g1[i];
            }
            for i in 0..g2.len() {
                gradient[off2 + i] -= g2[i];
            }
        }

        let grad_norm = gradient.norm();
        let converged = grad_norm < 1e-10;

        // Apply update
        let mut off = 0;
        for v in 0..self.agents.len() {
            let d = self.sheaf.stalk_dims[v];
            for i in 0..d {
                self.agents[v].belief[i] -= step * gradient[off + i];
            }
            off += d;
        }

        // Collect combined belief
        let mut combined: DVector<f64> = DVector::zeros(total_dim);
        let mut off = 0;
        for v in 0..self.agents.len() {
            let d = self.sheaf.stalk_dims[v];
            for i in 0..d {
                combined[off + i] = self.agents[v].belief[i];
            }
            off += d;
        }

        (combined, converged)
    }

    /// Run synchronization until convergence or max iterations.
    pub fn converge(
        &mut self,
        step: f64,
        max_iter: usize,
        tol: f64,
    ) -> ConvergenceResult {
        let has_obstruction = self.sheaf.cohomology(tol).h1_dim > 0;

        for i in 0..max_iter {
            let dis = self.disagreement();
            if dis < tol {
                return ConvergenceResult {
                    iterations: i + 1,
                    final_disagreement: dis,
                    converged: true,
                    obstruction_detected: false,
                };
            }

            let (_, conv) = self.synchronize(step);
            if conv {
                return ConvergenceResult {
                    iterations: i + 1,
                    final_disagreement: self.disagreement(),
                    converged: true,
                    obstruction_detected: false,
                };
            }
        }

        ConvergenceResult {
            iterations: max_iter,
            final_disagreement: self.disagreement(),
            converged: false,
            obstruction_detected: has_obstruction,
        }
    }
}

/// Result of running the convergence loop.
#[derive(Clone, Debug)]
pub struct ConvergenceResult {
    pub iterations: usize,
    pub final_disagreement: f64,
    pub converged: bool,
    pub obstruction_detected: bool,
}

// ── Consensus ───────────────────────────────────────────────────────────────

/// Can agents on this sheaf ever reach agreement? True iff H¹ = 0.
pub fn can_agree(sheaf: &CellularSheaf, tol: f64) -> bool {
    sheaf.cohomology(tol).h1_dim == 0
}

/// Compute a basis for the forced disagreements (H¹ basis).
///
/// Returns a matrix whose columns are obstruction vectors in the edge cochain
/// space. Empty matrix if H¹ = 0.
pub fn forced_disagreement(sheaf: &CellularSheaf, tol: f64) -> DMatrix<f64> {
    sheaf.cohomology(tol).h1_basis
}

/// Quality metrics for the current consensus state.
#[derive(Clone, Debug)]
pub struct ConsensusQuality {
    pub agreement_score: f64,
    pub h0_quality: f64,
    pub h1_obstruction: f64,
    pub h0_dim: usize,
    pub h1_dim: usize,
}

impl AgentNetwork {
    /// Compute consensus quality metrics.
    pub fn quality(&self, tol: f64) -> ConsensusQuality {
        let coh = self.sheaf.cohomology(tol);
        let dis = self.disagreement();
        let total_dim = self.sheaf.total_vertex_dim();

        ConsensusQuality {
            agreement_score: 1.0 - dis / (dis + 1.0),
            h0_quality: if total_dim > 0 {
                coh.h0_dim as f64 / total_dim as f64
            } else {
                1.0
            },
            h1_obstruction: coh.h1_dim as f64,
            h0_dim: coh.h0_dim,
            h1_dim: coh.h1_dim,
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Build a 2×2 rotation-90° matrix: [[0, 1], [-1, 0]].
pub fn rot90() -> DMatrix<f64> {
    DMatrix::from_row_slice(2, 2, &[0.0, 1.0, -1.0, 0.0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    const TOL: f64 = 1e-8;

    fn id2() -> DMatrix<f64> {
        DMatrix::identity(2, 2)
    }

    fn id(n: usize) -> DMatrix<f64> {
        DMatrix::identity(n, n)
    }

    // ── Matrix / linear algebra ──

    #[test]
    fn test_null_space_identity() {
        let m = DMatrix::identity(3, 3);
        let (rank, basis) = null_space(&m, TOL);
        assert_eq!(rank, 3);
        assert_eq!(basis.ncols(), 0);
    }

    #[test]
    fn test_null_space_zero_matrix() {
        let m = DMatrix::zeros(2, 3);
        let (rank, basis) = null_space(&m, TOL);
        assert_eq!(rank, 0);
        assert_eq!(basis.ncols(), 3);
    }

    #[test]
    fn test_null_space_rank_1() {
        // [1 2 3; 2 4 6] → rank 1, nullity 2
        let m = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 2.0, 4.0, 6.0]);
        let (rank, basis) = null_space(&m, TOL);
        assert_eq!(rank, 1);
        assert_eq!(basis.ncols(), 2);
        // Verify m * basis ≈ 0
        let product = &m * &basis;
        for i in 0..product.nrows() {
            for j in 0..product.ncols() {
                assert!(
                    product[(i, j)].abs() < 1e-10,
                    "Null-space vector not in kernel at ({},{}): {}",
                    i,
                    j,
                    product[(i, j)]
                );
            }
        }
    }

    // ── Sheaf construction ──

    #[test]
    fn test_empty_sheaf() {
        let s = CellularSheaf::new(vec![]);
        let c = s.cohomology(TOL);
        assert_eq!(c.h0_dim, 0);
        assert_eq!(c.h1_dim, 0);
    }

    #[test]
    fn test_single_vertex() {
        let s = CellularSheaf::new(vec![3]);
        let c = s.cohomology(TOL);
        assert_eq!(c.h0_dim, 3, "single vertex: H0 = 3 (whole stalk)");
        assert_eq!(c.h1_dim, 0);
    }

    #[test]
    fn test_disconnected_two_vertices() {
        let s = CellularSheaf::new(vec![2, 2]);
        let c = s.cohomology(TOL);
        assert_eq!(c.h0_dim, 4, "disconnected: H0 = 2+2 = 4");
        assert_eq!(c.h1_dim, 0);
    }

    // ── Single edge ──

    #[test]
    fn test_single_edge_identity() {
        let mut s = CellularSheaf::new(vec![2, 2]);
        s.add_edge(0, 1, id2(), id2());

        let c = s.cohomology(TOL);
        assert_eq!(c.h0_dim, 2, "single edge identity: H0 = 2");
        assert_eq!(c.h1_dim, 0, "single edge identity: H1 = 0");
    }

    #[test]
    fn test_single_edge_mismatched_restrictions() {
        let mut s = CellularSheaf::new(vec![2, 2]);
        s.add_edge(0, 1, id2(), rot90());

        let c = s.cohomology(TOL);
        // Even with mismatched restrictions, single edge has H¹ = 0 (no cycle)
        assert_eq!(c.h0_dim, 2, "single edge mismatched: H0 = 2");
        assert_eq!(c.h1_dim, 0, "single edge mismatched: H1 = 0");
    }

    // ── Triangle ──

    #[test]
    fn test_triangle_identity() {
        let mut s = CellularSheaf::new(vec![2, 2, 2]);
        s.add_edge(0, 1, id2(), id2());
        s.add_edge(1, 2, id2(), id2());
        s.add_edge(0, 2, id2(), id2());

        let c = s.cohomology(TOL);
        assert_eq!(c.h0_dim, 2, "triangle identity: H0 = 2 (all agree)");
        assert_eq!(c.h1_dim, 2, "triangle identity: H1 = 2 (cycle × stalk_dim)");
    }

    #[test]
    fn test_triangle_orthogonal() {
        let mut s = CellularSheaf::new(vec![2, 2, 2]);
        s.add_edge(0, 1, id2(), rot90());
        s.add_edge(1, 2, id2(), rot90());
        s.add_edge(0, 2, id2(), id2());

        let c = s.cohomology(1e-6);
        assert_eq!(c.h0_dim, 0, "triangle orthogonal: H0 = 0 (no global sections)");
        assert_eq!(c.h1_dim, 0, "triangle orthogonal: H1 = 0 (surjective coboundary)");
    }

    // ── Star topology ──

    #[test]
    fn test_star_topology() {
        let mut s = CellularSheaf::new(vec![3, 1, 1, 1]);
        for i in 0..3 {
            let mut r_center = DMatrix::zeros(1, 3);
            r_center[(0, i)] = 1.0;
            s.add_edge(0, i + 1, r_center, id(1));
        }

        let c = s.cohomology(TOL);
        assert!(
            c.h0_dim >= 1,
            "star: H0 >= 1 (can agree on center's view)"
        );
    }

    // ── Path graph ──

    #[test]
    fn test_path_graph() {
        let mut s = CellularSheaf::new(vec![2, 2, 2, 2]);
        s.add_edge(0, 1, id2(), id2());
        s.add_edge(1, 2, id2(), id2());
        s.add_edge(2, 3, id2(), id2());

        let c = s.cohomology(TOL);
        assert_eq!(c.h0_dim, 2, "path graph: H0 = 2");
        assert_eq!(c.h1_dim, 0, "path graph: H1 = 0 (tree, no cycle)");
    }

    // ── Laplacian ──

    #[test]
    fn test_laplacian_symmetric() {
        let mut s = CellularSheaf::new(vec![2, 2]);
        s.add_edge(0, 1, id2(), id2());

        let l = s.laplacian();
        for i in 0..l.nrows() {
            for j in 0..l.ncols() {
                assert!(
                    (l[(i, j)] - l[(j, i)]).abs() < 1e-10,
                    "Laplacian not symmetric at ({},{}): {} vs {}",
                    i,
                    j,
                    l[(i, j)],
                    l[(j, i)]
                );
            }
        }
    }

    #[test]
    fn test_laplacian_identity_structure() {
        let mut s = CellularSheaf::new(vec![2, 2]);
        s.add_edge(0, 1, id2(), id2());

        let l = s.laplacian();
        assert!((l[(0, 0)] - 1.0).abs() < 1e-10, "L[0,0] = 1");
        assert!((l[(3, 3)] - 1.0).abs() < 1e-10, "L[3,3] = 1");
        assert!((l[(0, 2)] - (-1.0)).abs() < 1e-10, "L[0,2] = -1");
    }

    #[test]
    fn test_laplacian_positive_semidefinite() {
        let mut s = CellularSheaf::new(vec![2, 2, 2]);
        s.add_edge(0, 1, id2(), id2());
        s.add_edge(1, 2, id2(), rot90());

        let l = s.laplacian();
        // xᵀLx ≥ 0 for any x
        let x = DVector::from_vec(vec![1.0, -2.0, 3.0, 0.5, -1.0, 4.0]);
        let val = &x.transpose() * &l * &x;
        assert!(
            val[(0, 0)] >= -1e-10,
            "Laplacian not PSD: xᵀLx = {}",
            val[(0, 0)]
        );
    }

    // ── Agent disagreement ──

    #[test]
    fn test_disagreement_positive() {
        let mut s = CellularSheaf::new(vec![2, 2]);
        s.add_edge(0, 1, id2(), id2());

        let mut net = AgentNetwork::new(&s);
        net.set_belief(0, DVector::from_vec(vec![1.0, 0.0]));
        net.set_belief(1, DVector::from_vec(vec![2.0, 0.0]));

        assert!(
            net.disagreement() > 0.0,
            "disagreement > 0 when beliefs differ"
        );
    }

    #[test]
    fn test_disagreement_zero() {
        let mut s = CellularSheaf::new(vec![2, 2]);
        s.add_edge(0, 1, id2(), id2());

        let mut net = AgentNetwork::new(&s);
        let b = DVector::from_vec(vec![1.0, 2.0]);
        net.set_belief(0, b.clone());
        net.set_belief(1, b);

        assert!(
            net.disagreement().abs() < 1e-10,
            "disagreement ≈ 0 when beliefs match"
        );
    }

    #[test]
    fn test_single_agent_disagreement() {
        let s = CellularSheaf::new(vec![3]);
        let mut net = AgentNetwork::new(&s);
        net.set_belief(0, DVector::from_vec(vec![1.0, 2.0, 3.0]));

        assert!(
            net.disagreement().abs() < 1e-10,
            "single agent: zero disagreement"
        );
    }

    // ── Synchronization ──

    #[test]
    fn test_sync_reduces_disagreement() {
        let mut s = CellularSheaf::new(vec![2, 2]);
        s.add_edge(0, 1, id2(), id2());

        let mut net = AgentNetwork::new(&s);
        net.set_belief(0, DVector::from_vec(vec![5.0, 3.0]));
        net.set_belief(1, DVector::from_vec(vec![1.0, 7.0]));

        let before = net.disagreement();
        net.synchronize(0.1);
        let after = net.disagreement();

        assert!(after < before, "sync should reduce disagreement");
    }

    #[test]
    fn test_converge_h1_zero() {
        let mut s = CellularSheaf::new(vec![2, 2]);
        s.add_edge(0, 1, id2(), id2());

        let mut net = AgentNetwork::new(&s);
        net.set_belief(0, DVector::from_vec(vec![1.0, 2.0]));
        net.set_belief(1, DVector::from_vec(vec![3.0, 4.0]));

        let cr = net.converge(0.1, 500, 1e-6);
        assert!(cr.converged, "should converge when H1=0");
        assert!(cr.final_disagreement.abs() < 1e-4, "disagreement → 0");
    }

    #[test]
    fn test_converge_result_fields() {
        let mut s = CellularSheaf::new(vec![2, 2]);
        s.add_edge(0, 1, id2(), id2());

        let mut net = AgentNetwork::new(&s);
        net.set_belief(0, DVector::from_vec(vec![1.0, 0.0]));
        net.set_belief(1, DVector::from_vec(vec![-1.0, 0.0]));

        let cr = net.converge(0.1, 500, 1e-6);
        assert!(cr.converged);
        assert!(!cr.obstruction_detected);
        assert!(cr.iterations > 0);
    }

    // ── can_agree ──

    #[test]
    fn test_can_agree_identity_triangle_false() {
        let mut s = CellularSheaf::new(vec![2, 2, 2]);
        s.add_edge(0, 1, id2(), id2());
        s.add_edge(1, 2, id2(), id2());
        s.add_edge(0, 2, id2(), id2());

        assert!(
            !can_agree(&s, TOL),
            "identity triangle: can_agree = false (H1=2)"
        );
    }

    #[test]
    fn test_can_agree_orthogonal_triangle_true() {
        let mut s = CellularSheaf::new(vec![2, 2, 2]);
        s.add_edge(0, 1, id2(), rot90());
        s.add_edge(1, 2, id2(), rot90());
        s.add_edge(0, 2, id2(), id2());

        assert!(
            can_agree(&s, 1e-6),
            "orthogonal triangle: can_agree = true (H1=0)"
        );
    }

    #[test]
    fn test_can_agree_single_edge_true() {
        let mut s = CellularSheaf::new(vec![2, 2]);
        s.add_edge(0, 1, id2(), id2());

        assert!(
            can_agree(&s, TOL),
            "single edge: can_agree = true"
        );
    }

    // ── forced_disagreement ──

    #[test]
    fn test_forced_disagreement_identity_triangle() {
        let mut s = CellularSheaf::new(vec![2, 2, 2]);
        s.add_edge(0, 1, id2(), id2());
        s.add_edge(1, 2, id2(), id2());
        s.add_edge(0, 2, id2(), id2());

        let fd = forced_disagreement(&s, 1e-6);
        assert!(fd.ncols() > 0, "returns basis for H1>0");
        assert_eq!(fd.nrows(), 6, "edge dimension = 6");
    }

    #[test]
    fn test_forced_disagreement_none() {
        let mut s = CellularSheaf::new(vec![2, 2]);
        s.add_edge(0, 1, id2(), id2());

        let fd = forced_disagreement(&s, TOL);
        assert_eq!(fd.ncols(), 0, "empty when H1=0");
    }

    // ── Quality metric ──

    #[test]
    fn test_quality_agreement_score() {
        let mut s = CellularSheaf::new(vec![2, 2]);
        s.add_edge(0, 1, id2(), id2());

        let mut net = AgentNetwork::new(&s);
        let b = DVector::from_vec(vec![1.0, 0.0]);
        net.set_belief(0, b.clone());
        net.set_belief(1, b);

        let q = net.quality(TOL);
        assert!(
            (q.agreement_score - 1.0).abs() < 1e-6,
            "agreement = 1 when beliefs match"
        );
        assert!(q.h0_dim > 0, "h0_dim > 0 for identity sheaf");
        assert_eq!(q.h1_dim, 0, "h1_dim = 0 for identity sheaf");
    }

    #[test]
    fn test_quality_h1_comparison() {
        // Identity triangle: H¹=2
        let mut s1 = CellularSheaf::new(vec![2, 2, 2]);
        s1.add_edge(0, 1, id2(), id2());
        s1.add_edge(1, 2, id2(), id2());
        s1.add_edge(0, 2, id2(), id2());

        let mut net1 = AgentNetwork::new(&s1);
        let b = DVector::from_vec(vec![1.0, 0.0]);
        for v in 0..3 {
            net1.set_belief(v, b.clone());
        }
        let q1 = net1.quality(TOL);

        // Orthogonal triangle: H¹=0
        let mut s2 = CellularSheaf::new(vec![2, 2, 2]);
        s2.add_edge(0, 1, id2(), rot90());
        s2.add_edge(1, 2, id2(), rot90());
        s2.add_edge(0, 2, id2(), id2());

        let mut net2 = AgentNetwork::new(&s2);
        for v in 0..3 {
            net2.set_belief(v, b.clone());
        }
        let q2 = net2.quality(1e-6);

        assert_eq!(q1.h1_dim, 2, "identity triangle has H1=2");
        assert_eq!(q2.h1_dim, 0, "orthogonal triangle has H1=0");
        assert!(
            q1.h0_dim > q2.h0_dim,
            "identity has more global sections than orthogonal"
        );
    }

    // ── H⁰ basis verification ──

    #[test]
    fn test_h0_basis_are_global_sections() {
        let mut s = CellularSheaf::new(vec![2, 2]);
        s.add_edge(0, 1, id2(), id2());

        let c = s.cohomology(TOL);
        assert_eq!(c.h0_basis.ncols(), 2, "2 global sections");

        // For identity sheaf, global section: x₀ = x₁
        let x0_0 = c.h0_basis[(0, 0)]; // v0, component 0 of basis 0
        let x1_0 = c.h0_basis[(2, 0)]; // v1, component 0 of basis 0
        assert!(
            (x0_0 - x1_0).abs() < 1e-6,
            "global section: v0 = v1"
        );
    }

    // ── Regression: catches old wrong formula ──

    #[test]
    fn test_h1_regression_old_formula() {
        let mut s = CellularSheaf::new(vec![2, 2, 2]);
        s.add_edge(0, 1, id2(), id2());
        s.add_edge(1, 2, id2(), id2());
        s.add_edge(0, 2, id2(), id2());

        let c = s.cohomology(TOL);

        // Old buggy formula: H¹ = ideal_H0 - actual_H0 = 2 - 2 = 0
        // Correct: H¹ = dim(coker(d₀)) = 6 - 4 = 2
        assert_eq!(
            c.h1_dim, 2,
            "regression: identity triangle H1=2, not 0 (old formula gave 0)"
        );
        assert_eq!(c.h1_basis.ncols(), 2, "H1 basis has 2 columns");
        assert_eq!(c.h1_basis.nrows(), 6, "H1 basis has edge_dim=6 rows");

        // Verify H¹ basis vectors are in ker(d₀ᵀ)
        let d0 = build_coboundary(&s);
        let d0t = d0.transpose();
        for col in 0..c.h1_basis.ncols() {
            let v = c.h1_basis.column(col);
            let result = &d0t * v;
            let norm = result.norm();
            assert!(
                norm < 1e-10,
                "H1 basis vector {} not in ker(d0^T): ||d0^T h|| = {}",
                col,
                norm
            );
        }
    }

    // ── Agent network lifecycle ──

    #[test]
    fn test_agent_network_lifecycle() {
        let s = CellularSheaf::new(vec![3, 2, 1]);
        let net = AgentNetwork::new(&s);
        assert_eq!(net.agents.len(), 3);
    }

    // ── Larger sheaf: 5 vertices in a cycle ──

    #[test]
    fn test_five_cycle() {
        let mut s = CellularSheaf::new(vec![1, 1, 1, 1, 1]);
        for i in 0..5 {
            s.add_edge(i, (i + 1) % 5, id(1), id(1));
        }

        let c = s.cohomology(TOL);
        assert_eq!(c.h0_dim, 1, "5-cycle: H0 = 1 (connected, dim=1 stalks)");
        assert_eq!(c.h1_dim, 1, "5-cycle: H1 = 1 (one cycle × dim-1 stalks)");
    }

    // ── Different stalk dimensions ──

    #[test]
    fn test_asymmetric_stalks() {
        // Vertex 0 has dim 3, vertex 1 has dim 1, restriction maps project
        let mut s = CellularSheaf::new(vec![3, 1]);
        let r1 = DMatrix::from_row_slice(1, 3, &[1.0, 0.0, 0.0]); // project x
        let r2 = DMatrix::identity(1, 1);
        s.add_edge(0, 1, r1, r2);

        let c = s.cohomology(TOL);
        // H⁰ = { (a, b, c, a) : arbitrary b, c } → dim 3
        assert_eq!(c.h0_dim, 3, "asymmetric: H0 = 3");
        assert_eq!(c.h1_dim, 0, "asymmetric: H1 = 0 (no cycle)");
    }

    // ── Verify coboundary structure ──

    #[test]
    fn test_coboundary_single_edge() {
        let mut s = CellularSheaf::new(vec![2, 2]);
        s.add_edge(0, 1, id2(), id2());

        let d0 = build_coboundary(&s);
        assert_eq!(d0.nrows(), 2, "edge dim = 2");
        assert_eq!(d0.ncols(), 4, "vertex dim = 4");

        // Should be [I | -I]
        assert!((d0[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((d0[(0, 2)] - (-1.0)).abs() < 1e-10);
        assert!((d0[(1, 1)] - 1.0).abs() < 1e-10);
        assert!((d0[(1, 3)] - (-1.0)).abs() < 1e-10);
    }

    // ── Frobenius-style: Laplacian diagonal dominance ──

    #[test]
    fn test_laplacian_diagonal_nonneg() {
        let mut s = CellularSheaf::new(vec![2, 2, 2]);
        s.add_edge(0, 1, id2(), id2());
        s.add_edge(1, 2, id2(), id2());

        let l = s.laplacian();
        for i in 0..l.nrows() {
            assert!(
                l[(i, i)] >= 0.0,
                "Laplacian diagonal [{},{}] = {} should be ≥ 0",
                i,
                i,
                l[(i, i)]
            );
        }
    }

    // ── Total dimension helpers ──

    #[test]
    fn test_total_dims() {
        let mut s = CellularSheaf::new(vec![2, 3, 1]);
        s.add_edge(0, 1, DMatrix::zeros(2, 2), DMatrix::zeros(2, 3));
        assert_eq!(s.total_vertex_dim(), 6);
        assert_eq!(s.total_edge_dim(), 2);
    }

    // ── Convenience methods (h0, h1, spectral_gap, cohomology_basis) ──

    #[test]
    fn test_h0_h1_convenience() {
        let mut s = CellularSheaf::new(vec![2, 2, 2]);
        s.add_edge(0, 1, id2(), id2());
        s.add_edge(1, 2, id2(), id2());
        s.add_edge(0, 2, id2(), id2());

        assert_eq!(s.h0(TOL), 2, "h0() convenience: identity triangle H0=2");
        assert_eq!(s.h1(TOL), 2, "h1() convenience: identity triangle H1=2");
    }

    #[test]
    fn test_h0_h1_orthogonal() {
        let mut s = CellularSheaf::new(vec![2, 2, 2]);
        s.add_edge(0, 1, id2(), rot90());
        s.add_edge(1, 2, id2(), rot90());
        s.add_edge(0, 2, id2(), id2());

        assert_eq!(s.h0(1e-6), 0, "h0() orthogonal: H0=0");
        assert_eq!(s.h1(1e-6), 0, "h1() orthogonal: H1=0");
    }

    #[test]
    fn test_spectral_gap_identity_edge() {
        let mut s = CellularSheaf::new(vec![2, 2]);
        s.add_edge(0, 1, id2(), id2());

        let gap = s.spectral_gap();
        // 4x4 Laplacian: eigenvalues are 0 (×2) and 2 (×2) → gap = 2
        assert!(
            (gap - 2.0).abs() < 1e-10,
            "spectral gap for identity edge should be 2, got {}",
            gap
        );
    }

    #[test]
    fn test_spectral_gap_small_sheaf_zero() {
        // Single vertex: 1x1 Laplacian is 0 → gap = 0
        let s = CellularSheaf::new(vec![1]);
        assert!(
            s.spectral_gap().abs() < 1e-10,
            "spectral gap for single vertex should be 0"
        );
    }

    #[test]
    fn test_spectral_gap_identity_triangle() {
        let mut s = CellularSheaf::new(vec![2, 2, 2]);
        s.add_edge(0, 1, id2(), id2());
        s.add_edge(1, 2, id2(), id2());
        s.add_edge(0, 2, id2(), id2());

        let gap = s.spectral_gap();
        assert!(gap >= 0.0, "spectral gap should be non-negative, got {}", gap);
        assert!(gap > 0.0, "spectral gap should be > 0 for triangle, got {}", gap);
    }

    #[test]
    fn test_cohomology_basis_h0_single_edge() {
        let mut s = CellularSheaf::new(vec![2, 2]);
        s.add_edge(0, 1, id2(), id2());

        let basis = s.cohomology_basis(0, TOL);
        assert_eq!(basis.len(), 2, "H0 basis should have 2 vectors");
        assert_eq!(basis[0].len(), 4, "each basis vector should have dim=4");
    }

    #[test]
    fn test_cohomology_basis_h1_identity_triangle() {
        let mut s = CellularSheaf::new(vec![2, 2, 2]);
        s.add_edge(0, 1, id2(), id2());
        s.add_edge(1, 2, id2(), id2());
        s.add_edge(0, 2, id2(), id2());

        let basis = s.cohomology_basis(1, TOL);
        assert_eq!(basis.len(), 2, "H1 basis should have 2 vectors");
        assert_eq!(basis[0].len(), 6, "each H1 vector should have edge_dim=6");
    }

    #[test]
    fn test_cohomology_basis_invalid_dim() {
        let s = CellularSheaf::new(vec![2, 2]);
        assert!(s.cohomology_basis(2, TOL).is_empty(), "dim=2 should return empty");
        assert!(s.cohomology_basis(3, TOL).is_empty(), "dim=3 should return empty");
    }

    #[test]
    fn test_disagreement_restriction_mismatch() {
        let mut s = CellularSheaf::new(vec![2, 2]);
        s.add_edge(0, 1, id2(), rot90());

        let mut net = AgentNetwork::new(&s);
        let b = DVector::from_vec(vec![1.0, 0.0]);
        net.set_belief(0, b.clone());
        net.set_belief(1, b);
        let dis = net.disagreement();
        assert!(
            dis > 0.0,
            "disagreement > 0 when restrictions differ with same beliefs, got {}",
            dis
        );
    }

    #[test]
    fn test_converge_can_reach_zero_with_harmonic_beliefs() {
        // Identity triangle has H^1=2, but if beliefs are already a global section
        // (all agents have the same belief), disagreement is 0 even with H^1 > 0.
        let mut s = CellularSheaf::new(vec![2, 2, 2]);
        s.add_edge(0, 1, id2(), id2());
        s.add_edge(1, 2, id2(), id2());
        s.add_edge(0, 2, id2(), id2());

        let coh = s.cohomology(TOL);
        assert_eq!(coh.h1_dim, 2, "identity triangle has H1=2");
        assert!(!can_agree(&s, TOL), "identity triangle: can_agree = false");

        let mut net = AgentNetwork::new(&s);
        let b = DVector::from_vec(vec![3.0, 7.0]);
        net.set_belief(0, b.clone());
        net.set_belief(1, b.clone());
        net.set_belief(2, b);

        let cr = net.converge(0.1, 100, 1e-10);
        // Even though H^1 > 0, starting from a global section means zero disagreement
        assert!(cr.converged, "should converge from global section");
        assert!(cr.final_disagreement.abs() < 1e-8, "disagreement near zero");
    }

    #[test]
    fn test_quality_disagreement_score_mid() {
        let mut s = CellularSheaf::new(vec![2, 2]);
        s.add_edge(0, 1, id2(), id2());

        let mut net = AgentNetwork::new(&s);
        net.set_belief(0, DVector::from_vec(vec![1.0, 0.0]));
        net.set_belief(1, DVector::from_vec(vec![0.0, 1.0]));

        let q = net.quality(TOL);
        assert!(
            q.agreement_score < 1.0 && q.agreement_score > 0.0,
            "agreement between 0 and 1 when beliefs differ, got {}",
            q.agreement_score
        );
    }

    #[test]
    fn test_sheaf_laplacian_alias() {
        let mut s = CellularSheaf::new(vec![2, 2]);
        s.add_edge(0, 1, id2(), id2());

        let l1 = s.laplacian();
        let l2 = s.sheaf_laplacian();
        assert_eq!(l1, l2, "sheaf_laplacian() should match laplacian()");
    }
}
