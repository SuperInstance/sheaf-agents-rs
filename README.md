# sheaf-agents-rs

Your microservices can't agree on the state of the world. You added more message queues. It got worse. The disagreement is structural.

This is a Rust library that models multi-agent systems as **cellular sheaves on graphs**. It computes exactly where your agents *structurally cannot agree* — not because of latency or bugs, but because the math makes it impossible.

## The Idea in One Sentence

**H¹ > 0 means communication can't help.** Your agents need a new perspective, not more couples therapy.

## Show Me

```rust
use sheaf_agents::{CellularSheaf, can_agree, rot90};
use nalgebra::DMatrix;

// Two agents looking at the same 2D data through different lenses
let mut sheaf = CellularSheaf::new(vec![2, 2]);

// Agent 0 sees (x, y). Agent 1 sees (y, -x) — a 90° rotation.
let id = DMatrix::identity(2, 2);
sheaf.add_edge(0, 1, id.clone(), rot90());

// Can they agree?
let ok = can_agree(&sheaf, 1e-8); // true — no cycle, no obstruction

// Now add a third agent and close the triangle...
```

That single edge is fine. But connect three agents in a triangle with two rotation edges and one identity edge, and you get H¹ = 0 (coboundary is surjective) but H⁰ = 0 (no global sections). The cycle composes to a 180° rotation — `rot90 ∘ rot90 ∘ identity = -identity ≠ identity`. There is no global belief compatible with all three edges. Not "hard to find." *Doesn't exist.*

## Why You Should Care

Every team that's ever said "we just need better communication" and been wrong was dealing with H¹ > 0.

The disagreement wasn't in the messages. It was in the structure of who sees what and how those perspectives compose around cycles.

**Applications:**

- **Microservice consensus** — Your services form a graph. Each has its own schema (stalk). The API translations between them are restriction maps. H¹ tells you if a globally consistent state is even possible.
- **Multi-model AI agreement** — Different models processing the same input through different latent spaces. H¹ measures whether those spaces can ever be reconciled.
- **Sensor fusion** — Multiple sensors with different measurement models. If the composition of coordinate transforms around any loop isn't the identity, you have structural disagreement.
- **Legislative voting** — Voting systems where the order of pairwise comparisons matters (Condorcet cycles). The sheaf structure determines if a transitive ranking exists.

## Why Committees Deadlock

It's not personalities. It's not bad facilitation. It's the sheaf structure.

Three committee members comparing options pairwise: A vs B, B vs C, A vs C. If each comparison uses different criteria (different restriction maps), you get a cycle where the "group preference" around the loop is incoherent. Condorcet's paradox is a special case of H¹ > 0.

The fix isn't "better discussion." It's restructuring the comparison criteria so the maps compose to the identity around every cycle.

## Quick Start

```toml
[dependencies]
sheaf-agents = "0.1"
nalgebra = "0.33"
```

```rust
use sheaf_agents::{CellularSheaf, AgentNetwork, can_agree, forced_disagreement};
use nalgebra::{DMatrix, DVector};

// Build a sheaf
let mut sheaf = CellularSheaf::new(vec![2, 2, 2]);
let id = DMatrix::identity(2, 2);
sheaf.add_edge(0, 1, id.clone(), id.clone());
sheaf.add_edge(1, 2, id.clone(), id.clone());
sheaf.add_edge(0, 2, id.clone(), id.clone());

// Check cohomology
let coh = sheaf.cohomology(1e-8);
println!("H⁰ = {} (global sections)", coh.h0_dim); // 2
println!("H¹ = {} (obstructions)", coh.h1_dim);     // 2

// Can agents ever agree?
println!("Can agree: {}", can_agree(&sheaf, 1e-8)); // false

// What can't they agree on?
let fd = forced_disagreement(&sheaf, 1e-6);
println!("Forced disagreements: {} dimensions", fd.ncols()); // 2

// Simulate agents
let mut net = AgentNetwork::new(&sheaf);
net.set_belief(0, DVector::from_vec(vec![1.0, 0.0]));
net.set_belief(1, DVector::from_vec(vec![0.0, 1.0]));
net.set_belief(2, DVector::from_vec(vec![1.0, 1.0]));

let result = net.converge(0.1, 1000, 1e-6);
println!("Converged: {}", result.converged);           // false
println!("Obstruction: {}", result.obstruction_detected); // true
```

## API

### Sheaf (the math layer)

| Type/Method | What it does |
|------------|-------------|
| `CellularSheaf::new(stalk_dims)` | Build a sheaf with given stalk dimensions |
| `sheaf.add_edge(v1, v2, r1, r2)` | Add an edge with restriction maps |
| `sheaf.laplacian()` | Compute the sheaf Laplacian (disagreement energy) |
| `sheaf.cohomology(tol)` | Compute H⁰ and H¹ dimensions + bases |

### Agent Layer (the simulation layer)

| Type/Method | What it does |
|------------|-------------|
| `AgentNetwork::new(&sheaf)` | Spawn agents on a sheaf |
| `net.set_belief(agent, belief)` | Set an agent's local state |
| `net.synchronize(step)` | One gradient descent step on disagreement |
| `net.converge(step, max_iter, tol)` | Run until convergence or obstruction |
| `net.disagreement()` | Measure current total disagreement |

### Consensus (the answer layer)

| Function | What it does |
|----------|-------------|
| `can_agree(&sheaf, tol)` | Is H¹ = 0? Can agreement exist at all? |
| `forced_disagreement(&sheaf, tol)` | Compute an H¹ basis — *what* agents cannot agree on |
| `net.quality(tol)` | Score the quality of current consensus |

## How It Works

1. **Stalks** — Each agent has a vector space of possible beliefs. Dimension = how many independent things they track.
2. **Restriction maps** — Each edge has two linear maps saying how each agent's beliefs project onto what the other can observe. These encode perspective differences.
3. **Sheaf Laplacian** — Generalizes the graph Laplacian. Measures total local inconsistency: how much each agent's projected belief disagrees with its neighbor's projected belief on every edge.
4. **H⁰** (global sections) — The kernel of the coboundary d₀. Beliefs where every edge sees perfect agreement. This is the space of possible consensus states.
5. **H¹** (obstructions) — The cokernel of d₀, computed as ker(d₀ᵀ). Dimensions of disagreement that survive regardless of what beliefs agents hold. If this is positive, consensus is *structurally impossible*.

The synchronization algorithm is gradient descent on the disagreement energy. When H¹ = 0, it converges. When H¹ > 0, it oscillates forever — not because the algorithm is bad, but because there's nowhere to converge to.

## The Math (For the Curious)

A **cellular sheaf** F on a graph G assigns:
- A vector space F(v) (the *stalk*) to each vertex v
- A linear map F(v→w): F(v) → F(e) (the *restriction map*) to each half-edge

The **coboundary map** δ₀: C⁰ → C¹ sends a 0-cochain (assignment of values to vertices) to a 1-cochain (assignment of values to edges) by applying restriction maps:

```
(δ₀ x)(edge v→w) = F(w→e)(x_w) − F(v→e)(x_v)
```

- **H⁰ = ker(δ₀)** — assignments where every edge sees zero difference. These are global sections: the states where all agents agree.
- **H¹ = coker(δ₀) = ker(δ₀ᵀ)** — the 1-cochains that aren't the image of any 0-cochain. These are the obstructions: disagreements with no upstream cause.

The **sheaf Laplacian** is L = δ₀ᵀδ₀. It's positive semidefinite. Its kernel equals H⁰.

For the ordinary graph Laplacian, every edge has restriction map = identity, all stalks are 1-dimensional (ℝ), and H¹ = 0 for connected components. The sheaf Laplacian generalizes this by letting stalks have different dimensions and restriction maps be arbitrary linear transformations.

## License

MIT
