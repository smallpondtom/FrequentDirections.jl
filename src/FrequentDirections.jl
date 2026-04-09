"""
    FrequentDirections.jl

Implementation of the Frequent Directions (FD) and Fast Frequent Directions (Fast-FD)
algorithms from Ghashami et al. (2016) "Frequent Directions: Simple and Deterministic 
Matrix Sketching" SIAM J. Comput.

The algorithms produce a sketch B ∈ ℝ^{ℓ×d} of an input matrix A ∈ ℝ^{n×d} such that
for any unit vector x ∈ ℝ^d:
    0 ≤ ‖Ax‖² - ‖Bx‖² ≤ ‖A - A_k‖²_F / (ℓ - k)

This provides covariance error bounds:
    ‖AᵀA - BᵀB‖₂ ≤ ‖A - A_k‖²_F / (ℓ - k)
"""
module FrequentDirections

using LinearAlgebra

export FDSketch, FastFDSketch
export fit!, get_sketch, covariance_error, projection_error, reset!
export covariance_error_bound, projection_error_bound 
export run_fd_benchmark, compare_fd_algorithms

#==============================================================================#
# Abstract type and common interface
#==============================================================================#

abstract type AbstractFDSketch end

"""
    get_sketch(sketch::AbstractFDSketch) -> Matrix

Return the current sketch matrix B.
"""
function get_sketch end

"""
    reset!(sketch::AbstractFDSketch)

Reset the sketch to its initial state.
"""
function reset! end

"""
    fit!(sketch::AbstractFDSketch, A::AbstractMatrix)

Fit the sketch to the input matrix A by processing each row.
"""
function fit! end

#==============================================================================#
# Algorithm 1: Basic Frequent Directions
#==============================================================================#

"""
    FDSketch{T<:AbstractFloat}

Basic Frequent Directions sketch (Algorithm 1 from Ghashami et al.).

Maintains an ℓ × d sketch matrix B. Each row update triggers an SVD
and shrinkage operation. Total time complexity: O(ndℓ²).

# Fields
- `ℓ::Int`: sketch size (number of rows to retain)
- `d::Int`: ambient dimension (number of columns)
- `B::Matrix{T}`: the sketch matrix
- `next_zero_row::Int`: index of next row to fill
- `n_rows_processed::Int`: count of rows processed
- `runtime::Float64`: cumulative runtime in seconds
"""
mutable struct FDSketch{T<:AbstractFloat} <: AbstractFDSketch
    ℓ::Int
    d::Int
    B::Matrix{T}
    next_zero_row::Int
    n_rows_processed::Int
    runtime::Float64
end

"""
    FDSketch(ℓ::Int, d::Int; T::Type=Float64)

Create a new Frequent Directions sketch with sketch size ℓ and ambient dimension d.
"""
function FDSketch(ℓ::Int, d::Int; T::Type{<:AbstractFloat}=Float64)
    @assert ℓ > 0 "Sketch size ℓ must be positive"
    @assert d > 0 "Dimension d must be positive"
    B = zeros(T, ℓ, d)
    return FDSketch{T}(ℓ, d, B, 1, 0, 0.0)
end

function reset!(sketch::FDSketch{T}) where T
    fill!(sketch.B, zero(T))
    sketch.next_zero_row = 1
    sketch.n_rows_processed = 0
    sketch.runtime = 0.0
    return sketch
end

function get_sketch(sketch::FDSketch)
    # In basic FD, the sketch is always ℓ rows
    # The last row may be zero after shrinkage, but we return all ℓ rows
    # Find actual non-zero rows
    n_nonzero = 0
    for i in 1:sketch.ℓ
        if norm(sketch.B[i, :]) > eps(eltype(sketch.B)) * sketch.d
            n_nonzero = i
        end
    end
    return sketch.B[1:max(1, n_nonzero), :]
end

"""
    update!(sketch::FDSketch, row::AbstractVector)

Process a single row and update the sketch.
"""
function update!(sketch::FDSketch{T}, row::AbstractVector) where T
    @assert length(row) == sketch.d "Row dimension mismatch"
    
    t_start = time()
    
    # Insert row into the last position
    sketch.B[sketch.ℓ, :] .= row
    
    # Compute SVD: B = U * Σ * Vᵀ
    F = svd(sketch.B)
    
    # Number of singular values is min(ℓ, d)
    n_sv = length(F.S)
    
    # Shrinkage: subtract σ_ℓ² from all squared singular values
    # If n_sv < ℓ, use the smallest available singular value
    shrink_idx = min(sketch.ℓ, n_sv)
    δ = F.S[shrink_idx]^2
    
    # Compute shrunk singular values: √max(σᵢ² - δ, 0)
    shrunk_S = zeros(T, sketch.ℓ)
    for i in 1:min(sketch.ℓ, n_sv)
        shrunk_S[i] = sqrt(max(F.S[i]^2 - δ, zero(T)))
    end
    
    # Reconstruct B = Σ_shrunk * Vᵀ (sketch rows are scaled right singular vectors)
    # Take top ℓ rows of Vt (or all if n_sv < ℓ)
    n_rows = min(sketch.ℓ, n_sv)
    sketch.B[1:n_rows, :] .= Diagonal(shrunk_S[1:n_rows]) * F.Vt[1:n_rows, :]
    
    # Zero out remaining rows if any
    if n_rows < sketch.ℓ
        sketch.B[n_rows+1:sketch.ℓ, :] .= zero(T)
    end
    
    sketch.n_rows_processed += 1
    sketch.runtime += time() - t_start
    
    return sketch
end

function fit!(sketch::FDSketch{T}, A::AbstractMatrix) where T
    n, d = size(A)
    @assert d == sketch.d "Dimension mismatch: expected $(sketch.d), got $d"
    
    for i in 1:n
        update!(sketch, view(A, i, :))
    end
    
    return sketch
end

#==============================================================================#
# Algorithm 2: Fast Frequent Directions
#==============================================================================#

"""
    FastFDSketch{T<:AbstractFloat}

Fast Frequent Directions sketch (Algorithm 2 from Ghashami et al.).

Maintains a 2ℓ × d buffer matrix. SVD is computed only when the buffer is full,
resulting in amortized O(dℓ) time per row instead of O(dℓ²).

# Fields
- `ℓ::Int`: sketch size (number of rows to retain)
- `d::Int`: ambient dimension (number of columns)
- `B::Matrix{T}`: the buffer matrix (2ℓ × d)
- `next_zero_row::Int`: index of next zero row in buffer
- `n_rows_processed::Int`: count of rows processed
- `runtime::Float64`: cumulative runtime in seconds
"""
mutable struct FastFDSketch{T<:AbstractFloat} <: AbstractFDSketch
    ℓ::Int
    d::Int
    B::Matrix{T}
    next_zero_row::Int
    n_rows_processed::Int
    runtime::Float64
end

"""
    FastFDSketch(ℓ::Int, d::Int; T::Type=Float64)

Create a new Fast Frequent Directions sketch with sketch size ℓ and ambient dimension d.
The internal buffer has size 2ℓ × d.
"""
function FastFDSketch(ℓ::Int, d::Int; T::Type{<:AbstractFloat}=Float64)
    @assert ℓ > 0 "Sketch size ℓ must be positive"
    @assert d > 0 "Dimension d must be positive"
    B = zeros(T, 2ℓ, d)
    return FastFDSketch{T}(ℓ, d, B, 1, 0, 0.0)
end

function reset!(sketch::FastFDSketch{T}) where T
    fill!(sketch.B, zero(T))
    sketch.next_zero_row = 1
    sketch.n_rows_processed = 0
    sketch.runtime = 0.0
    return sketch
end

function get_sketch(sketch::FastFDSketch)
    # Return the top ℓ rows (the sketch after rotation)
    # Following the reference: return self._sketch[:self.ell,:]
    return sketch.B[1:sketch.ℓ, :]
end

"""
    update!(sketch::FastFDSketch, row::AbstractVector)

Process a single row and update the sketch. SVD is computed only when buffer is full.
Based on the FrequentDirections implementation pattern.
"""
function update!(sketch::FastFDSketch{T}, row::AbstractVector) where T
    @assert length(row) == sketch.d "Row dimension mismatch"
    
    t_start = time()
    
    # Rotate BEFORE inserting if buffer is full
    if sketch.next_zero_row > 2 * sketch.ℓ
        _rotate!(sketch)
    end
    
    # Insert row into the next available position
    sketch.B[sketch.next_zero_row, :] .= row
    sketch.next_zero_row += 1
    
    sketch.n_rows_processed += 1
    sketch.runtime += time() - t_start
    
    return sketch
end

"""
    _rotate!(sketch::FastFDSketch)

Perform the shrinkage rotation when the buffer is full.
"""
function _rotate!(sketch::FastFDSketch{T}) where T
    # Compute SVD: B = U * Σ * Vᵀ
    # With full_matrices=false, Vt has shape min(2ℓ, d) × d
    F = svd(sketch.B)
    
    n_sv = length(F.S)
    
    if n_sv >= sketch.ℓ
        # Normal case: shrink by σ_ℓ² (using 1-based indexing)
        δ = F.S[sketch.ℓ]^2
        
        # Compute shrunk singular values for top ℓ components
        shrunk_S = zeros(T, sketch.ℓ)
        for i in 1:sketch.ℓ
            shrunk_S[i] = sqrt(max(F.S[i]^2 - δ, zero(T)))
        end
        
        # Reconstruct: sketch rows are Σ_shrunk * Vᵀ (scaled right singular vectors)
        # B[:ℓ, :] = diag(shrunk_S) * Vt[:ℓ, :]
        sketch.B[1:sketch.ℓ, :] .= Diagonal(shrunk_S) * F.Vt[1:sketch.ℓ, :]
        
        # Zero out the bottom half
        sketch.B[sketch.ℓ+1:2*sketch.ℓ, :] .= zero(T)
        
        # Next available row is ℓ+1
        sketch.next_zero_row = sketch.ℓ + 1
    else
        # Edge case: fewer singular values than ℓ
        # Just keep what we have without shrinkage
        sketch.B[1:n_sv, :] .= Diagonal(F.S) * F.Vt
        sketch.B[n_sv+1:2*sketch.ℓ, :] .= zero(T)
        sketch.next_zero_row = n_sv + 1
    end
    
    return sketch
end

function fit!(sketch::FastFDSketch{T}, A::AbstractMatrix) where T
    n, d = size(A)
    @assert d == sketch.d "Dimension mismatch: expected $(sketch.d), got $d"
    
    for i in 1:n
        update!(sketch, view(A, i, :))
    end
    
    return sketch
end

#==============================================================================#
# Error Metrics
#==============================================================================#

"""
    covariance_error(sketch::AbstractFDSketch, A::AbstractMatrix) -> Float64

Compute the covariance error: ‖AᵀA - BᵀB‖₂ where B is the sketch.
This is the spectral norm of the difference of the covariance matrices.
"""
function covariance_error(sketch::AbstractFDSketch, A::AbstractMatrix)
    B = get_sketch(sketch)
    AᵀA = A' * A
    BᵀB = B' * B
    # Spectral norm = largest singular value
    return opnorm(AᵀA - BᵀB, 2)
end

"""
    covariance_error_normalized(sketch::AbstractFDSketch, A::AbstractMatrix, k::Int) -> Float64

Compute the normalized covariance error: ‖AᵀA - BᵀB‖₂ / (‖A - A_k‖²_F / (ℓ - k))
where A_k is the best rank-k approximation. Should be ≤ 1 by Theorem 3.1.
"""
function covariance_error_normalized(sketch::AbstractFDSketch, A::AbstractMatrix, k::Int)
    @assert k < sketch.ℓ "k must be less than sketch size ℓ"
    
    cov_err = covariance_error(sketch, A)
    
    # Compute ‖A - A_k‖²_F = sum of squared singular values beyond rank k
    s = svdvals(A)
    tail_energy = sum(s[k+1:end].^2)
    
    bound = tail_energy / (sketch.ℓ - k)
    
    return cov_err / bound
end

function covariance_error_bound(sketch::AbstractFDSketch, A::AbstractMatrix, k::Int)
    @assert k < sketch.ℓ "k must be less than sketch size ℓ"
    # Compute ‖A - A_k‖²_F = sum of squared singular values beyond rank k
    s = svdvals(A)
    tail_energy = sum(s[k+1:end].^2)
    bound = tail_energy / (sketch.ℓ - k)
    return bound
end

"""
    projection_error(sketch::AbstractFDSketch, A::AbstractMatrix) -> Float64

Compute the projection error: ‖A - A·π_B(A)‖_F where π_B(A) projects A onto
the row space of B.

Note: This computes the Frobenius norm of the residual when projecting A
onto the subspace spanned by the rows of B.
"""
function projection_error(sketch::AbstractFDSketch, A::AbstractMatrix)
    B = get_sketch(sketch)
    
    # Project A onto row space of B
    # Row space of B is spanned by rows of B = column space of Bᵀ
    # Projection matrix onto column space of Bᵀ is: Bᵀ(BBᵀ)⁻¹B
    # But we want to project rows of A, so we compute: A * Bᵀ(BBᵀ)⁻¹B
    
    if size(B, 1) == 0
        return norm(A, 2)
    end
    
    # Use SVD for numerical stability
    F = svd(B)
    # V contains the right singular vectors (basis for row space of B)
    # Only use components with non-negligible singular values
    tol = eps(eltype(B)) * max(size(B)...) * maximum(F.S)
    r = sum(F.S .> tol)
    
    if r == 0
        return norm(A, 2)
    end
    
    V_r = F.V[:, 1:r]  # d × r matrix
    
    # Project each row of A onto span of V_r columns
    # A_proj = A * V_r * V_rᵀ
    A_proj = A * V_r * V_r'
    
    return norm(A - A_proj, 2)  # Frobenius norm
end

"""
    projection_error_frobenius(sketch::AbstractFDSketch, A::AbstractMatrix) -> Float64

Compute the squared Frobenius projection error: ‖A - A·P_B‖²_F
"""
function projection_error_frobenius(sketch::AbstractFDSketch, A::AbstractMatrix)
    B = get_sketch(sketch)
    
    if size(B, 1) == 0
        return norm(A)^2
    end
    
    F = svd(B)
    tol = eps(eltype(B)) * max(size(B)...) * maximum(F.S)
    r = sum(F.S .> tol)
    
    if r == 0
        return norm(A)^2
    end
    
    V_r = F.V[:, 1:r]
    A_proj = A * V_r * V_r'
    
    return norm(A - A_proj)^2
end


"""
    projection_error_bound(sketch::AbstractFDSketch, A::AbstractMatrix, k::Int) -> Float64
"""
function projection_error_bound(sketch::AbstractFDSketch, A::AbstractMatrix, k::Int)
    s = svdvals(A)
    tail_energy = sum(s[k+1:end].^2)
    bound = (1 + k / (sketch.ℓ - k)) * tail_energy
    return bound
end


#==============================================================================#
# Benchmark Function
#==============================================================================#

"""
    FDBenchmarkResult

Results from running the FD benchmark.
"""
struct FDBenchmarkResult
    algorithm::String
    n::Int
    d::Int
    ℓ::Int
    runtime::Float64
    covariance_error::Float64
    covariance_error_bound::Float64
    projection_error_squared::Float64
    projection_error_bound::Float64
end

function Base.show(io::IO, r::FDBenchmarkResult)
    println(io, "FD Benchmark Result ($(r.algorithm))")
    println(io, "  Matrix size: $(r.n) × $(r.d), sketch size: $(r.ℓ)")
    println(io, "  Runtime: $(round(r.runtime * 1000, digits=3)) ms")
    println(io, "  Covariance error: $(round(r.covariance_error, sigdigits=4)) (bound: $(round(r.covariance_error_bound, sigdigits=4)))")
    println(io, "  Projection error²: $(round(r.projection_error_squared, sigdigits=4)) (bound: $(round(r.projection_error_bound, sigdigits=4)))")
end

"""
    run_fd_benchmark(A::AbstractMatrix, ℓ::Int; k::Int=0, use_fast::Bool=true)

Run the Frequent Directions algorithm on matrix A with sketch size ℓ.

# Arguments
- `A`: Input matrix of size n × d
- `ℓ`: Sketch size
- `k`: Rank parameter for computing error bounds (default: 0)
- `use_fast`: If true, use Fast-FD (Algorithm 2), else use basic FD (Algorithm 1)

# Returns
- `FDBenchmarkResult`: Struct containing timing and error metrics
"""
function run_fd_benchmark(A::AbstractMatrix, ℓ::Int; k::Int=0, use_fast::Bool=true)
    n, d = size(A)
    @assert ℓ ≤ min(n, d) "Sketch size ℓ must be ≤ min(n, d)"
    @assert k < ℓ "Rank parameter k must be < ℓ"
    
    T = eltype(A)
    if !(T <: AbstractFloat)
        T = Float64
        A = convert(Matrix{Float64}, A)
    end
    
    # Create and run sketch
    if use_fast
        sketch = FastFDSketch(ℓ, d; T=T)
        alg_name = "Fast-FD"
    else
        sketch = FDSketch(ℓ, d; T=T)
        alg_name = "FD"
    end
    
    fit!(sketch, A)
    
    # Compute error metrics
    cov_err = covariance_error(sketch, A)
    proj_err_sq = projection_error_frobenius(sketch, A)
    
    # Compute bounds (Theorem 3.1 and 3.2)
    # ‖AᵀA - BᵀB‖₂ ≤ ‖A - A_k‖²_F / (ℓ - k)
    s = svdvals(A)
    tail_energy = sum(s[k+1:end].^2)
    cov_bound = tail_energy / (ℓ - k)
    
    # For projection error, the bound is (1 + k/(ℓ-k)) ‖A - A_k‖²_F
    # which simplifies to ℓ/(ℓ-k) ‖A - A_k‖²_F
    proj_bound = (ℓ / (ℓ - k)) * tail_energy
    
    return FDBenchmarkResult(
        alg_name,
        n, d, ℓ,
        sketch.runtime,
        cov_err,
        cov_bound,
        proj_err_sq,
        proj_bound
    )
end

"""
    compare_fd_algorithms(A::AbstractMatrix, ℓ::Int; k::Int=0)

Compare FD and Fast-FD algorithms on the same input matrix.
"""
function compare_fd_algorithms(A::AbstractMatrix, ℓ::Int; k::Int=0)
    result_fd = run_fd_benchmark(A, ℓ; k=k, use_fast=false)
    result_fast = run_fd_benchmark(A, ℓ; k=k, use_fast=true)
    
    println("="^60)
    println("Comparison: FD vs Fast-FD")
    println("="^60)
    println()
    println(result_fd)
    println()
    println(result_fast)
    println()
    println("Speedup (Fast-FD vs FD): $(round(result_fd.runtime / result_fast.runtime, digits=2))×")
    
    return (fd=result_fd, fast_fd=result_fast)
end

end # module


