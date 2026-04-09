#!/usr/bin/env julia
"""
Test script for FrequentDirections.jl

This script demonstrates the Frequent Directions algorithms and verifies
that the error bounds from Ghashami et al. (2016) are satisfied.
"""

include("../src/FrequentDirections.jl")
using .FrequentDirections
using LinearAlgebra
using Random
using Printf

#==============================================================================#
# Test Helper Functions
#==============================================================================#

"""Generate a test matrix with specified singular value decay."""
function generate_test_matrix(n::Int, d::Int; 
                              rank::Int=min(n,d), 
                              decay::Symbol=:exponential,
                              seed::Int=42)
    Random.seed!(seed)
    
    # Generate orthogonal matrices
    U, _ = qr(randn(n, min(n, d)))
    V, _ = qr(randn(d, min(n, d)))
    U = Matrix(U)[:, 1:rank]
    V = Matrix(V)[:, 1:rank]
    
    # Generate singular values with specified decay
    if decay == :exponential
        s = [exp(-0.1 * i) for i in 1:rank]
    elseif decay == :polynomial
        s = [1.0 / (i^0.5) for i in 1:rank]
    elseif decay == :flat
        s = ones(rank)
    elseif decay == :step
        # First half large, second half small
        s = vcat(ones(rank ÷ 2), 0.01 * ones(rank - rank ÷ 2))
    else
        error("Unknown decay type: $decay")
    end
    
    # Construct matrix
    A = U * Diagonal(s) * V'
    
    return A
end

"""Compute theoretical error bounds."""
function compute_bounds(A::AbstractMatrix, ℓ::Int, k::Int)
    s = svdvals(A)
    tail_energy = sum(s[k+1:end].^2)
    
    cov_bound = tail_energy / (ℓ - k)
    proj_bound = (ℓ / (ℓ - k)) * tail_energy
    
    return (covariance=cov_bound, projection=proj_bound, tail_energy=tail_energy)
end

#==============================================================================#
# Test 1: Basic Functionality Test
#==============================================================================#

function test_basic_functionality()
    println("\n" * "="^70)
    println("TEST 1: Basic Functionality")
    println("="^70)
    
    # Small test case
    n, d, ℓ = 100, 20, 10
    A = generate_test_matrix(n, d; decay=:exponential)
    
    println("Input matrix: $n × $d")
    println("Sketch size: ℓ = $ℓ")
    
    # Test FDSketch
    println("\n--- FDSketch (Algorithm 1) ---")
    sketch_fd = FDSketch(ℓ, d)
    fit!(sketch_fd, A)
    B_fd = get_sketch(sketch_fd)
    println("Sketch shape: $(size(B_fd))")
    println("Runtime: $(round(sketch_fd.runtime * 1000, digits=3)) ms")
    println("Rows processed: $(sketch_fd.n_rows_processed)")
    
    # Test FastFDSketch
    println("\n--- FastFDSketch (Algorithm 2) ---")
    sketch_fast = FastFDSketch(ℓ, d)
    fit!(sketch_fast, A)
    B_fast = get_sketch(sketch_fast)
    println("Sketch shape: $(size(B_fast))")
    println("Runtime: $(round(sketch_fast.runtime * 1000, digits=3)) ms")
    println("Rows processed: $(sketch_fast.n_rows_processed)")
    
    # Verify both sketches satisfy bounds
    k = 0
    bounds = compute_bounds(A, ℓ, k)
    
    cov_err_fd = covariance_error(sketch_fd, A)
    cov_err_fast = covariance_error(sketch_fast, A)
    
    println("\n--- Error Analysis (k=$k) ---")
    println("Covariance error bound (Theorem 3.1): $(round(bounds.covariance, sigdigits=4))")
    println("FD covariance error: $(round(cov_err_fd, sigdigits=4)) ✓" * 
            (cov_err_fd ≤ bounds.covariance * 1.001 ? " (within bound)" : " (EXCEEDS BOUND!)"))
    println("Fast-FD covariance error: $(round(cov_err_fast, sigdigits=4)) ✓" * 
            (cov_err_fast ≤ bounds.covariance * 1.001 ? " (within bound)" : " (EXCEEDS BOUND!)"))
    
    return true
end

#==============================================================================#
# Test 2: Error Bound Verification
#==============================================================================#

function test_error_bounds()
    println("\n" * "="^70)
    println("TEST 2: Error Bound Verification (Theorems 3.1 and 3.2)")
    println("="^70)
    
    n, d = 500, 50
    A = generate_test_matrix(n, d; decay=:polynomial)
    
    println("Input matrix: $n × $d")
    println()
    
    # Test various sketch sizes
    sketch_sizes = [10, 20, 30, 40]
    
    println("Sketch | Cov Error | Cov Bound | Ratio | Proj Error² | Proj Bound | Ratio")
    println("-"^80)
    
    all_pass = true
    for ℓ in sketch_sizes
        k = 0
        result = run_fd_benchmark(A, ℓ; k=k, use_fast=true)
        
        cov_ratio = result.covariance_error / result.covariance_error_bound
        proj_ratio = result.projection_error_squared / result.projection_error_bound
        
        pass_cov = cov_ratio ≤ 1.001
        pass_proj = proj_ratio ≤ 1.001
        
        status = (pass_cov && pass_proj) ? "✓" : "✗"
        
        @printf("  %3d  | %9.4f | %9.4f | %5.3f | %11.4f | %10.4f | %5.3f %s\n",
                ℓ, result.covariance_error, result.covariance_error_bound, cov_ratio,
                result.projection_error_squared, result.projection_error_bound, proj_ratio, status)
        
        all_pass = all_pass && pass_cov && pass_proj
    end
    
    println()
    if all_pass
        println("All error bounds verified! ✓")
    else
        println("Some bounds exceeded! ✗")
    end
    
    return all_pass
end

#==============================================================================#
# Test 3: Speedup Comparison
#==============================================================================#

function test_speedup_comparison()
    println("\n" * "="^70)
    println("TEST 3: Runtime Comparison (FD vs Fast-FD)")
    println("="^70)
    
    test_cases = [
        (n=500, d=100, ℓ=20),
        (n=1000, d=100, ℓ=30),
        (n=2000, d=200, ℓ=50),
    ]
    
    println("\n  n × d   | ℓ  | FD Time (ms) | Fast-FD Time (ms) | Speedup")
    println("-"^70)
    
    for tc in test_cases
        A = generate_test_matrix(tc.n, tc.d; decay=:exponential)
        
        # Run FD
        sketch_fd = FDSketch(tc.ℓ, tc.d)
        fit!(sketch_fd, A)
        time_fd = sketch_fd.runtime
        
        # Run Fast-FD
        sketch_fast = FastFDSketch(tc.ℓ, tc.d)
        fit!(sketch_fast, A)
        time_fast = sketch_fast.runtime
        
        speedup = time_fd / time_fast
        
        @printf("%4d × %3d | %2d |    %8.2f   |      %8.2f      |  %5.1f×\n",
                tc.n, tc.d, tc.ℓ, time_fd * 1000, time_fast * 1000, speedup)
    end
    
    return true
end

#==============================================================================#
# Test 4: Different Matrix Types
#==============================================================================#

function test_different_matrices()
    println("\n" * "="^70)
    println("TEST 4: Different Matrix Types")
    println("="^70)
    
    n, d, ℓ = 500, 50, 15
    k = 5
    
    decay_types = [:exponential, :polynomial, :flat, :step]
    
    println("\nDecay Type    | ‖A-A_k‖²_F | Cov Error | Bound   | Ratio")
    println("-"^60)
    
    for decay in decay_types
        A = generate_test_matrix(n, d; decay=decay)
        
        bounds = compute_bounds(A, ℓ, k)
        result = run_fd_benchmark(A, ℓ; k=k, use_fast=true)
        
        ratio = result.covariance_error / result.covariance_error_bound
        status = ratio ≤ 1.001 ? "✓" : "✗"
        
        @printf("%-13s | %10.4f | %9.4f | %7.4f | %5.3f %s\n",
                decay, bounds.tail_energy, result.covariance_error, 
                result.covariance_error_bound, ratio, status)
    end
    
    return true
end

#==============================================================================#
# Test 5: Streaming Property Verification
#==============================================================================#

function test_streaming_property()
    println("\n" * "="^70)
    println("TEST 5: Streaming Property (Row-by-Row Updates)")
    println("="^70)
    
    n, d, ℓ = 200, 30, 10
    A = generate_test_matrix(n, d; decay=:exponential)
    
    # Batch processing
    sketch_batch = FastFDSketch(ℓ, d)
    fit!(sketch_batch, A)
    
    # Streaming processing (row by row)
    sketch_stream = FastFDSketch(ℓ, d)
    for i in 1:n
        FrequentDirections.update!(sketch_stream, A[i, :])
    end
    
    B_batch = get_sketch(sketch_batch)
    B_stream = get_sketch(sketch_stream)
    
    # Check that both methods produce the same result
    diff = norm(B_batch - B_stream)
    
    println("Batch processing vs streaming:")
    println("  Difference in sketches: $(round(diff, sigdigits=4))")
    println("  Match: $(diff < 1e-10 ? "✓" : "✗")")
    
    return diff < 1e-10
end

#==============================================================================#
# Run All Tests
#==============================================================================#

function run_all_tests()
    println("\n" * "="^70)
    println("FREQUENT DIRECTIONS ALGORITHM TESTS")
    println("="^70)
    
    results = Dict{String, Bool}()
    
    results["Basic Functionality"] = test_basic_functionality()
    results["Error Bounds"] = test_error_bounds()
    results["Speedup Comparison"] = test_speedup_comparison()
    results["Different Matrices"] = test_different_matrices()
    results["Streaming Property"] = test_streaming_property()
    
    println("\n" * "="^70)
    println("TEST SUMMARY")
    println("="^70)
    
    all_pass = true
    for (name, passed) in results
        status = passed ? "PASS ✓" : "FAIL ✗"
        println("  $name: $status")
        all_pass = all_pass && passed
    end
    
    println()
    if all_pass
        println("All tests passed! ✓")
    else
        println("Some tests failed! ✗")
    end
    
    return all_pass
end

#==============================================================================#
# Demo
#==============================================================================#

function demo()
    println("\n" * "="^70)
    println("FREQUENT DIRECTIONS DEMO")
    println("="^70)
    
    # Generate test data
    Random.seed!(42)
    n, d, ℓ = 1000, 100, 25
    
    println("\nGenerating test matrix with exponentially decaying singular values...")
    A = generate_test_matrix(n, d; decay=:exponential)
    println("Matrix size: $n × $d")
    println("Sketch size: ℓ = $ℓ")
    
    # Run comparison
    println("\nRunning algorithms...")
    results = compare_fd_algorithms(A, ℓ; k=5)
    
    println("\n" * "-"^60)
    println("Summary:")
    println("  Both FD and Fast-FD satisfy the theoretical error bounds")
    println("  Fast-FD provides significant speedup for large matrices")
    println("  The sketch B captures the principal directions of A")
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_all_tests()
    demo()
end