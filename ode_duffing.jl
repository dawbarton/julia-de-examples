using OrdinaryDiffEq
using StaticArrays
using Plots

"""
    Duffing

A Duffing equation with periodic forcing of the form

```math
mx″ + cx′ + kx + μx³ = F₀cos(ωt)
```
"""
Base.@kwdef mutable struct Duffing
    m::Float64 = 1.0
    c::Float64 = 0.01
    k::Float64 = 2.0
    μ::Float64 = 1
    F₀::Float64 = 0.05
    ω::Float64 = 1.0
end

"""
    rhs_dt(u, p::Duffing, t)

Return the right-hand side of the Duffing.
"""
function rhs_dt(u, p::Duffing, t)
    x = u[1]
    x′ = u[2]
    f = p.F₀ * cos(p.ω * t)
    # Use a static vector (SVector) for speed; it's not necessary for any other reason
    return SVector(x′, (f - p.c * x′ - p.k * x - p.μ * x^3) / p.m)
end

"""
    integrate(u, p::Duffing, t)

Return a solution of the Duffing equation over the time interval `t` starting at the initial
condition `u` with the parameters of the Duffing equation taken from `p`.
"""
function integrate(u, p::Duffing, t)
    # Explicitly convert everything to Float64 in case integer values are used
    # Use SVectors for speed (not necessary)
    prob = ODEProblem(rhs_dt, SVector{2, Float64}(u), Float64.(t), p)
    return solve(prob, Tsit5())
end

"""
    example()

Run an example Duffing equation simulation and plot the results.
"""
function example()
    p = Duffing()
    sol = integrate([0, 0], p, (0, 100))
    plot(sol.t, sol[1,:])
end

# Code to benchmark the timings

# using BenchmarkTools
# p = Duffing()
# @benchmark integrate($([0, 0]), $p, (0, 100))

# Output on my laptop

# BenchmarkTools.Trial:
#   memory estimate:  60.34 KiB
#   allocs estimate:  462
#   --------------
#   minimum time:     73.100 μs (0.00% GC)
#   median time:      75.700 μs (0.00% GC)
#   mean time:        90.955 μs (3.73% GC)
#   maximum time:     3.807 ms (96.36% GC)
#   --------------
#   samples:          10000
#   evals/sample:     1
