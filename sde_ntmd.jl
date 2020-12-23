# This is just a bit of code to set up a temporary package environment since not everyone
# will have the same environment set up; this is largely unnecessary when working only on
# your own code.
# See https://bkamins.github.io/julialang/2020/06/28/automatic-project-environments.html
using Pkg: Pkg
# Change to a newly created temporary directory
cd(mktempdir()) do
    Pkg.activate(".")  # activate a new project in the temporary directory
    Pkg.UPDATED_REGISTRY_THIS_SESSION[] = true  # don't update the registry; on Windows it's slow...
    # Add the required packages
    Pkg.add([Pkg.PackageSpec(name="StochasticDiffEq"),
        Pkg.PackageSpec(name="OrdinaryDiffEq"),
        Pkg.PackageSpec(name="StaticArrays"),
        Pkg.PackageSpec(name="Plots"),])
end

# Load required packages
using StochasticDiffEq  # SDE solvers
using OrdinaryDiffEq  # ODE solvers
using StaticArrays  # SVectors for speed
using Plots  # Plotting

# Note on StaticArrays: Ordinary vectors (like Matlab) are created with square brackets,
# e.g., [1, 2, 3], and are good for most purposes. However, since they are variable size the
# compiler cannot optimise as much as it could if they were a fixed size. StaticArrays
# provides fixed-size arrays which allow the compiler to do lots of optimisations and so
# often work much faster for small arrays (typically where there are fewer than 20 elements
# in the array).

"""
    NTMD

A nonlinear tuned-mass-damper system with stochastic forcing of the form

```math
m₁x₁″ + c₁(x₁′ - x₂′) + k₁x₁ + k₂(x₁ - x₂) + μ(x₁ - x₂)^3 = f
m₂x₂″ + c₂(x₂′ - x₁′) + k₂(x₂ - x₁) + μ(x₂ - x₁)^3 = 0
```

where

```math
f = F₀cos(ωt) + ξ(t)
```

and `ξ(t)` is a Brownian motion with variance `σ`.

"""
Base.@kwdef mutable struct NTMD
    m₁::Float64 = 1.0
    c₁::Float64 = 0.02
    k₁::Float64 = 1.0
    m₂::Float64 = 0.5
    c₂::Float64 = 0.05
    k₂::Float64 = 0.95
    μ::Float64 = 0.1
    σ::Float64 = 0.5
    F₀::Float64 = 1.0
    ω::Float64 = 1.0
end

function ntmd_dt(u, p::NTMD, t)
    # Deterministic part
    x₁ = u[1]
    x₁′ = u[2]
    x₂ = u[3]
    x₂′ = u[4]
    f = p.F₀*cos(p.ω*t)
    return SVector(
        x₁′,
        (f - p.c₁*(x₁′ - x₂′) - p.k₁*x₁ - p.k₂*(x₁ - x₂) - p.μ*(x₁ - x₂)^3)/p.m₁,
        x₂′,
        (-p.c₂*(x₂′ - x₁′) - p.k₂*(x₂ - x₁) - p.μ*(x₂ - x₁)^3)/p.m₂,
    )
end

function ntmd_dW(u, p::NTMD, t)
    # Stochastic part
    return SVector(0.0, p.σ/p.m₁, 0.0, 0.0)
end

function ntmd_integrate_sde(u₀, p::NTMD, t)
    # Solve the full SDE
    sde = SDEProblem(ntmd_dt, ntmd_dW, SVector{4, Float64}(u₀), (0.0, Float64(t)), p)
    return solve(sde, SOSRA())
end

function ntmd_integrate_ode(u₀, p::NTMD, t)
    # Solve only the deterministic part of the SDE
    ode = ODEProblem(ntmd_dt, SVector{4, Float64}(u₀), (0.0, Float64(t)), p)
    return solve(ode, Tsit5())
end

function ntmd_example()
    p = NTMD(ω=1.1)  # NTMD with default parameters except ω
    sol = ntmd_integrate_ode(ones(4), p, 2π/p.ω*1000)  # integrate to remove the transient
    u₀ = sol[:, end]  # use the end point for more simulations
    sol = ntmd_integrate_ode(u₀, p, 2π/p.ω*20)  # integrate the deterministic part only (for comparison purposes)
    sol_sde = ntmd_integrate_sde(u₀, p, 2π/p.ω*20)  # integrate the full stochastic differential equation
    plot(sol.t, sol[1, :])
    plot!(sol_sde.t, sol_sde[1, :])
    gui()
    return sol_sde
end
