using ModelingToolkit
using OrdinaryDiffEq
using Plots

function duffing()
    @parameters t m c k μ F₀ ω
    @variables x₁(t) x₂(t)
    D = Differential(t)

    # mx″ + cx′ + kx + μx³ = F₀cos(ωt)
    eqns = [
        D(x₁) ~ x₂,
        D(x₂) ~ (F₀*cos(ω*t) - c*x₂ - k*x₁ - μ*x₁^3)/m
    ]
    sys = ODESystem(eqns)

    # Compile the ODEFunction with the states and parameters in the specified order
    return ODEFunction(sys, [x₁, x₂], [m, c, k, μ, F₀, ω])
end

function integrate(ode, u, p, t)
    # Explicitly convert times to Float64 in case integer values are used
    prob = ODEProblem(ode, u, Float64.(t), p)
    return solve(prob, Tsit5())
end

function example()
    ode = duffing()
    m = 1.0
    c = 0.01
    k = 2.0
    μ = 1
    F₀ = 0.05
    ω = 1.0
    sol = integrate(ode, [0, 0], [m, c, k, μ, F₀, ω], (0, 100))
    plot(sol.t, sol[1,:])
end

# Code to benchmark the timings

# using BenchmarkTools
# ode = duffing()
# m = 1.0
# c = 0.01
# k = 2.0
# μ = 1
# F₀ = 0.05
# ω = 1.0
# @benchmark integrate($ode, $([0, 0]), $([m, c, k, μ, F₀, ω]), (0, 100))

# Output on my laptop

# BenchmarkTools.Trial:
#   memory estimate:  168.48 KiB
#   allocs estimate:  1606
#   --------------
#   minimum time:     101.200 μs (0.00% GC)
#   median time:      107.600 μs (0.00% GC)
#   mean time:        141.544 μs (9.58% GC)
#   maximum time:     8.184 ms (97.48% GC)
#   --------------
#   samples:          10000
#   evals/sample:     1
