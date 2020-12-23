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
    Pkg.add(Pkg.PackageSpec(name="Plots"))
end

# Load required packages
using Plots

function integrate_sde(sde, x0, t1; dt=0.01)
    # Euler-Maruyama for a 1D SDE
    x = x0
    for t in range(0, t1, step=dt)
        (f, g) = sde(x)
        x = x + f*dt + g*sqrt(dt)*randn()
    end
    return x
end

function integrate_many(sde, x0, t1; n = 1, kwargs...)
    # Integrate many sample paths using Euler-Maruyama
    x = zeros(n)
    Threads.@threads for i in eachindex(x)  # use threads to speed up the computation
        x[i] = integrate_sde(sde, x0, t1; kwargs...)
    end
    return x
end

ornstein_uhlenbeck(x, θ, σ) = (-θ*x, σ) # Ornstein-Uhlenbeck equation (this is short hand notation for a function)
cubic(x, θ, σ) = (θ*x - x^3, σ) # SDE with two potential wells

println("Using $(Threads.nthreads()) threads")

x = integrate_many(x->ornstein_uhlenbeck(x, 1.0, 0.1), 0.0, 10.0, n = 1_000_000) # integrate a million sample paths
plt1 = histogram(x)
title!("Ornstein-Uhlenbeck: θ=1.0, σ=0.1")

x = integrate_many(x->cubic(x, 1.0, 0.1), 0.0, 10.0, n = 1_000_000) # integrate a million sample paths
plt2 = histogram(x)
title!("Cubic: θ=1.0, σ=0.1")

x = integrate_many(x->cubic(x, 1.0, 1.5), 0.0, 10.0, n = 1_000_000) # integrate a million sample paths
plt3 = histogram(x)
title!("Cubic: θ=1.0, σ=1.5")

# Use `gui(plt1)` etc to show the different figures
