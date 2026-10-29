using CSV
using DataFrames
using DifferentialEquations
using Plots
using Dates
using JLD2
"""
BD settling model with 30 layers (can be adapted) and only TSS considered as component.
The VESILIND settling velocity function is used to calculate the settling
fluxes per layer. 
The Gudonov flux constraint is applied to the settling
fluxes per layer.
Compression is added to the model.
Batch settling model so no convective fluxes. No dispersion implemented either

Layer indexing (1-based, Julia convention): 
  layer 3  = top layer
  layer 32 = bottom layer
"""

# Original mechanistic compression function
function compression_reference(X, alpha, Rhos, Rhof, v0, r_v, X_crit, g)
    if X < X_crit
        return zero(X)
    else
        return alpha * (24 * 3600)^2 * Rhos * v0 *
            (exp(-r_v * X) - exp(-r_v * X_crit)) /
            (-r_v * g * (Rhos - Rhof))
    end
end

function batch_settler_equations!(dy, y, par, t)
    # Unpack parameters
    v0        = par[1]
    r_v       = par[2]
    total_height  = par[3]
    
    alpha = par[4] 
    Rhos = par[5]
    Rhof = par[6]
    X_crit = par[7]
    g = par[8]

    # extra parameters BD
    C_min = par[9]
    C_max = par[10]

    nr_of_layers = 30


    h = total_height / nr_of_layers # layer height, checked in west, should not include the 4 extra layers

    # Velocities and fluxes: in batch we only consider settling fluxes, no inflow or outflow
    vs  = zeros(nr_of_layers+4)
    J_G_Help  = zeros(nr_of_layers+4) # initial gravitational settling flux
    J_G  = zeros(nr_of_layers+4) #  gravitational settling flux with gudonov constraints
    D_comp = zeros(nr_of_layers+4) # primitive of the compression  

    # Vesilind settling velocity calculation per layer
    for i in 1:nr_of_layers+4
           vs[i] = max(0.0,  v0 * (exp(-r_v * y[i] )))
    end

    # Settling flux per layer
    for i in 1:nr_of_layers+4 # for bottom and extra layers, settling flux =0
        J_G_Help[i] = y[i] * vs[i]
    end

    # Calculate flux limitation
    C_fbk_max = clamp(1 / r_v, C_min, C_max) # returns 1/rv unles 1/rv is outside interval C_min and C_max
    Fbk_max = v0 * C_fbk_max * exp(-r_v * C_fbk_max)

    for i in 1:nr_of_layers+3
        if y[i] < y[i+1]
            J_G[i] = min(J_G_Help[i], J_G_Help[i+1])
        elseif (C_fbk_max - y[i]) * (C_fbk_max - y[i+1]) < 0
            J_G[i] = Fbk_max
        else
            J_G[i] = max(J_G_Help[i], J_G_Help[i+1])
        end
    end

    # Primitive of compression per layer
    # primitive D of dcomp canbe expressed in closed form (see compression_reference fct) --> Dnum i =D(C i)(exact primitive).
    for i in 1:nr_of_layers+4
        D_comp[i] = compression_reference( y[i], alpha, Rhos, Rhof, v0, r_v, X_crit, g)
        # D_comp[i] = 0
    end

    # --- Closed batch boundaries ---
    # No settling flux through the top and bottom of the physical column
    J_G[2]             = 0.0   # between layer 2 (effluent) and layer 3 (top)
    J_G[nr_of_layers+2] = 0.0  # between layer 32 (bottom) and layer 33 (underflow)

    # No compression flux through the boundaries (zero gradient)
    D_comp[2]             = D_comp[3]
    D_comp[nr_of_layers+3] = D_comp[nr_of_layers+2]

    dy .= 0.0

    # Suspended solids (TSS)
    # Differential equations - 4 extra layers: 2 in effluent zone + 2 in underflow zone

    # Extra layers in effluent zone
    dy[1] = 0 # only convective flux, so for batch experiment = 0
    dy[2] = 0 # (1/h) *(-J_G[2]+  (1/h) *( D_comp[3]-D_comp[2]))
    # Layers 1 to N-1 IN settler
    for i in 3:nr_of_layers+2
        dy[i] = (1/h) * (J_G[i-1] - J_G[i] + (1/h) *(D_comp[i+1]+D_comp[i-1]-2*D_comp[i])) 
    end
	# Extra layers in underflow zone
    dy[nr_of_layers+3] = 0 # (1/h) * (J_G[nr_of_layers+2] + (1/h)* (D_comp[nr_of_layers+2]-D_comp[nr_of_layers+3]))
    dy[nr_of_layers+4] = 0

    return nothing
end


function calculate_sludge_blanket_height(X_layers, X_lim, total_height) 
    """ Calculate sludge blanket height based on a TSS threshold. 
    Layer 3 = top layer 
    Layer 32 = bottom layer 
    X_lim = TSS concentration defining the sludge blanket interface. 
    The sludge blanket is defined as the region from the bottom up where the layer concentration is above X_lim. 
    """ 
    NrOfLayers = 30
    h = total_height / NrOfLayers # layer height

    # Only consider the 30 physical layers, from top (3) to bottom (32)
    X = X_layers[3:32]

    # Find the first layer below the threshold, starting from the bottom
    index_of_blanket = findfirst(x -> x <= X_lim, reverse(X))
   
    if isnothing(index_of_blanket) # All layers are above the threshold 
        H_S = total_height
    elseif index_of_blanket == 1 # Even the bottom layer is below the threshold 
        H_S = 0.0 
    else # Layers below the threshold crossing constitute the blanket 
        H_S = (index_of_blanket-1) * h 
    end 
    return H_S, index_of_blanket 
end

function settling_column(y0, tspan, par_settler)
    """
    Batch settler model with 30 layers and 1 component (MLSS).
    """

    prob = ODEProblem(batch_settler_equations!, y0, tspan, par_settler)
    sol = solve(
        prob, 
        Tsit5(),  # Tsit5 = Tsitouras 5/4 Runge–Kutta method, Runge–Kutta method designed for solving non-stiff ordinary differential equations (ODEs).
        saveat = 1/14400) # save results every 6 seconds (1/14400 days) for plotting and analysis

    return sol
end

function check_mass_balance(X_init)
    # Initial concentration profile (start concentration in each layer)
    X_layers = zeros(34)
    X_layers[3:32] .= X_init

    ### PARAMETERS SETTLER ### for now: west default
    v0 = 250 #m/d max practical settling velocity
    r_v = 0.000576 #m3/g hindered settling parameter
    total_height = 0.28 #m column height

    ### compression parameters
    alpha = 0.8 # m2/s2
    Rhos = 1050000 # g/m3
    Rhof = 998000 # g/m3
    X_crit = 6000 # g/m3
    g = 73231257600 # m/d2

    C_min = 0 # g/m3
    C_max = 20000 # g/m3

    par_settler = [v0, r_v, total_height, alpha, Rhos, Rhof, X_crit, g, C_min, C_max]

    tspan = (0.0, 30/1440) # 0 to 30 min, in days

    sol = settling_column(X_layers, tspan, par_settler)

    # Mass balance
    D = 0.095
    A = pi * D^2 / 4
    h = total_height / 30

    # Total mass in all 34 model layers (g)
    M_all = [sum(X) * h * A for X in sol.u]

    # Mass in the 30 physical layers only (g)
    M_column = [sum(X[3:32]) * h * A for X in sol.u]

    # Initial mass in the physical column (g)
    M0 = M_column[1]

    println("Initial physical-column mass: ", M0, " g")
    println("Final physical-column mass:   ", M_column[end], " g")
    println("Change in column mass:         ", M_column[end] - M0, " g")
    println("Relative column mass change:   ",
            100 * (M_column[end] - M0) / M0, " %")

    println("Initial all-layer mass:        ", M_all[1], " g")
    println("Final all-layer mass:          ", M_all[end], " g")
    println("Change in all-layer mass:       ", M_all[end] - M_all[1], " g")

    # Plot mass over time
    time_min = sol.t .* 1440

    p_mass = plot(time_min, M_column,
        label = "Physical column",
        xlabel = "Time (min)",
        ylabel = "TSS mass (g)",
        title = "Batch settling mass balance")

    plot!(p_mass, time_min, M_all,
        label = "All 34 model layers",
        linestyle = :dash)

    savefig(p_mass,
        "output/BD_batch_settling_model_wcompression/mass_balance$(X_init).png")
end

function test_one_init_concentration(X_init)
    # Initial concentration profile (start concentration in each layer)
    X_layers = zeros(34)
    X_layers[3:32] .= X_init

    ### PARAMETERS SETTLER ### for now: west default
    v0 = 250 #m/d max practical settling velocity
    r_v = 0.000576 #m3/g hindered settling parameter
    total_height = 0.28 #m column height

    ### compression parameters
    alpha = 0.5# m2/s2
    Rhos = 1050000 # g/m3
    Rhof = 998000 # g/m3
    X_crit = 10000 # g/m3
    g = 73231257600 # m/d2

    C_min = 0 # g/m3
    C_max = 20000 # g/m3

    par_settler = [v0, r_v, total_height, alpha, Rhos, Rhof, X_crit, g, C_min, C_max]

    tspan = (0.0, 30/1440) # 0 to 30 min, in days

    sol = settling_column(X_layers, tspan, par_settler)
    times = sol.t

    # Convert solution to matrix: rows = time, columns = layers
    X_layers_arr = permutedims(reduce(hcat, sol.u))

    X_lim = 900 #mg/L
    H_S_all = Float64[]
    for X in sol.u
        H_S, index_of_blanket =
            calculate_sludge_blanket_height(X,X_lim,total_height)
        push!(H_S_all, H_S)
    end
    
    time_min = times .* 1440.0

    # All 30 layers over time
    p1 = plot(
        xlabel = "Time (min)",
        ylabel = "TSS",
        legend = :topright,
        title = "All 30 layers"
    )

    for layer in [3:4:32; 32]
        plot!(
            p1,
            time_min,
            X_layers_arr[:, layer],
            label = "Layer $(layer)"
        )
    end
    savefig("output/BD_batch_settling_model_wcompression/X_layers_Xinit$(X_init).png")

    p2 = plot(
        time_min,
        H_S_all, 
        label="Model sludge blanket height", 
        ylabel="Sludge blanket height (m)", 
        xlabel="Time", 
        xticks = 0:5:30,
        ylims=(0, 0.28)
        ) 
    savefig("output/BD_batch_settling_model_wcompression/sludge_blanket_height_Xinit$(X_init).png") 
end

function generate_references()
    ## generate reference solution for training the neural network

    ### PARAMETERS SETTLER for now: west default
    v0 = 250 #m/d max practical settling velocity
    r_v = 0.000576 #m3/g hindered settling parameter
    total_height = 0.28 #m column height

    ### compression parameters
    alpha = 0.5 # m2/s2
    Rhos = 1050000 # g/m3
    Rhof = 998000 # g/m3
    X_crit = 10000 # g/m3
    g = 73231257600 # m/d2

    C_min = 0 # g/m3
    C_max = 20000 # g/m3

    par_settler = [v0, r_v, total_height, alpha, Rhos, Rhof, X_crit, g, C_min, C_max]

    X_inits = [1000.0, 2000.0, 3000.0, 5000.0, 6000.0, 7000.0, 8000.0, 9000.0, 10000.0]
    X_inits_test = [2500.0, 5500.0, 7500.0, 11000.0]
    tspan = (0.0, 30/1440)
    saveat = 1/14400

    refs = [solve(ODEProblem(batch_settler_equations!,  [zeros(2); fill(X0, 30); zeros(2)], tspan, par_settler), Tsit5(); saveat, reltol=1e-8, abstol=1e-8) for X0 in X_inits]
    Y_refs = [Array(sol) for sol in refs]
    t_refs = [sol.t for sol in refs]
    @save "output/BD_settler_reference_train.jld2" X_inits Y_refs t_refs

    refs_test  = [solve(ODEProblem(batch_settler_equations!,  [zeros(2); fill(X0, 30); zeros(2)], tspan, par_settler), Tsit5(); saveat, reltol=1e-8, abstol=1e-8) for X0 in X_inits_test]
    Y_refs_test  = [Array(sol) for sol in refs_test]
    t_refs_test  = [sol.t for sol in refs_test ]
    @save "output/BD_settler_reference_test.jld2" X_inits_test Y_refs_test t_refs_test

    p1 = plot(layout=(3,3), size=(1200,900), xlabel="Time [min]", ylabel="TSS concentration", legend=:best)

    for (j, (X0, Y_ref, t_ref)) in enumerate(zip(X_inits, Y_refs, t_refs))
        plot!(p1[j], t_ref .* 1440, Array(Y_ref[[3:4:32; 32], :])', title="X₀ = $(X0)", lw=2)
    end

    savefig(p1, "output/BD_reference_train_profiles.png")
end 

# Differentiable (interpolated) SBH, identical to the one used in the UDE loss
function sludge_blanket_height(X::AbstractVector, X_lim, total_height; nr_of_layers = 30)
    h = total_height / nr_of_layers
    for k in 1:nr_of_layers                           # k = 1 is the bottom layer
        x_k = X[nr_of_layers + 3 - k]
        if x_k <= X_lim
            k == 1 && return zero(x_k)                # even the bottom layer is below threshold
            x_above = X[nr_of_layers + 3 - (k - 1)]
            frac = (x_above - X_lim) / (x_above - x_k)
            return (k - 1.5 + frac) * h
        end
    end
    return oftype(float(X[3]), total_height)          # all layers above threshold
end

function generate_SBH_references(; X_lim = 900.0, interpolate = true)
    ### PARAMETERS SETTLER (same as generate_references)
    v0 = 250
    r_v = 0.000576
    total_height = 0.28

    ### compression parameters (the "true" mechanistic compression)
    alpha = 0.5
    Rhos = 1050000
    Rhof = 998000
    X_crit = 10000
    g = 73231257600

    C_min = 0
    C_max = 20000

    par_settler = [v0, r_v, total_height, alpha, Rhos, Rhof, X_crit, g, C_min, C_max]

    X_inits      = [1000.0, 2000.0, 3000.0, 5000.0, 6000.0, 7000.0, 8000.0, 9000.0, 10000.0]
    X_inits_test = [2500.0, 5500.0, 7500.0, 11000.0]
    tspan  = (0.0, 30/1440)
    saveat = 1/14400

    # SBH time series for one simulated solution
    sbh_series(Y) = interpolate ?
        [sludge_blanket_height(view(Y, :, k), X_lim, total_height) for k in axes(Y, 2)] :
        [calculate_sludge_blanket_height(view(Y, :, k), X_lim, total_height)[1] for k in axes(Y, 2)]

    # Simulate and reduce to SBH directly, nothing else is kept
    function simulate_sbh(X0s)
        sols = [solve(ODEProblem(batch_settler_equations!, [zeros(2); fill(X0, 30); zeros(2)], tspan, par_settler),
                      Tsit5(); saveat, reltol = 1e-8, abstol = 1e-8) for X0 in X0s]
        return [sbh_series(Array(sol)) for sol in sols], [sol.t for sol in sols]
    end

    SBH_refs, t_refs = simulate_sbh(X_inits)
    @save "BD_settler_SBH_reference_train.jld2" X_inits SBH_refs t_refs X_lim

    SBH_refs_test, t_refs_test = simulate_sbh(X_inits_test)
    @save "BD_settler_SBH_reference_test.jld2" X_inits_test SBH_refs_test t_refs_test X_lim

    # --- plots ---
    function make_plot(X0s, ts, SBHs, title)
        n = length(X0s)
        nc = ceil(Int, sqrt(n)); nr = cld(n, nc)
        p = plot(layout = (nr, nc), size = (350 * nc, 300 * nr), legend = false, plot_title = title)
        for j in 1:n
            plot!(p[j], ts[j] .* 1440, SBHs[j]; lw = 2, title = "X₀ = $(X0s[j])",
                  xlabel = "Time [min]", ylabel = "SBH [m]", ylims = (0, total_height))
        end
        return p
    end
    savefig(make_plot(X_inits, t_refs, SBH_refs, "SBH train references (X_lim = $X_lim)"),
            "output/BD_batch_settling_model_wcompression/BD_SBH_reference_train.png")
    savefig(make_plot(X_inits_test, t_refs_test, SBH_refs_test, "SBH test references (X_lim = $X_lim)"),
            "output/BD_batch_settling_model_wcompression/BD_SBH_reference_test.png")

    return nothing
end

function main()
    #test_one_init_concentration(2500)
    #check_mass_balance(1000)
    generate_references()
    #generate_SBH_references(X_lim = 900.0)
end


main()