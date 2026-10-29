## ##############################################################################################################
# Settler UDE: BD-type 30-layer batch settler where the compression term is replaced by a NN.
# Optimization workflow mirrors the ASM hybrid UDE script:
#   - parameter transform / back-transform
#   - Adam schedule (n_epochs, l_cutoff, batchsize, optimizer)
#   - minibatching (shuffled DataLoader over settling experiments)
#   - early stopping in a callback, intermediate saving, loss CSV
#   - ForwardDiff (PolyesterForwardDiff) gradients through Rodas5P(autodiff=AutoFiniteDiff())
#   - validate mode: load params, simulate, plot, save, stop
## ##############################################################################################################

using CSV
using DataFrames
using DifferentialEquations
using Plots
using Printf
using Statistics
using Random
using JLD2

using Lux
using NNlib: softplus
using ComponentArrays

using Zygote
using MLUtils
using Optimization
using OptimizationOptimisers
using ForwardDiff
using PolyesterForwardDiff
using DifferentiationInterface


# get project path
path_project = abspath(pwd())


## ##############################################################################################################
@info "Loading settings"
#################################################################################################################

validate = false


run_name = "settler_UDE_compression"

reference_file = joinpath(path_project, "BD_settler_reference.jld2")   # contains X_inits, Y_refs, t_refs
validation_params_file = joinpath(path_project, "results", "settler_ude_final_result.jld2") 

reference_file_test = joinpath(path_project, "BD_settler_reference_test.jld2")   # contains X_inits, Y_refs, t_refs

# # Start training from these params (last existing file wins)
training_params_file = [
    # Add paths to pretraining files here
]

# Solver tolerances used inside the loss
solver_reltol = 1e-4
solver_abstol = 1e-3

# Chunksize for ForwardDiff (321 NN params -> not an exact divisor, last chunk is just partially filled)
# chunk is a group of parameters for which ForwardDiff calculates derivatives simultaneously
chunksize = 12

# Scaling of NN input / output and of the loss
X_scale = 8000.0
D_scale = 250000.0

# Settler parameters (west default)
v0 = 250.0               # m/d max practical settling velocity
r_v = 0.000576            # m3/g hindered settling parameter
total_height = 0.28      # m column height
C_min = 0 #g/m3
C_max = 20000 #g/m3 
X_crit = 10000.0          # g/m3 compression threshold
par_settler = [v0, r_v, total_height, C_min, C_max]

## ##############################################################################################################
@info "Defining the model"
#################################################################################################################

# Neural network: concentration -> (normalised) compression
NNmodel = Lux.Chain(
    Lux.Dense(1, 16, tanh),
    Lux.Dense(16, 4, tanh),
    Lux.Dense(4, 1, softplus), # non-negative compression
)
rng = Random.Xoshiro(42)
p_NN_init, st_NN = Lux.setup(rng, NNmodel)
p_NN_init = Float64.(ComponentArray(p_NN_init))
p_ax = getaxes(p_NN_init)   # needed to rebuild the ComponentArray from the plain optimisation vector

struct SettlerNNParams{P,N,S,M}
    settler::P
    p_NN::N
    st_NN::S
    NNmodel::M
    X_scale::Float64
    D_scale::Float64
    X_crit::Float64
end

# Compression for all layers in one NN call (features x batch = 1 x n_layers)
function compression_nn(X::AbstractVector, p_NN, st_NN, NNmodel, X_scale, D_scale, X_crit)
    pred, _ = NNmodel(reshape(X ./ X_scale, 1, :), p_NN, st_NN)
    return [
        X[i] < X_crit ? zero(pred[1, i]) :
        D_scale * ((X[i] - X_crit) / X_scale) * pred[1, i]
        for i in eachindex(X)
    ]
end

function batch_settler_ude_equations!(dy, y, par, t)
    v0, r_v, total_height, C_min, C_max = par.settler
    
    nr_of_layers = 30
    h = total_height / nr_of_layers

    # Element type must cover both states and (possibly dual) NN parameters
    T = promote_type(eltype(y), eltype(par.p_NN))   
    J_G_Help  = zeros(T, nr_of_layers+4) # initial gravitational settling flux
    J_G  = zeros(T, nr_of_layers+4) #  gravitational settling flux with gudonov constraints

    # Vesilind settling flux
    for i in 1:nr_of_layers+4
        J_G_Help[i] = max(0.0, v0 * exp(-r_v * y[i])) * y[i]
    end
    
    # Calculate Godunov flux limitation
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

    D_comp = compression_nn(y, par.p_NN, par.st_NN, par.NNmodel, par.X_scale, par.D_scale, par.X_crit)

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

## ##############################################################################################################
@info "Loading reference data"
#################################################################################################################

# X_inits: initial MLSS per experiment, Y_refs: 10 x nt matrices, t_refs: time vectors [d]
@load reference_file X_inits Y_refs t_refs
n_exp = length(X_inits)
all_idx = collect(1:n_exp)

@printf "\tNumber of settling experiments:\t%d\n" n_exp
println("\tX init of settling experiments:\t", X_inits)
# for i in eachindex(Y_refs)
#     println("Experiment $i:")
#     println("  Y_refs[$i]: ", size(Y_refs[i]))
#     println("  t_refs[$i]: ", size(t_refs[i]))
# end

# Base problem (Float64). Everything below is built via remake from this one.
tspan_full = (0.0, maximum(last.(t_refs)))
prob_optim = ODEProblem(
    batch_settler_ude_equations!,
    [zeros(2); fill(X_inits[1], 30); zeros(2)],
    tspan_full,
    SettlerNNParams(par_settler, p_NN_init, st_NN, NNmodel, X_scale, D_scale, X_crit),
)

## ##############################################################################################################
@info "Setting up Parameter Optimization"
#################################################################################################################

# Parameter transforms. Only NN weights are tuned -> identity.
# If mechanistic parameters (v0, r_v, ...) are added later, put the softplus-inverse trick from the ASM script here.
# only the NN weights are tuned, and they’re unconstrained, so both functions are just copy
parameter_transform(x) = copy(x)
parameter_back_transform(x) = copy(x)

# Simulate one settling experiment with NN parameters x (already back-transformed)
function simulate_experiment(x, prob, X0, t)
    T = eltype(x)
    par = prob.p
    par_new = SettlerNNParams(par.settler, ComponentArray(x, p_ax), par.st_NN, par.NNmodel, par.X_scale, par.D_scale, par.X_crit)
    newprob = remake(prob; u0 = [zeros(2); fill(T(X0), 30); zeros(2)], tspan = (first(t), last(t)), p = par_new)
    return solve(
        newprob,
        Tsit5(); 
        saveat = 1/14400,
        reltol = solver_reltol,
        abstol = solver_abstol,
        maxiters = 10^6,
        verbose = DEVerbosity(SciMLLogging.None()),
    )
end

# Loss: mean over the experiments in idx of the scaled mean squared error of all layer profiles
function loss(x::AbstractVector, idx::AbstractVector{<:Integer})
    xp = parameter_back_transform(x)
    total = zero(eltype(x))
    for j in idx
        sol = simulate_experiment(xp, prob_optim, X_inits[j], t_refs[j])
        if !SciMLBase.successful_retcode(sol) 
            println("Loss = Inf: experiment $j failed with retcode $(sol.retcode)")
            return convert(eltype(x), Inf)
        end
            
        pred = hcat(sol.(t_refs[j])...)
        if size(pred, 2) != length(t_refs[j])
            println("Loss = Inf: experiment $j returned $(size(pred, 2)) sampled time points, expected $(length(t_refs[j]))")
            return convert(eltype(x), Inf)  # failed -> inf loss
        end
        if !all(isfinite, pred)
            println("Loss = Inf: experiment $j contains NaN or Inf in the solution")
            return convert(eltype(x), Inf)
        end
        err = (pred .- Y_refs[j]) ./ X_scale
        if !all(isfinite, err)
            println("Loss = Inf: experiment $j has NaN or Inf in the scaled errors")
            return convert(eltype(x), Inf)
        end
        total += sum(abs2, err) / length(Y_refs[j]) # MSE loss (abs2 is sum of squared abs values)
    end
    return total / length(idx)
end
evaluate_loss(x, idx = all_idx) = loss(x, idx) #for evaluation use all losses

# Plot reference vs prediction for all experiments
function plot_fit(x, plot_title)
    xp = parameter_back_transform(x)
    nc = ceil(Int, sqrt(n_exp))
    nr = cld(n_exp, nc)
    pl = plot(layout = (nr, nc), size = (350 * nc, 300 * nr), legend = false, plot_title = plot_title)
    for j in 1:n_exp
        sol = simulate_experiment(xp, prob_optim, X_inits[j], t_refs[j])
        pred = hcat(sol.(t_refs[j])...)
        plot!(pl[j], t_refs[j] .* 1440, pred[[3:4:32; 32], :]'; color = :red, title = "X₀ = $(X_inits[j])",
              xlabel = "Time [min]", ylabel = "TSS [g/m³]")
        plot!(pl[j], t_refs[j] .* 1440, Array(Y_refs[j][[3:4:32; 32], :])'; color = :black, linestyle = :dot, alpha = 1)
    end
    return pl
end
# ########### Compare compression NN to the mechanistic function
# Mechanistic compression parameters. NOTE: separate name X_crit_mech on purpose.
# Overwriting the global X_crit (as the old commented block did with X_crit = 6000)
# would silently change the cutoff used inside the NN model.
alpha       = 0.5            # m2/s2
Rhos        = 1050000.0       # g/m3
Rhof        = 998000.0        # g/m3
g           = 73231257600.0   # m/d2
X_crit_mech = 10000.0          # g/m3

function compression_reference(X, alpha, Rhos, Rhof, v0, r_v, X_crit, g)
    if X < X_crit
        return zero(X)
    else
        return alpha * (24 * 3600)^2 * Rhos * v0 *
            (exp(-r_v * X) - exp(-r_v * X_crit)) /
            (-r_v * g * (Rhos - Rhof))
    end
end

X_values = collect(range(0.0, 12000.0, length = 500))
D_reference = [compression_reference(X, alpha, Rhos, Rhof, v0, r_v, X_crit_mech, g) for X in X_values]

# Highest concentration seen in the training data (NN is only constrained up to here)
X_data_max = maximum(maximum.(Y_refs))

# NN compression for a (back-transformed) parameter vector x
function compression_nn_curve(x, X_vals = X_values)
    p = ComponentArray(Float64.(x), p_ax)
    return compression_nn(X_vals, p, st_NN, NNmodel, X_scale, D_scale, X_crit)
end

# Plot mechanistic vs. NN compression; `curves` is a vector of (label, x, style) tuples
function plot_compression(curves, plot_title)
    pl = plot(X_values, D_reference; label = "Mechanistic compression", linewidth = 2.5,
        color = :dodgerblue, xlabel = "TSS concentration X [g/m³]",
        ylabel = "Compression D_comp", title = plot_title,
        legend = :topleft, size = (900, 550), grid = true,
        tickfontsize = 11, guidefontsize = 14, titlefontsize = 16, legendfontsize = 11)
    for (label, x, color, ls) in curves
        plot!(pl, X_values, compression_nn_curve(x); label = label,
            linewidth = 2.5, linestyle = ls, color = color)
    end
    return pl
end

function save_test_predictions(x, output_prefix)
    isempty(reference_file_test) && return
    @load reference_file_test X_inits_test Y_refs_test t_refs_test
    n_exp_test = length(X_inits_test)
    # Store predictions for each test experiment
    test_preds = Vector{Matrix{Float64}}(undef, n_exp_test)
    
    nc_test = ceil(Int, sqrt(length(X_inits_test)))
    nr_test = cld(length(X_inits_test), nc_test)
    pl_test = plot(
        layout = (nr_test, nc_test),
        size = (350 * nc_test, 300 * nr_test),
        legend = false,
        plot_title = "Held-out predictions",
    )
    for j in 1:n_exp_test
        sol = simulate_experiment(x, prob_optim, X_inits_test[j], t_refs_test[j])
        pred = hcat(sol.(t_refs_test[j])...)
        # Save predictions for this experiment
        test_preds[j] = pred
        plot!(pl_test[j], t_refs_test[j] .* 1440, pred[[3:4:32; 32], :]'; color = :red, title = "X₀ = $(X_inits_test[j])",
              xlabel = "Time [min]", ylabel = "TSS [g/m³]")
        plot!(pl_test[j], t_refs_test[j] .* 1440, Array(Y_refs_test[j][[3:4:32; 32], :])'; color = :black, linestyle = :dot, alpha = 1)
    end

    test_save = joinpath(path_project, "results", "$(output_prefix)/test_predictions.jld2")
    save(
        test_save,
        "X_inits", X_inits_test,
        "t", t_refs_test,
        "Y_pred", test_preds,
        "param_opt", x,
    )
    savefig(pl_test, joinpath(path_project, "results", "$(output_prefix)/test_predictions.png"))
    @info "Saved held-out predictions" test_save
end

# Initial parameters (default NN, optionally overwritten by files)
initial_params = Vector{Float64}(p_NN_init)
if validate
    initial_params = Vector{Float64}(load(validation_params_file)["param_opt"])
else
    for pfile in training_params_file
        if isfile(pfile)
            initial_params = Vector{Float64}(load(pfile)["param_opt"])
        end
    end
end
length(initial_params) == length(p_NN_init) || error("Loaded parameter vector has the wrong length")

initial_params_transformed = parameter_transform(initial_params)

initial_loss = evaluate_loss(initial_params_transformed)
@printf "Initial Loss: %g\n" initial_loss

mkpath(joinpath(path_project, "results"))
pl_init = plot_fit(initial_params_transformed, validate ? "Validation" : "Initial parameters")
savefig(pl_init, joinpath(path_project, "results/$(run_name)", validate ? "settler_validation.png" : "settler_initial_fit.png"))
pl_comp_init = plot_compression(
    [(validate ? "NN (loaded params)" : "NN (initial)", initial_params, :coral, :dash)],
    validate ? "Compression: validation NN vs. mechanistic" : "Compression: initial NN vs. mechanistic",
)
savefig(pl_comp_init, joinpath(path_project, "results/$(run_name)",
    validate ? "settler_validation_compression.png" : "settler_initial_compression.png"))

if validate
    #################################################################################################################
    @info "Saving Result"
    #################################################################################################################
    preds = [
        hcat(simulate_experiment(initial_params, prob_optim, X_inits[j], t_refs[j]).(t_refs[j])...)
        for j in 1:n_exp
    ]
    save(joinpath(path_project, "results/$(run_name)", "comp_sol_settler_ude.jld2"),
        "X_inits", X_inits, "t_refs", t_refs, "Y_pred", preds, "Y_refs", Y_refs)
    save_test_predictions(initial_params, "settler_validation")
    error("Don't continue to optimization if validating")
end

# Test autodiff
# grad_polyfd = zeros(size(initial_params_transformed)) # Initialise a vector for the gradient 
# @time PolyesterForwardDiff.threaded_gradient!(
#     x -> evaluate_loss(x), # the function whose gradient you want to calculate
#     grad_polyfd, # vector into which the calculated gradient is written
#     initial_params_transformed, # parameter values at which the gradient is evaluated.
#     ForwardDiff.Chunk(chunksize), # how many parameter derivatives ForwardDiff calculates together using chunked forward-mode automatic differentiation.
# );
# @printf "\tGradient norm: %g (any NaN: %s)\n" sqrt(sum(abs2, grad_polyfd)) string(any(isnan, grad_polyfd))

# sol = simulate_experiment(initial_params, prob_optim, X_inits[9], t_refs[9])
# @show sol.stats.nf sol.stats.naccept sol.stats.nreject

######################################################


# ## !! if training takes too long: consider adjoint training method (can be faster with little states)
## ##############################################################################################################
@info "Optimizing the parameters"
#################################################################################################################

# The optimizers to use and for how many epochs
# One epoch = one pass over all experiments. batchsize = number of experiments per gradient step
# (batchsize >= n_exp -> full batch)
# NOTE: losses are scaled by X_scale², so l_cutoff values are much smaller than in Florian's script
optimizers = [
    # n_epochs, l_cutoff, batchsize, optimizer
    25 1e-6 3 OptimizationOptimisers.Adam(0.01)
    # 200 1e-6 3 OptimizationOptimisers.Adam(0.01)
    # 300 1e-7 3 OptimizationOptimisers.Adam(0.001)
    # 400 1e-8 3 OptimizationOptimisers.Adam(0.0001)
    #  200 1e-8 9 OptimizationOptimisers.Adam(0.001)   # full batch
]

# Early stopping settings
n_mean_dec = 5 # if loss not lower than the mean of the last n losses -> stop

# Minibatching settings
batch_randomization_rng = Xoshiro(42)

# File paths for saving
loss_save = joinpath(path_project, "results", "$(run_name)/losses.csv")
intermediate_save = joinpath(path_project, "results", "$(run_name)/optim#optim_iter_#counter.jld2")
intermediate_optimizer_save = joinpath(path_project, "results", "$(run_name)/optim_#optim_final.jld2")
opt_param_save = joinpath(path_project, "results", "$(run_name)/final_result.jld2")
isfile(loss_save) && rm(loss_save)

# Bookkeeping variables
n_optimizers = size(optimizers, 1)
loss_star = copy(initial_loss)
params_star = copy(initial_params)
losses = Vector{Float64}[]

const last_iter = Ref(0)
const best_u    = Ref(Float64[])
const best_loss = Ref(Inf)
const last_u    = Ref(Float64[])

function callback(state, l, optim, n_optims, l_cutoff, n_epochs, n_batches, n_mean_dec)
    # skip the extra post-loop call (same or lower iter than before)
    state.iter <= last_iter[] && return false
    last_iter[] = state.iter

    # Only act at the end of an epoch
    state.iter % n_batches == 0 || return false
    epoch = state.iter ÷ n_batches

    # Full-data loss at end of epoch (for n_batches == 1 the batch loss is already that)
    #If there is more than one batch, it evaluates the loss over the full dataset using the current parameters.
    #if there is only one batch, it uses l directly.
    loss = n_batches > 1 ? evaluate_loss(state.u) : l 
    last_u[] = copy(state.u)
    if loss < best_loss[]
        best_loss[] = loss
        best_u[] = copy(state.u)
    end
    push!(losses[end], loss)
    println("Optimizer $(optim)/$(n_optims), Epoch $(epoch)/$(n_epochs): $loss")
    CSV.write(loss_save, DataFrame(loss = [loss]), writeheader = false, append = true) #This logs the loss for each completed epoch.

    #Save intermediate parameters every five epochs
    if mod(epoch, 5) == 0 
        xp = parameter_back_transform(state.u)
        save(
            Base.replace(intermediate_save, "#counter" => epoch, "#optim" => optim),
            "param_opt", xp,
            "loss", loss,
        )
    end
    # If loss is larger than mean over last ones -> stop
    length(losses[end]) > n_mean_dec + 1 && mean(losses[end][end-n_mean_dec-1:end-1]) < loss && return true
    # Terminate if loss is equal to last one (gradient probably vanished)
    length(losses[end]) > 1 && losses[end][end-1] == loss && return true
    # Terminate if loss is small
    return loss < l_cutoff
end

# The optimization function: p is the batch of experiment indices from the DataLoader
optf = OptimizationFunction(
    (u, batch_idx) -> evaluate_loss(u, batch_idx), #u=NN parameters, batch_idx=batch of experiment indices used to calculate the loss
    AutoPolyesterForwardDiff(; chunksize),
)
optprob = OptimizationProblem(optf, parameter_transform(params_star), all_idx)

for (i, (n_epochs, l_cutoff, batchsize, optim)) in enumerate(eachrow(optimizers))
    @info "Starting Optimization with $(i)th optimizer for $(n_epochs) epochs."
    #explicitly access the global best parameters and loss
    global params_star
    global loss_star
    #add an empty vector to losses for recording the loss history of this particular run
    push!(losses, Float64[])

    params = copy(params_star)

    @info "Loading Dataloader"
    dataloader = DataLoader(all_idx; batchsize = Int(batchsize), shuffle = true, rng = batch_randomization_rng)
    n_batches = length(dataloader)
    
    @info "Updating Optprob"
    optprob_updated = remake(optprob, u0 = copy(parameter_transform(params)), p = dataloader) #Remaking the problem ensures the next run starts from the latest best parameters.

    @info "Starting the optimization"
    last_iter[] = 0; best_loss[] = Inf
    best_u[]    = copy(parameter_transform(params_star))   # safe fallback
    last_u[]    = copy(best_u[])
    sol_opt = solve(
        optprob_updated, optim;
        callback = (state, l) -> callback(state, l, i, n_optimizers, l_cutoff, n_epochs, n_batches, n_mean_dec),
        epochs = Int(n_epochs),
    )

    @info "Saving the result"
    params   = parameter_back_transform(best_u[])   # or last_u[] if you want the final epoch
    loss_now = best_loss[]
    if loss_now < loss_star
        global params_star = copy(params)
        global loss_star = copy(loss_now)
    end

    save(
        Base.replace(intermediate_optimizer_save, "#optim" => i),
        "param_opt", params_star,
        "loss", loss_star,
    )
end

pl_final = plot_fit(parameter_transform(params_star), "Optimized parameters")
savefig(pl_final, joinpath(path_project, "results", "$(run_name)/final_fit.png"))
pl_comp_final = plot_compression(
    [
        ("NN (final)",   params_star,    :coral,  :dash),
    ],
    "Compression: mechanistic vs. NN (initial and final)",
)
savefig(pl_comp_final, joinpath(path_project, "results", "$(run_name)/final_compression.png"))
@printf "Final Loss: %g (initial: %g)\n" loss_star initial_loss
save_test_predictions(params_star, run_name)