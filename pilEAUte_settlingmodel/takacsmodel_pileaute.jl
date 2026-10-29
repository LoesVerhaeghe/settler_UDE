using CSV
using DataFrames
using DifferentialEquations
using Dates
using Plots

"""
Based on code by Ward Quaghebeur.
Takacs settling model with 10 layers and only TSS considered as component.
The Takacs settling velocity function is used to calculate the settling
fluxes per layer. The Stenstrom flux constraint is applied to the settling
fluxes per layer.

Layer indexing (1-based, Julia convention):
  layer 1  = bottom layer
  layer 10 = top layer
"""
function settler_equations(y, p, t)
    # Unpack influent and parameters
    X_f, flowrates, par = p         # X_f: influent TSS concentration
    Q_f, Q_w, Q_r = flowrates       # influent flow, waste flow, recycle flow

    # Unpack parameters
    v0_max    = par[1]
    v0        = par[2]
    r_h       = par[3]
    r_p       = par[4]
    f_ns      = par[5]
    X_t       = par[6]
    area      = par[7]
    h         = par[8]
    feedlayer = Int(par[9])         

    v_in = Q_f / area
    # Q_u = Q_r + Q_w
    Q_u = Q_r                       # for pileaute data
    Q_e = Q_f - Q_u

    v_up = Q_e / area
    v_dn = Q_u / area

    # Velocities and fluxes
    vs  = zeros(10)
    Js  = zeros(10)
    Jsc = zeros(10)

    # Takacs settling velocity calculation per layer
    for i in 1:10
        vs[i] = max(0.0, min(v0_max,
                    v0 * (exp(-r_h * (y[i] - f_ns * X_f)) -
                          exp(-r_p * (y[i] - f_ns * X_f)))))
    end

    # Calculate settling fluxes per layer
    for i in 1:10
        Js[i] = vs[i] * y[i]        # settling flux = v_s(X) * X
        if i > feedlayer            # Stenstrom settling flux constraint
            if y[i-1] > X_t
                Jsc[i] = min(Js[i], Js[i-1])
            else
                Jsc[i] = Js[i]
            end
        end
    end

    # Differentials
    dy = zeros(10)

    # Suspended solids (TSS)
    for i in 1:10
        if i == feedlayer           # feed layer
            dy[i] = (v_in * X_f + Jsc[i+1] - (v_up + v_dn) * y[i] -
                     min(Js[i], Js[i-1])) / h
        elseif i == 1                # bottom layer
            dy[i] = (v_dn * (y[i+1] - y[i]) + min(Js[i], Js[i+1])) / h
        elseif i == 10                # top layer
            dy[i] = (v_up * (y[i-1] - y[i]) - Jsc[i]) / h
        elseif i < feedlayer          # below feed layer
            dy[i] = (v_dn * (y[i+1] - y[i]) + min(Js[i], Js[i+1]) -
                     min(Js[i], Js[i-1])) / h
        elseif i > feedlayer          # above feed layer
            dy[i] = (v_up * (y[i-1] - y[i]) + Jsc[i+1] - Jsc[i]) / h
        end
    end

    # For soluble components, settling fluxes are not considered, only the
    # flow velocities and influent concentrations. 
    # for components 11:20 (S_I), 21:30 (S_S), 31:40 (S_O), 41:50 (S_NO),
    # 51:60 (S_NH), 61:70 (S_ND), 71:80 (S_ALK), each following the pattern:
    #
    # for i in (feedlayer+OFFSET-9):(feedlayer+OFFSET+9)
    #     if i < feedlayer + OFFSET
    #         dy[i] = (v_dn * (y[i+1] - y[i])) / h
    #     elseif i > feedlayer + OFFSET
    #         dy[i] = (v_up * (y[i-1] - y[i])) / h
    #     else
    #         dy[i] = (v_in * u[k] - (v_dn + v_up) * y[i]) / h
    #     end
    # end

    return dy
end



function settler_pileaute(y, dt, par_settler, feed_MLSS_conc, Q_inflow, Q_waste, Q_recycle)
    """
    Settler model with 10 layers and 1 component (MLSS).
    """
    flowrates = (Q_inflow, Q_waste, Q_recycle)
    p = (feed_MLSS_conc, flowrates, par_settler)

    prob = ODEProblem(settler_equations, y, (dt[1], dt[2]), p)
    sol = solve(prob, Tsit5()) # Tsit5 = Tsitouras 5/4 Runge–Kutta method, Runge–Kutta method designed for solving non-stiff ordinary differential equations (ODEs).

    X_layers = sol.u[end]           # final state of the layers after integration

    X_under = X_layers[1]
    X_eff = X_layers[end]

    return X_layers, X_under, X_eff
end

#############################################
# model with constant parameters
#############################################

function main()
    df = CSV.read("data/influent_file.csv", DataFrame)

    # Initial concentration profile (from WEST steady state simulation)
    X_layers = [2094.8, 208.06, 208.06, 208.06, 208.06, 22.31, 13.04, 9.8, 6.92, 5.33]

    X_under = Float64[]
    X_eff = Float64[]
    X_layers_all = Vector{Vector{Float64}}()

    ### PARAMETERS SETTLER ###
    v0_max = 250
    v0 = 286.56
    r_h = 0.00087
    r_p = 0.002861
    f_ns = 0.00228
    X_t = 3000
    area = 1.2
    h = 2.3
    feedlayer = 6                    # Python feedlayer=5 (0-based) -> Julia 6 (1-based)

    par_settler = [v0_max, v0, r_h, r_p, f_ns, X_t, area, h, feedlayer]

    for i in 1:(nrow(df) - 1)
        dt = (df.time[i], df.time[i+1])

        MLSS = df.MLSS[i]
        Q_feed = df.Q_feed[i]
        Q_RAS = df.Q_RAS[i]
        # Q_WAS = df.Q_WAS[i]
        Q_WAS = 0.0

        X_layers, Xu, Xe = settler_pileaute(
            X_layers, dt, par_settler, MLSS, Q_feed, Q_WAS, Q_RAS,
        )

        push!(X_layers_all, X_layers) # similar to append
        push!(X_under, Xu)
        push!(X_eff, Xe)
    end

    X_layers_arr = permutedims(reduce(hcat, X_layers_all))  # rows = time, cols = layers
    layer_5_over_time = X_layers_arr[:, 6]                   # Python index 5 -> Julia 6

    lab_measurements = CSV.read("data/lab_measurements_raw.csv", DataFrame)
    lab_measurements.time = DateTime.(lab_measurements.time)

    # Convert to days since 2023-10-01
    ref_time = DateTime(2023, 10, 1)
    lab_measurements.time_days =
        [Dates.value(t - ref_time) / (1000 * 60 * 60 * 24) for t in lab_measurements.time]

    p1 = plot(df.time, df.TSS_RAS, label="measured underflow concentration")
    plot!(p1, df.time[2:end], X_under, label="model underflow concentration")
    savefig("underflow_comparison.png")

    p2 = plot(df.time[2:end], layer_5_over_time, label="Layer 5", ylim=(0, 1000))
    savefig("layer5.png")
    p3 = plot(df.time[2:end], X_eff, label="model effluent concentration", ylim=(0, 120))
    plot!(p3, lab_measurements.time_days, lab_measurements.TSSeffluent,
          label="Measured effluent", ylim=(0, 120))
    savefig("effluent_comparison.png")
    
    display(p1)
    display(p2)
    display(p3)

    return X_layers_all, X_under, X_eff
end

main()