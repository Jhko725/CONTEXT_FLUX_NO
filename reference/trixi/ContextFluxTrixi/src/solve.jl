"""The surface-flux names `TrixiBackend` may pass."""
const SURFACE_FLUXES = Dict{String, Any}("lax_friedrichs" => Trixi.flux_lax_friedrichs)

# Hennemann–Gassner constants (D-HG) [Hennemann et al. 2021, JCP 426:109935 §4; Trixi's
# elixir defaults]. Fixed here, not `TrixiBackend` options; a change bumps the package version.
const ALPHA_MAX = 0.5
const ALPHA_MIN = 0.001
const ALPHA_SMOOTH = true

"""
    node_coordinates(x_spans, Nxs, polydeg) -> NTuple{d, Array{Float64}}

**Arguments:**
- `x_spans`: `[d, 2]`, lower and upper bound per axis.
- `Nxs`: `[d]`, cells (= DG elements) per axis.
- `polydeg`: polynomial degree; `polydeg + 1` Lobatto–Legendre nodes per axis per element.

**Returns:** one coordinate array per axis, each of shape `(polydeg + 1,)^d × Nxs` — node
indices first, then element indices — in Trixi's own element and node order (x fastest;
`cache.elements.node_coordinates[:, i, j, e]` has `x` from node `i` and `y` from node `j`),
so [`solve`](@ref) takes the initial state evaluated on these without a permutation.
"""
function node_coordinates(x_spans::AbstractMatrix, Nxs::AbstractVector{<:Integer},
                          polydeg::Integer)
    d = length(Nxs)
    size(x_spans) == (d, 2) ||
        throw(ArgumentError("x_spans must be [$(d), 2], got $(size(x_spans))"))
    polydeg >= 1 || throw(ArgumentError("polydeg must be >= 1"))
    all(>=(1), Nxs) || throw(ArgumentError("every Nx must be >= 1"))
    nodes = Trixi.LobattoLegendreBasis(polydeg).nodes
    n = polydeg + 1
    shape = (ntuple(_ -> n, d)..., Tuple(Int.(Nxs))...)
    return ntuple(d) do k
        lo, hi = Float64(x_spans[k, 1]), Float64(x_spans[k, 2])
        dx = (hi - lo) / Nxs[k]
        X = Array{Float64}(undef, shape)
        for I in CartesianIndices(X)
            node, cell = I[k], I[d + k]
            X[I] = lo + (cell - 0.5) * dx + 0.5 * dx * nodes[node]
        end
        X
    end
end

# `SemidiscretizationHyperbolic` needs an initial-condition function; the state is overwritten
# with `u0_nodes` right after, so this only fixes the element type.
initial_condition_zero(x, t, equations) = zero(SVector{Trixi.nvariables(equations), Float64})

"""
    element_means(u, weights) -> Matrix [n_eqn, n_elements]

Element means by LGL quadrature of a wrapped state `u` (`[n_eqn, nodes..., elements]`);
`weights` sum to 2 per axis.
"""
function element_means(u::AbstractArray{<:Any, 3}, weights)
    nv, n, ne = size(u)
    means = zeros(nv, ne)
    for e in 1:ne, i in 1:n, v in 1:nv
        means[v, e] += weights[i] * u[v, i, e]
    end
    return means .* 0.5
end

function element_means(u::AbstractArray{<:Any, 4}, weights)
    nv, n, _, ne = size(u)
    means = zeros(nv, ne)
    for e in 1:ne, j in 1:n, i in 1:n, v in 1:nv
        means[v, e] += weights[i] * weights[j] * u[v, i, j, e]
    end
    return means .* 0.25
end

"""
    solve(equations, parameters, u0_nodes, x_spans, Nxs, t_out,
          polydeg, cfl, surface_flux, shock_capturing) -> Array{Float64}

Integrate one trajectory and return element means.

**Arguments:**
- `equations::AbstractString`: a key of `EQUATIONS` (`TrixiSpec.equations`).
- `parameters::AbstractDict`: its keyword arguments, name → value.
- `u0_nodes`: `[n_eqn, (polydeg + 1)^d..., Nxs...]`, the initial state on
  [`node_coordinates`](@ref).
- `x_spans`, `Nxs`, `polydeg`: as for `node_coordinates`.
- `t_out::AbstractVector`: strictly increasing output times; the integration runs from
  `t_out[1]` to `t_out[end]`.
- `cfl::Real`: the CFL number of Trixi's `StepsizeCallback`.
- `surface_flux::AbstractString`: a key of `SURFACE_FLUXES`.
- `shock_capturing::Bool`: Hennemann–Gassner blending (D-HG) or the plain weak form.

**Returns:** `[length(t_out), n_eqn, Nxs...]`, the element means by LGL quadrature at every
output time; the first equals the mean of `u0_nodes`.

Throws `ArgumentError` for a mis-specified problem and [`SolveFailed`](@ref) for a failed
integration (D-FAIL).
"""
function solve(equations::AbstractString, parameters::AbstractDict, u0_nodes::AbstractArray,
               x_spans::AbstractMatrix, Nxs::AbstractVector{<:Integer}, t_out::AbstractVector,
               polydeg::Integer, cfl::Real, surface_flux::AbstractString,
               shock_capturing::Bool)
    eq = build_equations(equations, parameters)
    d = length(Nxs)
    Trixi.ndims(eq) == d ||
        throw(ArgumentError("$(equations) is $(Trixi.ndims(eq))-D, the grid is $(d)-D"))
    size(x_spans) == (d, 2) ||
        throw(ArgumentError("x_spans must be [$(d), 2], got $(size(x_spans))"))
    polydeg >= 1 || throw(ArgumentError("polydeg must be >= 1"))
    cfl > 0 || throw(ArgumentError("cfl must be > 0"))
    n = polydeg + 1
    nv = Trixi.nvariables(eq)
    cells = Tuple(Int.(Nxs))
    expected = (nv, ntuple(_ -> n, d)..., cells...)
    size(u0_nodes) == expected ||
        throw(ArgumentError("u0_nodes has shape $(size(u0_nodes)), expected $(expected)"))
    t = Vector{Float64}(t_out)
    (length(t) >= 1 && all(t[i] < t[i + 1] for i in 1:(length(t) - 1))) ||
        throw(ArgumentError("t_out must be non-empty and strictly increasing, got $(t)"))
    flux_surface = get(SURFACE_FLUXES, String(surface_flux)) do
        throw(ArgumentError("unknown surface flux $(repr(String(surface_flux))); known: " *
                            "$(sort!(collect(keys(SURFACE_FLUXES))))"))
    end

    mesh = Trixi.StructuredMesh(cells, Tuple(Float64.(x_spans[:, 1])),
                                Tuple(Float64.(x_spans[:, 2])); periodicity = true)
    basis = Trixi.LobattoLegendreBasis(polydeg)
    volume_integral = if shock_capturing
        indicator = Trixi.IndicatorHennemannGassner(eq, basis; alpha_max = ALPHA_MAX,
                                                    alpha_min = ALPHA_MIN,
                                                    alpha_smooth = ALPHA_SMOOTH,
                                                    variable = indicator_variable(eq))
        Trixi.VolumeIntegralShockCapturingHG(indicator; volume_flux_dg = volume_flux_dg(eq),
                                             volume_flux_fv = flux_surface)
    else
        Trixi.VolumeIntegralWeakForm()
    end
    solver = Trixi.DGSEM(basis, flux_surface, volume_integral)
    semi = Trixi.SemidiscretizationHyperbolic(mesh, eq, initial_condition_zero, solver;
                                              boundary_conditions = Trixi.boundary_condition_periodic)

    # The initial state goes straight into the ODE state in Trixi's layout
    # [n_eqn, nodes..., elements]: Julia's column-major reshape of (Nxs...) is Trixi's
    # LinearIndices element order.
    ode = Trixi.semidiscretize(semi, (t[1], t[end]))
    Trixi.wrap_array(ode.u0, semi) .=
        reshape(Array{Float64}(u0_nodes), nv, ntuple(_ -> n, d)..., prod(cells))

    weights = basis.weights
    out = Array{Float64}(undef, length(t), nv, cells...)
    store!(k, u_ode) = begin
        means = element_means(Trixi.wrap_array(u_ode, semi), weights)
        all(isfinite, means) || throw(SolveFailed("non-finite state at t = $(t[k])"))
        selectdim(out, 1, k) .= reshape(means, nv, cells...)
    end
    store!(1, ode.u0)
    length(t) == 1 && return out

    # Each output time is saved by exactly one mechanism: `save_start` the first, `saveat` the
    # interior ones, `save_end` the last (a `saveat` entry at the end time is not saved).
    sol = try
        ODE.solve(ode, ODE.SSPRK33(); dt = 1.0, save_everystep = false, save_start = true,
                  save_end = true, saveat = t[2:(end - 1)], tstops = t,
                  callback = SciMLBase.CallbackSet(Trixi.StepsizeCallback(cfl = cfl)))
    catch err
        err isa InterruptException && rethrow()
        throw(SolveFailed("integration raised " * sprint(showerror, err)))
    end
    SciMLBase.successful_retcode(sol) ||
        throw(SolveFailed("integration stopped with return code $(sol.retcode)"))
    (length(sol.t) == length(t) && all(isapprox.(sol.t, t; atol = 1e-12))) ||
        throw(SolveFailed("saved states at $(sol.t), output times $(t)"))
    for k in 2:length(t)
        store!(k, sol.u[k])
    end
    return out
end

"""
    versions() -> Dict{String, String}

`"trixi"`, `"julia"`, `"threads"`, and `"context_flux_trixi"` — this package's version, which
pins the D-VF/D-WS/D-HG constants — for `TrixiBackend.describe()` (D-PROV).
"""
function versions()
    return Dict{String, String}(
        "trixi" => string(pkgversion(Trixi)),
        "julia" => string(VERSION),
        "threads" => string(Threads.nthreads()),
        "context_flux_trixi" => string(pkgversion(@__MODULE__)),
    )
end
