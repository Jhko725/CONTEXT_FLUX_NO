# The conservation laws Trixi does not ship, each written to the pattern of Trixi's
# `InviscidBurgersEquation1D`: `flux`, `max_abs_speed_naive` (what `flux_lax_friedrichs` falls
# back to since Trixi 0.13), `max_abs_speeds` (the CFL step), `varnames`, `cons2prim`. The
# flux formulas are those of `simulations/pde/{cubic,sine_1d,shallow_water}.py`.

# --------------------------------------------------------------------------------------------
# Cubic flux, 1-D and 2-D (the same f along both axes)

"""
    CubicFluxEquation1D(a, b, c)

`u_t + f(u)_x = 0` with `f(u) = a u³ + b u² + c u` (`CubicFlux1D` in Python).
"""
struct CubicFluxEquation1D{RealT <: Real} <: Trixi.AbstractEquations{1, 1}
    a::RealT
    b::RealT
    c::RealT
end

"""
    CubicFluxEquation2D(a, b, c)

`u_t + f(u)_x + f(u)_y = 0` with the cubic `f` of [`CubicFluxEquation1D`](@ref) along both
axes (`CubicFlux2D` in Python).
"""
struct CubicFluxEquation2D{RealT <: Real} <: Trixi.AbstractEquations{2, 1}
    a::RealT
    b::RealT
    c::RealT
end

const CubicFluxEquation = Union{CubicFluxEquation1D, CubicFluxEquation2D}

@inline cubic(eq::CubicFluxEquation, u) = ((eq.a * u + eq.b) * u + eq.c) * u
@inline dcubic(eq::CubicFluxEquation, u) = (3 * eq.a * u + 2 * eq.b) * u + eq.c

"""
    max_abs_dflux(u_ll, u_rr, equations)

`max |f'(u)|` over `u ∈ [u_ll, u_rr]`, the exact local Lax–Friedrichs bound (D-WS). The
endpoint maximum is exact for a convex flux; the non-convex fluxes here also have `f'`
checked at its interior extremum. [LeVeque 2002 §12.5: the local LF flux is monotone only
when the speed bounds `|f'|` on the whole interval.]
"""
@inline function max_abs_dflux(u_ll::Real, u_rr::Real, eq::CubicFluxEquation)
    lo, hi = minmax(u_ll, u_rr)
    s = max(abs(dcubic(eq, lo)), abs(dcubic(eq, hi)))
    if !iszero(eq.a)
        vertex = -eq.b / (3 * eq.a)  # the extremum of the quadratic f'
        if lo <= vertex <= hi
            s = max(s, abs(dcubic(eq, vertex)))
        end
    end
    return s
end

Trixi.varnames(::typeof(Trixi.cons2cons), ::CubicFluxEquation) = ("u",)
Trixi.varnames(::typeof(Trixi.cons2prim), ::CubicFluxEquation) = ("u",)
@inline Trixi.cons2prim(u, ::CubicFluxEquation) = u

@inline Trixi.flux(u, orientation::Integer, eq::CubicFluxEquation) = SVector(cubic(eq, u[1]))

# `StructuredMesh{2}` calls the surface and FV-subcell fluxes with a (contravariant) normal
# vector: F · n = (n₁ + n₂) f(u), both components being f.
@inline function Trixi.flux(u, normal_direction::AbstractVector, eq::CubicFluxEquation2D)
    return SVector((normal_direction[1] + normal_direction[2]) * cubic(eq, u[1]))
end

@inline function Trixi.max_abs_speed_naive(u_ll, u_rr, orientation::Integer,
                                           eq::CubicFluxEquation)
    return max_abs_dflux(u_ll[1], u_rr[1], eq)
end

@inline function Trixi.max_abs_speed_naive(u_ll, u_rr, normal_direction::AbstractVector,
                                           eq::CubicFluxEquation2D)
    return abs(normal_direction[1] + normal_direction[2]) * max_abs_dflux(u_ll[1], u_rr[1], eq)
end

@inline Trixi.max_abs_speeds(u, eq::CubicFluxEquation1D) = (abs(dcubic(eq, u[1])),)

@inline function Trixi.max_abs_speeds(u, eq::CubicFluxEquation2D)
    s = abs(dcubic(eq, u[1]))
    return (s, s)
end

# --------------------------------------------------------------------------------------------
# Sine flux

"""
    SineFluxEquation1D(a, b)

`u_t + (a sin(b u))_x = 0` (`SineFlux1D` in Python).
"""
struct SineFluxEquation1D{RealT <: Real} <: Trixi.AbstractEquations{1, 1}
    a::RealT
    b::RealT
end

@inline sine_flux(eq::SineFluxEquation1D, u) = eq.a * sin(eq.b * u)
@inline dsine_flux(eq::SineFluxEquation1D, u) = eq.a * eq.b * cos(eq.b * u)

# |f'| = |a b| |cos(b u)| reaches |a b| wherever b u is a multiple of π.
@inline function max_abs_dflux(u_ll::Real, u_rr::Real, eq::SineFluxEquation1D)
    lo, hi = minmax(eq.b * u_ll, eq.b * u_rr)
    if ceil(lo / π) <= floor(hi / π)
        return abs(eq.a * eq.b)
    end
    return max(abs(dsine_flux(eq, u_ll)), abs(dsine_flux(eq, u_rr)))
end

Trixi.varnames(::typeof(Trixi.cons2cons), ::SineFluxEquation1D) = ("u",)
Trixi.varnames(::typeof(Trixi.cons2prim), ::SineFluxEquation1D) = ("u",)
@inline Trixi.cons2prim(u, ::SineFluxEquation1D) = u

@inline Trixi.flux(u, orientation::Integer, eq::SineFluxEquation1D) = SVector(sine_flux(eq, u[1]))

@inline function Trixi.max_abs_speed_naive(u_ll, u_rr, orientation::Integer,
                                           eq::SineFluxEquation1D)
    return max_abs_dflux(u_ll[1], u_rr[1], eq)
end

@inline Trixi.max_abs_speeds(u, eq::SineFluxEquation1D) = (abs(dsine_flux(eq, u[1])),)

# --------------------------------------------------------------------------------------------
# Parametrized shallow water

"""
    ParametrizedShallowWaterEquations1D(gamma, alpha, beta, h_floor)

The three-parameter shallow-water-like system of `simulations/pde/shallow_water.py`: state
`(h, m)`, flux `(γ m, α m²/h + ½ β h²)`; `γ = α = 1, β = g` is shallow water. `h_floor`
floors the depth in `m/h` only, as the Python Riemann solver does; it is numerical, not
physical, and is not a Well scalar.
"""
struct ParametrizedShallowWaterEquations1D{RealT <: Real} <: Trixi.AbstractEquations{1, 2}
    gamma::RealT
    alpha::RealT
    beta::RealT
    h_floor::RealT
end

Trixi.varnames(::typeof(Trixi.cons2cons), ::ParametrizedShallowWaterEquations1D) = ("h", "m")
Trixi.varnames(::typeof(Trixi.cons2prim), ::ParametrizedShallowWaterEquations1D) = ("h", "v")

@inline function Trixi.cons2prim(u, eq::ParametrizedShallowWaterEquations1D)
    h, m = u
    return SVector(h, m / max(h, eq.h_floor))
end

@inline function Trixi.flux(u, orientation::Integer, eq::ParametrizedShallowWaterEquations1D)
    h, m = u
    h_safe = max(h, eq.h_floor)
    return SVector(eq.gamma * m, eq.alpha * m^2 / h_safe + 0.5 * eq.beta * h^2)
end

# Eigenvalues of the flux Jacobian, λ± = α v ± sqrt(α (α − γ) v² + γ β h)
# (`shallow_water.py` docstring); the radicand is clipped at zero where hyperbolicity is lost.
@inline function eigenvalues(u, eq::ParametrizedShallowWaterEquations1D)
    h, m = u
    h_safe = max(h, eq.h_floor)
    v = m / h_safe
    c = sqrt(max(eq.alpha * (eq.alpha - eq.gamma) * v^2 + eq.gamma * eq.beta * h_safe, 0))
    return (eq.alpha * v - c, eq.alpha * v + c)
end

@inline function Trixi.max_abs_speeds(u, eq::ParametrizedShallowWaterEquations1D)
    λ_minus, λ_plus = eigenvalues(u, eq)
    return (max(abs(λ_minus), abs(λ_plus)),)
end

@inline function Trixi.max_abs_speed_naive(u_ll, u_rr, orientation::Integer,
                                           eq::ParametrizedShallowWaterEquations1D)
    return max(Trixi.max_abs_speeds(u_ll, eq)[1], Trixi.max_abs_speeds(u_rr, eq)[1])
end

# --------------------------------------------------------------------------------------------
# Per-family choices of the blended volume integral (D-HG, D-VF)

"""
    first_variable(u, equations)

`u[1]`: the state of a scalar law, the depth of the parametrized shallow water.
"""
@inline first_variable(u, equations) = u[1]

"""
    indicator_variable(equations)

The quantity the Hennemann–Gassner indicator inspects (D-HG): the state itself for the
scalar laws, the depth for the parametrized shallow water, `density_pressure` for Euler.
"""
indicator_variable(::Trixi.AbstractEquations{<:Any, 1}) = first_variable
indicator_variable(::ParametrizedShallowWaterEquations1D) = first_variable
indicator_variable(::Trixi.CompressibleEulerEquations2D) = Trixi.density_pressure

"""
    volume_flux_dg(equations)

The two-point flux of the DG part of the blended volume integral (D-VF): `flux_central`,
the standard weak form, for every family. Adding a method for one family switches it to an
entropy-conservative flux (e.g. `flux_ranocha` for Euler); bump the package version with it
(D-PROV).
"""
volume_flux_dg(::Trixi.AbstractEquations) = Trixi.flux_central

# --------------------------------------------------------------------------------------------
# The name → constructor table (D-SPEC)

"""
    parameters(p, names...)

The values of `p` at `names`, as `Float64`s in that order; an `ArgumentError` if a name is
missing or `p` holds a key not in `names`.
"""
function parameters(p::AbstractDict, names::Vararg{String})
    d = Dict{String, Float64}(String(k) => Float64(v) for (k, v) in p)
    missing_names = [n for n in names if !haskey(d, n)]
    extra_names = sort!([k for k in keys(d) if !(k in names)])
    if !isempty(missing_names) || !isempty(extra_names)
        throw(ArgumentError("expected parameters $(collect(names)); missing $(missing_names), " *
                            "unexpected $(extra_names)"))
    end
    return ntuple(i -> d[names[i]], length(names))
end

"""`TrixiSpec.equations` name → constructor from the parameter dict."""
const EQUATIONS = Dict{String, Function}(
    "CubicFluxEquation1D" => p -> CubicFluxEquation1D(parameters(p, "a", "b", "c")...),
    "CubicFluxEquation2D" => p -> CubicFluxEquation2D(parameters(p, "a", "b", "c")...),
    "SineFluxEquation1D" => p -> SineFluxEquation1D(parameters(p, "a", "b")...),
    "ParametrizedShallowWaterEquations1D" =>
        p -> ParametrizedShallowWaterEquations1D(parameters(p, "gamma", "alpha", "beta",
                                                            "h_floor")...),
    "InviscidBurgersEquation1D" => p -> (parameters(p); Trixi.InviscidBurgersEquation1D()),
    "CompressibleEulerEquations2D" =>
        p -> Trixi.CompressibleEulerEquations2D(parameters(p, "gamma")...),
)

function build_equations(name::AbstractString, p::AbstractDict)
    constructor = get(EQUATIONS, String(name)) do
        throw(ArgumentError("unknown equations $(repr(String(name))); known: " *
                            "$(sort!(collect(keys(EQUATIONS))))"))
    end
    return constructor(p)
end
