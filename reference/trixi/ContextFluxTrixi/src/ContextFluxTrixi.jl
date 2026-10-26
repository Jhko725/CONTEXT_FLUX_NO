"""
    ContextFluxTrixi

The Julia side of `context_flux_no.simulations.trixi.TrixiBackend` (ADR-0004 amended,
ADR-0011; change document `2026-10-25-phase2a-trixi-backend`): Trixi.jl DGSEM on a
`StructuredMesh` with one element per finite-volume cell, Lax–Friedrichs surface flux,
Hennemann–Gassner shock capturing, SSPRK33 at a fixed CFL number, element means out.

The interface is three functions and one exception, specified in `simulations/trixi.py`'s
module docstring: [`node_coordinates`](@ref), [`solve`](@ref), [`versions`](@ref),
[`SolveFailed`](@ref). Arguments arrive as what PythonCall makes of Python objects (`PyArray`,
`PyDict`, `String`, …), so every function is typed on abstract containers and copies what it
keeps. Shapes are stated in index order, which both languages share.
"""
module ContextFluxTrixi

using Trixi
using StaticArrays: SVector
import OrdinaryDiffEqSSPRK as ODE  # `import`: its re-exported `solve` would shadow ours
import SciMLBase

export node_coordinates, solve, versions, SolveFailed

"""
    SolveFailed(msg)

The integration failed: a non-success return code, a non-finite state, or an error raised
while stepping. `TrixiBackend` turns it into `SolverFailure("trixi", msg)`. Errors in setting
the problem up (unknown equations, bad parameters, wrong shapes) are `ArgumentError`s and are
not wrapped (D-FAIL).
"""
struct SolveFailed <: Exception
    msg::String
end

Base.showerror(io::IO, e::SolveFailed) = print(io, "SolveFailed: ", e.msg)

include("equations.jl")
include("solve.jl")

end # module
