# Oracles for the Julia side of `TrixiBackend` (change document 2026-10-25-phase2a-trixi-backend,
# ledger 4d). Run with `Pkg.test("ContextFluxTrixi")` or `scripts/setup_julia.py --test`.

using Test
using ContextFluxTrixi
using ContextFluxTrixi: CubicFluxEquation1D, CubicFluxEquation2D, SineFluxEquation1D,
                        ParametrizedShallowWaterEquations1D, max_abs_dflux,
                        initial_condition_zero, indicator_variable, volume_flux_dg
using Trixi
using StaticArrays: SVector
using Random: MersenneTwister

const X1 = [0.0 1.0]            # x_spans of a 1-D grid
const X2 = [0.0 1.0; -1.0 1.0]  # x_spans of a 2-D grid
const POLYDEG = 3

"Means of `u_nodes` (`[n_eqn, nodes..., cells...]`) over each element by LGL quadrature."
function lgl_means(u_nodes, polydeg)
    w = LobattoLegendreBasis(polydeg).weights
    if ndims(u_nodes) == 3
        nv, n, Nx = size(u_nodes)
        return [sum(w[i] * u_nodes[v, i, c] for i in 1:n) / 2 for v in 1:nv, c in 1:Nx]
    end
    nv, n, _, Nx, Ny = size(u_nodes)
    return [sum(w[i] * w[j] * u_nodes[v, i, j, cx, cy] for i in 1:n, j in 1:n) / 4
            for v in 1:nv, cx in 1:Nx, cy in 1:Ny]
end

"A scalar `f(x...)` on the node arrays → `[1, nodes..., cells...]`."
nodal(f, coords) = reshape(map(f, coords...), 1, size(coords[1])...)

"An `SVector`-valued `f(x...)` on the node arrays → `[n_eqn, nodes..., cells...]`."
function nodal_system(f, coords)
    nv = length(f(getindex.(coords, 1)...))
    out = Array{Float64}(undef, nv, size(coords[1])...)
    for I in CartesianIndices(coords[1])
        out[:, Tuple(I)...] .= f(getindex.(coords, I)...)
    end
    return out
end

# The goldens' scalar initial condition (`tests/unit/make_goldens_simulations.py`).
golden_ic(x) = 0.5 + 0.4 * sin(2π * x) + 0.1 * cos(6π * x)
const CUBIC = Dict("a" => 1.0, "b" => -0.5, "c" => 0.25)
const LF = "lax_friedrichs"

@testset "ContextFluxTrixi" begin
    @testset "node_coordinates: shapes, cells, Trixi's own order (D-NODES)" begin
        for (xs, Nxs, polydeg) in ((X1, [8], 3), (X2, [4, 6], 2))
            d = length(Nxs)
            n = polydeg + 1
            coords = node_coordinates(xs, Nxs, polydeg)
            @test length(coords) == d
            @test all(size(c) == (ntuple(_ -> n, d)..., Nxs...) for c in coords)
            for k in 1:d
                dx = (xs[k, 2] - xs[k, 1]) / Nxs[k]
                inside = true
                for I in CartesianIndices(coords[k])
                    lo = xs[k, 1] + (I[d + k] - 1) * dx
                    inside &= lo - 1e-12 <= coords[k][I] <= lo + dx + 1e-12
                end
                @test inside
            end
            eq = d == 1 ? CubicFluxEquation1D(1.0, 0.0, 0.0) : CubicFluxEquation2D(1.0, 0.0, 0.0)
            mesh = StructuredMesh(Tuple(Nxs), Tuple(xs[:, 1]), Tuple(xs[:, 2]);
                                  periodicity = true)
            semi = SemidiscretizationHyperbolic(mesh, eq, initial_condition_zero,
                                                DGSEM(polydeg, flux_lax_friedrichs);
                                                boundary_conditions = boundary_condition_periodic)
            trixi_nodes = semi.cache.elements.node_coordinates  # [d, nodes..., elements]
            for k in 1:d
                ours = reshape(coords[k], ntuple(_ -> n, d)..., prod(Nxs))
                @test isapprox(ours, selectdim(trixi_nodes, 1, k); atol = 1e-12)
            end
        end
        # LGL quadrature of x² is the exact cell average (exact for polydeg >= 2)
        coords = node_coordinates(X1, [4], 2)
        means = lgl_means(coords[1] .^ 2 |> c -> reshape(c, 1, size(c)...), 2)
        exact = [((c / 4)^3 - ((c - 1) / 4)^3) / 3 * 4 for c in 1:4]
        @test isapprox(vec(means), exact; atol = 1e-14)
        @test_throws ArgumentError node_coordinates(X1, [4, 4], 2)
        @test_throws ArgumentError node_coordinates(X1, [4], 0)
    end

    @testset "exact Lax–Friedrichs bound (D-WS)" begin
        rng = MersenneTwister(1)
        cases = (
            (CubicFluxEquation1D(1.0, -0.5, 0.25), u -> (3 * 1.0 * u + 2 * -0.5) * u + 0.25),
            (CubicFluxEquation1D(-2.0, 1.0, 0.1), u -> (3 * -2.0 * u + 2 * 1.0) * u + 0.1),
            (SineFluxEquation1D(1.0, 2.0), u -> 1.0 * 2.0 * cos(2.0 * u)),
            (SineFluxEquation1D(0.7, 5.0), u -> 0.7 * 5.0 * cos(5.0 * u)),
        )
        for (eq, dflux) in cases
            lower_ok = upper_ok = true
            for _ in 1:200
                u_ll, u_rr = 3 .* rand(rng, 2) .- 1.5
                sampled = maximum(abs ∘ dflux, range(min(u_ll, u_rr), max(u_ll, u_rr);
                                                     length = 2001))
                bound = max_abs_dflux(u_ll, u_rr, eq)
                lower_ok &= bound >= sampled - 1e-12   # a bound
                upper_ok &= bound <= sampled + 1e-3    # and a tight one
            end
            @test lower_ok
            @test upper_ok
            # degenerate interval: the pointwise speed
            @test max_abs_dflux(0.3, 0.3, eq) ≈ abs(dflux(0.3))
        end
        eq2 = CubicFluxEquation2D(1.0, -0.5, 0.25)
        u, normal = SVector(0.3), SVector(0.6, -0.8)
        @test flux(u, normal, eq2) ≈ normal[1] * flux(u, 1, eq2) + normal[2] * flux(u, 2, eq2)
        @test Trixi.max_abs_speed_naive(SVector(0.1), SVector(0.4), normal, eq2) ≈
              abs(normal[1] + normal[2]) * max_abs_dflux(0.1, 0.4, eq2)
        @test Trixi.max_abs_speeds(u, eq2) == (abs((3 * 0.3 - 1.0) * 0.3 + 0.25),
                                               abs((3 * 0.3 - 1.0) * 0.3 + 0.25))
    end

    @testset "flux oracles: Trixi's Burgers and shallow water" begin
        burgers = InviscidBurgersEquation1D()
        cubic = CubicFluxEquation1D(0.0, 0.5, 0.0)
        for u in (SVector(-0.7), SVector(0.0), SVector(1.3))
            @test flux(u, 1, cubic) ≈ flux(u, 1, burgers)
            @test Trixi.max_abs_speeds(u, cubic)[1] ≈ Trixi.max_abs_speeds(u, burgers)[1]
        end
        @test Trixi.max_abs_speed_naive(SVector(-0.7), SVector(1.3), 1, cubic) ≈
              Trixi.max_abs_speed_naive(SVector(-0.7), SVector(1.3), 1, burgers)
        # standard shallow water at γ = α = 1, β = g: flux (h v, h v² + ½ g h²), eigenvalues
        # v ± sqrt(g h) [LeVeque 2002 §13.1] (Trixi's own equations moved to TrixiShallowWater.jl)
        g = 9.81
        sw = ParametrizedShallowWaterEquations1D(1.0, 1.0, g, 1e-8)
        for (h, v) in ((1.0, 0.3), (2.5, -1.2))
            @test flux(SVector(h, h * v), 1, sw) ≈ SVector(h * v, h * v^2 + 0.5 * g * h^2)
            @test Trixi.max_abs_speeds(SVector(h, h * v), sw)[1] ≈ abs(v) + sqrt(g * h)
            @test Trixi.max_abs_speed_naive(SVector(h, h * v), SVector(h, -h * v), 1, sw) ≈
                  abs(v) + sqrt(g * h)
        end
        @test volume_flux_dg(sw) === flux_central
        @test indicator_variable(CompressibleEulerEquations2D(1.4)) === density_pressure
        @test indicator_variable(sw)(SVector(2.0, 3.0), sw) == 2.0
    end

    @testset "solve: shapes, initial slice, conservation" begin
        t_out = [0.0, 0.02, 0.05]
        coords = node_coordinates(X1, [16], POLYDEG)
        u0 = nodal(golden_ic, coords)
        u = solve("CubicFluxEquation1D", CUBIC, u0, X1, [16], t_out, POLYDEG, 0.5, LF, true)
        @test size(u) == (3, 1, 16)
        @test isapprox(u[1, :, :], lgl_means(u0, POLYDEG); atol = 1e-14)
        @test all(isapprox(sum(u[k, 1, :]), sum(u[1, 1, :]); atol = 1e-12) for k in 2:3)

        sw_ic(x) = (h = 1.0 + 0.2 * sin(2π * x); SVector(h, 0.1 * h * cos(2π * x)))
        u0 = nodal_system(sw_ic, coords)
        sw = Dict("gamma" => 1.0, "alpha" => 1.0, "beta" => 1.0, "h_floor" => 1e-8)
        u = solve("ParametrizedShallowWaterEquations1D", sw, u0, X1, [16], t_out, POLYDEG,
                  0.5, LF, true)
        @test size(u) == (3, 2, 16)
        @test all(isapprox(sum(u[end, v, :]), sum(u[1, v, :]); atol = 1e-12) for v in 1:2)

        coords2 = node_coordinates(X2, [6, 5], POLYDEG)
        u0 = nodal((x, y) -> 0.5 * sin(2π * x) * cos(π * y), coords2)
        u = solve("CubicFluxEquation2D", Dict("a" => 1.0, "b" => 0.0, "c" => 0.0), u0, X2,
                  [6, 5], t_out, POLYDEG, 0.5, LF, true)
        @test size(u) == (3, 1, 6, 5)
        @test isapprox(u[1, :, :, :], lgl_means(u0, POLYDEG); atol = 1e-14)
        @test isapprox(sum(u[end, 1, :, :]), sum(u[1, 1, :, :]); atol = 1e-12)

        gamma = 1.4
        function euler_ic(x, y)
            rho = 1.0 + 0.2 * sin(2π * x) * cos(π * y)
            vx, vy = 0.1, -0.05
            p = 1.0 + 0.1 * cos(2π * x)
            return SVector(rho, rho * vx, rho * vy, p / (gamma - 1) + 0.5 * rho * (vx^2 + vy^2))
        end
        u0 = nodal_system(euler_ic, coords2)
        u = solve("CompressibleEulerEquations2D", Dict("gamma" => gamma), u0, X2, [6, 5],
                  t_out, POLYDEG, 0.5, LF, true)
        @test size(u) == (3, 4, 6, 5)
        @test all(isapprox(sum(u[end, v, :, :]), sum(u[1, v, :, :]); atol = 1e-11) for v in 1:4)

        # the plain weak form, and a single output time (no integration)
        u = solve("CubicFluxEquation1D", CUBIC, nodal(golden_ic, coords), X1, [16], [0.0],
                  POLYDEG, 0.5, LF, false)
        @test size(u) == (1, 1, 16)
        u = solve("InviscidBurgersEquation1D", Dict{String, Float64}(), nodal(golden_ic, coords),
                  X1, [16], [0.0, 0.01], POLYDEG, 0.5, LF, false)
        @test size(u) == (2, 1, 16)
    end

    @testset "cubic (0, ½, 0) reproduces Trixi's Burgers" begin
        coords = node_coordinates(X1, [32], POLYDEG)
        u0 = nodal(golden_ic, coords)
        args = (u0, X1, [32], [0.0, 0.1], POLYDEG, 0.5, LF, true)
        u_cubic = solve("CubicFluxEquation1D", Dict("a" => 0.0, "b" => 0.5, "c" => 0.0), args...)
        u_burgers = solve("InviscidBurgersEquation1D", Dict{String, Float64}(), args...)
        @test isapprox(u_cubic, u_burgers; rtol = 1e-10)
    end

    @testset "convergence against the method of characteristics (D-CONV)" begin
        a, b, c = CUBIC["a"], CUBIC["b"], CUBIC["c"]
        u0(x) = 0.5 + 0.1 * sin(2π * x)
        du0(x) = 0.2π * cos(2π * x)
        dflux(u) = (3a * u + 2b) * u + c
        ddflux(u) = 6a * u + 2b
        # breaking time t_b = −1 / min_ξ d/dξ f'(u0(ξ)); T well before it
        slope_min = minimum(ddflux(u0(ξ)) * du0(ξ) for ξ in range(0, 1; length = 20001))
        @test slope_min < 0
        T = -1 / slope_min / 4
        function exact(x, t)  # Newton for ξ in x = ξ + f'(u0(ξ)) t
            ξ = x
            for _ in 1:50
                ξ -= (ξ + dflux(u0(ξ)) * t - x) / (1 + ddflux(u0(ξ)) * du0(ξ) * t)
            end
            return u0(ξ)
        end
        errors = Float64[]
        for N in (16, 32)
            coords = node_coordinates(X1, [N], POLYDEG)
            # CFL 0.1 so that SSPRK33's O(dt³) stays below the spatial error
            u = solve("CubicFluxEquation1D", CUBIC, nodal(u0, coords), X1, [N], [0.0, T],
                      POLYDEG, 0.1, LF, true)
            reference = lgl_means(nodal(x -> exact(x, T), coords), POLYDEG)
            push!(errors, maximum(abs, u[end, :, :] .- reference))
        end
        @test errors[1] / errors[2] >= 8
    end

    @testset "failure seam (D-FAIL)" begin
        coords = node_coordinates(X1, [8], POLYDEG)
        u0 = nodal(golden_ic, coords)
        rest = (X1, [8], [0.0, 0.1], POLYDEG, 0.5, LF, true)
        @test_throws ArgumentError solve("NoSuchEquation", Dict{String, Float64}(), u0, rest...)
        @test_throws ArgumentError solve("CubicFluxEquation1D", Dict("a" => 1.0, "b" => -0.5), u0,
                                         rest...)
        @test_throws ArgumentError solve("CubicFluxEquation1D", merge(CUBIC, Dict("d" => 1.0)),
                                         u0, rest...)
        @test_throws ArgumentError solve("CubicFluxEquation2D", CUBIC, u0, rest...)
        @test_throws ArgumentError solve("CubicFluxEquation1D", CUBIC, u0[:, 1:3, :], rest...)
        @test_throws ArgumentError solve("CubicFluxEquation1D", CUBIC, u0, X1, [8], [0.0, 0.1],
                                         POLYDEG, 0.5, "roe", true)
        @test_throws ArgumentError solve("CubicFluxEquation1D", CUBIC, u0, X1, [8], [0.1, 0.0],
                                         POLYDEG, 0.5, LF, true)
        bad = copy(u0)
        bad[1, 1, 1] = Inf
        @test_throws SolveFailed solve("CubicFluxEquation1D", CUBIC, bad, X1, [8], [0.0], POLYDEG,
                                       0.5, LF, true)
        # negative pressure: the sound speed is a square root of a negative, raised while stepping
        coords2 = node_coordinates(X2, [4, 4], POLYDEG)
        u0e = nodal_system((x, y) -> SVector(1.0, 0.0, 0.0, -1.0), coords2)
        @test_throws SolveFailed solve("CompressibleEulerEquations2D", Dict("gamma" => 1.4), u0e,
                                       X2, [4, 4], [0.0, 0.1], POLYDEG, 0.5, LF, true)
    end

    @testset "versions (D-PROV)" begin
        v = versions()
        @test Set(keys(v)) == Set(["trixi", "julia", "threads", "context_flux_trixi"])
        @test v["context_flux_trixi"] == string(pkgversion(ContextFluxTrixi))
        @test v["julia"] == string(VERSION)
        @test v["trixi"] == string(pkgversion(Trixi))
    end
end
