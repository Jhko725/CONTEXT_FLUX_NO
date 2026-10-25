"""Set the Julia side of `TrixiBackend` up on this machine, once.
Usage: uv run python scripts/setup_julia.py [--test]

Through juliapkg (dev group): finds or installs Julia `~1.12` (`src/context_flux_no/
juliapkg.json`), instantiates the committed project `reference/trixi/` (shared-project mode:
its `Manifest.toml` is used as it is) with `ContextFluxTrixi` `dev`-ed in and PythonCall at
juliacall's version, precompiles, then starts Julia through juliacall once and prints
`TrixiBackend().describe()`. `--test` also runs `Pkg.test("ContextFluxTrixi")`.
"""

import argparse
import subprocess
import sys

from context_flux_no.simulations import trixi


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--test", action="store_true", help="run the Julia package tests"
    )
    args = parser.parse_args()

    project = trixi.REFERENCE_TRIXI
    if not (project / "Project.toml").is_file():
        print(
            f"no Julia project at {project} (the laptop step writes it)",
            file=sys.stderr,
        )
        return 1
    trixi.configure_environment()

    import juliapkg

    juliapkg.resolve(force=True)
    exe = juliapkg.executable()
    print(f"Julia at {exe}, project {juliapkg.project()}")
    if args.test:
        subprocess.run(
            [
                exe,
                f"--project={project}",
                "--startup-file=no",
                "-e",
                'import Pkg; Pkg.test("ContextFluxTrixi")',
            ],
            check=True,
        )
    print(trixi.TrixiBackend().describe())
    return 0


if __name__ == "__main__":
    sys.exit(main())
