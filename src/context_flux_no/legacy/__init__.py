"""Code that awaits its rewrite: `models`, `nn` and `plotting`.

Rule R7: the strict checks apply only to the clean paths listed under
`[tool.context_flux_no]` of `pyproject.toml`. Everything else in the package lives here,
stays importable and is otherwise untouched until the branch that rewrites it. A module
joins the clean list by leaving `legacy/` and never returns.
"""
