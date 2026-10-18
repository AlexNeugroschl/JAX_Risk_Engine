"""
`engine.solvers`: numerical methods the pipeline shares.

    roots.py   the root solver of every calibration and exercise boundary (decision A-21):
               safeguarded Newton (the default) or bisection, a fixed number of steps on
               every backend; `implicit_root` and `implicit_solution`, a root's and a
               system's solution's derivative by the implicit function theorem

Depends on JAX only: the pipeline imports it, it imports nothing from the pipeline.
"""
