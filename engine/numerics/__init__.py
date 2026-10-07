"""
`engine.numerics`: numerical methods the pipeline shares.

    roots.py   the root solver of every calibration and exercise boundary (decision A-21):
               safeguarded Newton (the default) or bisection, a fixed number of steps on
               every backend; `implicit_root`, a root's derivative by the implicit function
               theorem

Depends on JAX only: the pipeline imports it, it imports nothing from the pipeline.
"""
