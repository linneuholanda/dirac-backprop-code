# Backpropagation as Dirac-bracket dynamics: numerical test

Code for the numerical test (Sec. IV) of

> N. L. Holanda and I. E. Cunha, *Backpropagation as Dirac-bracket dynamics of dissipative training* (2026).

The paper describes gradient-based optimizers as dissipative mechanical systems, with the Bateman–Galley doubling of the degrees of freedom. It shows that backpropagation is Dirac evolution in this description. The layers of a feed-forward network are second-class constraints, and the Hamiltonian contains only the cost of the network output. This code tests that construction numerically, to high precision, on a small network.

## Running

This is a [uv](https://docs.astral.sh/uv/) project (Python ≥ 3.12, NumPy, SciPy, JAX on CPU in double precision, Matplotlib).

```bash
uv run run-dirac   # the numerical test, about 3 minutes on a laptop CPU
uv run run-seeds   # repeat every check with 4 other random seeds (~12 min; no files written)
```

`run-dirac` prints its checks, writes the figure of the paper to `figures/dirac_backprop.pdf`, and writes the numbers quoted in the paper to `results/dirac_backprop.json`. Both files are included in the repository. A line marked `[OK]` or `[FAIL]` is a check with a tolerance. A line without a mark reports a quantity for information, such as the size of an effect.

## What is checked

The checks are made at random points of the constraint surface away from the physical limit (x₁ ≠ x₂, p₁ ≠ p₂). The numbers refer to the sections and equations of the paper.

1. The Dirac evolution of the parameters and their momenta equals ordinary training. This is checked for the heavy ball and for γ = 3/t, with both the squared-error and the cross-entropy costs (Eq. (35)).
2. The Dirac evolution of the activations is the forward-mode derivative, and the momenta of the activations stay zero (Eq. (36)).
3. The Dirac brackets of the activations and the cost with the parameters and momenta (Eq. (38)).
4. The Lagrange multipliers of the layer constraints are the backpropagated errors (Eq. (39)).
5. With Nesterov's look-ahead layer, the Dirac evolution equals Hamilton's equations of the look-ahead Hamiltonian (Eq. (16)).
6. Negative control: the Poisson bracket alone misses the gradient.
7. Integrated in the full constrained phase space, the Dirac flow reproduces training for the heavy ball, for γ = 3/t and for the Nesterov look-ahead ODE (Eq. (17)).
8. App. C, the look-ahead constraints:
   - the Dirac brackets, including Δ(y₁, y₂) = h/μ between the copies (Eq. (C9));
   - the multipliers (Eqs. (C5) and (C6));
   - the g₋ terms of the equations of motion (Eq. (C7));
   - the evolution of the constrained coordinates (Eq. (C8)).

## Methods

- The optimizer ODE μ(t)ẍ + γ(t)ẋ + ∇L = 0 is integrated with SciPy's DOP853 (rtol 1e-11).
- The Dirac bracket is implemented generically:
  - Ω represents the bracket Γ = Γ₁ − Γ₂ of the doubled system;
  - the gradients of the constraints come from automatic differentiation of the layer equations;
  - the flow is ż = Ω∇H − Ω∇Θᵀ C⁻¹(∇Θ Ω∇H + ∂Θ/∂t).

  The Hamiltonian contains only the cost of the network output, and no backward pass is written on the Dirac side. The reference is ordinary training, with the gradient from reverse-mode automatic differentiation.
- The doubled systems are integrated in the full doubled phase space, starting on the physical limit. Round-off then seeds the anti-damped mirror mode, and the two copies separate exponentially. The trajectories are therefore compared through the average of the copies, which deviates only at second order in the separation. The separation itself is plotted.

## Layout

```
src/gbl_experiments/dirac_backprop.py   the numerical test
src/gbl_experiments/common.py           paths, plotting style, ODE integration, result output
src/gbl_experiments/seeds.py            rerun of all checks with other seeds
results/dirac_backprop.json             numbers quoted in the paper
figures/dirac_backprop.pdf              figure of Sec. IV
```

## License

MIT; see `LICENSE`.
