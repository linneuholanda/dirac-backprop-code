"""Numerical test (Sec. IV): backpropagation as Dirac evolution (Sec. III), checked numerically.

The Dirac machinery is implemented generically: phase-space points z = (q1, p1, q2, p2) of the doubled
system, the bracket Gamma = Gamma_1 - Gamma_2 as a constant matrix Omega, constraint functions Theta(z)
whose gradients are taken by automatic differentiation of the *layer equations alone*, and

    dz/dt = Omega dH - Omega dTheta^T C^{-1} dTheta Omega dH,     C = dTheta Omega dTheta^T.

No backward pass is coded anywhere on the Dirac side: the Hamiltonian contains the cost of the network
output only, Eq. (22). The reference is ordinary training, Hamilton's equations of
H0 + L(x1) - L(x2) with grad L from reverse-mode automatic differentiation of the whole network.

Checks (on random points of the constraint surface with x1 != x2, p1 != p2, i.e. off the physical limit):
  1. Dirac evolution of (x, p) == reference, for heavy ball and for gamma = 3/t, for the squared-error
     and the cross-entropy costs (Eq. (35));
  2. Dirac evolution of the activations == forward-mode JVP, and the momenta of the activations stay 0
     (Eq. (36));
  3. Dirac brackets Delta(h_l, p) == +-dh_l/dx, Delta(h_l, x) == 0, Delta(c, p) == +-grad L
     (Eq. (38));
  4. Lagrange multipliers of the layer constraints == -+ the backpropagated errors (Eq. (39));
  5. Nesterov: look-ahead layer y = x + h vbar; Dirac evolution == Hamilton's equations of H_LA
     (Eq. (16));
  6. negative control: the Poisson bracket alone misses the gradient;
  8. Nesterov look-ahead appendix (App. C): Dirac brackets of the look-ahead
     points and activations, including Delta(y_1, y_2) = h/mu between the copies; multipliers
     u_psi = -+grad L(y), u_phi = -+delta(y), u_rho, u_pi; the g_- terms of the equations of motion;
  7. trajectories: the Dirac flow integrated in the full constrained phase space reproduces training
     (heavy ball, gamma = 3/t, Nesterov look-ahead ODE, Eq. (17)).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from .common import integrate, report, save_figure, save_results, setup_style

SIZES = (3, 5, 4, 2)        # input, two tanh hidden layers, linear output
N = 6
SEED = 2
H_LOOKAHEAD = 0.1


# --------------------------------------------------------------------------- network
class Network:
    def __init__(self, cost: str, seed: int | None = None):
        rng = np.random.default_rng(SEED if seed is None else seed)
        self.shapes = [(SIZES[i + 1], SIZES[i]) for i in range(len(SIZES) - 1)]
        self.P = sum(a * b for a, b in self.shapes)
        self.H = N * sum(SIZES[1:])
        self.h0 = jnp.asarray(rng.normal(size=(N, SIZES[0])))
        if cost == "lsq":
            self.y = jnp.asarray(rng.normal(size=(N, SIZES[-1])))
        else:
            self.y = jnp.asarray(np.eye(SIZES[-1])[rng.integers(0, SIZES[-1], size=N)])
        self.cost_name = cost
        self.x_init = np.concatenate([rng.normal(size=a * b) / np.sqrt(b) for a, b in self.shapes])

    def weights(self, x):
        out, i = [], 0
        for a, b in self.shapes:
            out.append(x[i:i + a * b].reshape(a, b)); i += a * b
        return out

    def split_h(self, h):
        out, i = [], 0
        for n in SIZES[1:]:
            out.append(h[i:i + N * n].reshape(N, n)); i += N * n
        return out

    @staticmethod
    def act(ell, z, n_layers):
        return z if ell == n_layers - 1 else jnp.tanh(z)

    def forward(self, x):
        """All activations, stacked layer by layer (samples inside each layer)."""
        hs, h = [], self.h0
        ws = self.weights(x)
        for ell, w in enumerate(ws):
            h = self.act(ell, h @ w.T, len(ws)); hs.append(h.ravel())
        return jnp.concatenate(hs)

    def residual(self, x, h):
        """Layer constraints phi_l = h_l - f_l(W_l, h_{l-1}), Eq. (23)."""
        ws, hl = self.weights(x), self.split_h(h)
        prev, out = self.h0, []
        for ell, (w, cur) in enumerate(zip(ws, hl)):
            out.append((cur - self.act(ell, prev @ w.T, len(ws))).ravel()); prev = cur
        return jnp.concatenate(out)

    def cost(self, h):
        """c(h_L) = (1/N) sum_n c^(n)(h_L^(n)), a function of the output activations only."""
        out = self.split_h(h)[-1]
        if self.cost_name == "lsq":
            return 0.5 * jnp.sum((out - self.y) ** 2) / N
        return -jnp.sum(self.y * jax.nn.log_softmax(out, axis=1)) / N

    def loss(self, x):
        return self.cost(self.forward(x))

    def errors(self, x):
        """Backpropagated errors delta_l = dc/dh_l (through all later layers), by reverse-mode AD."""
        ws, hl = self.weights(x), self.split_h(self.forward(x))
        out = []
        for ell in range(len(ws)):
            def tail(h_ell, ell=ell):
                h = h_ell.reshape(N, -1)
                for k in range(ell + 1, len(ws)):
                    h = self.act(k, h @ ws[k].T, len(ws))
                hs = [jnp.zeros(N * n) for n in SIZES[1:]]
                hs[-1] = h.ravel()
                return self.cost(jnp.concatenate(hs))
            out.append(jax.grad(tail)(hl[ell].ravel()))
        return jnp.concatenate(out)


# --------------------------------------------------------------------------- optimizer Hamiltonians
def optimizer(name):
    """mu(t), gamma(t) and their derivatives."""
    if name == "heavy ball":
        return dict(mu=lambda t: 1.0, mud=lambda t: 0.0, mudd=lambda t: 0.0,
                    gamma=lambda t: 1.0, gammad=lambda t: 0.0)
    return dict(mu=lambda t: 1.0, mud=lambda t: 0.0, mudd=lambda t: 0.0,
                gamma=lambda t: 3.0 / t, gammad=lambda t: -3.0 / t**2)


def H0(x1, p1, x2, p2, t, o):
    """Eq. (11) in Euclidean components, without the potential V."""
    mu, c = o["mu"](t), o["gamma"](t) - o["mud"](t)
    k = (o["gammad"](t) - o["mudd"](t)) / 4 + c**2 / (8 * mu)
    return (p1 @ p1 - p2 @ p2) / (2 * mu) - c / (2 * mu) * (p1 @ x2 - p2 @ x1) - k * (x1 @ x1 - x2 @ x2)


def vbar(x1, p1, x2, p2, t, o):
    g1 = jax.grad(H0, argnums=1)(x1, p1, x2, p2, t, o)
    g2 = jax.grad(H0, argnums=3)(x1, p1, x2, p2, t, o)
    return (g1 - g2) / 2        # v^a = dbar^a H0: v1 = dH0/dp1, v2 = -dH0/dp2


# --------------------------------------------------------------------------- generic constrained system
def omega(n):
    """Gamma = Gamma_1 - Gamma_2 on z = (q1, p1, q2, p2), each block of size n."""
    J = np.block([[np.zeros((n, n)), np.eye(n)], [-np.eye(n), np.zeros((n, n))]])
    return jnp.asarray(np.block([[J, np.zeros_like(J)], [np.zeros_like(J), -J]]))


class Constrained:
    """Doubled phase space with per-copy coordinates q = (x, [y,] h) and the network constraints."""

    def __init__(self, net: Network, opt: dict, lookahead: bool = False, h_la: float = H_LOOKAHEAD):
        self.net, self.opt, self.la, self.h_la = net, opt, lookahead, h_la
        P, H = net.P, net.H
        self.n = P + (P if lookahead else 0) + H
        self.Om = omega(self.n)

    def split(self, z):
        n = self.n
        return z[:n], z[n:2 * n], z[2 * n:3 * n], z[3 * n:]

    def parts(self, q):
        P = self.net.P
        x = q[:P]
        if self.la:
            return x, q[P:2 * P], q[2 * P:]
        return x, None, q[P:]

    def hamiltonian(self, z, t):
        q1, p1, q2, p2 = self.split(z)
        P = self.net.P
        x1, _, h1 = self.parts(q1); x2, _, h2 = self.parts(q2)
        return H0(x1, p1[:P], x2, p2[:P], t, self.opt) + self.net.cost(h1) - self.net.cost(h2)

    def constraints(self, z, t):
        q1, p1, q2, p2 = self.split(z)
        P = self.net.P
        x1, y1, h1 = self.parts(q1); x2, y2, h2 = self.parts(q2)
        out = []
        if self.la:
            vb = vbar(x1, p1[:P], x2, p2[:P], t, self.opt)
            out += [y1 - x1 - self.h_la * vb, p1[P:2 * P], y2 - x2 - self.h_la * vb, p2[P:2 * P]]
            w1, w2 = y1, y2
        else:
            w1, w2 = x1, x2
        out += [self.net.residual(w1, h1), p1[self.n - self.net.H:], self.net.residual(w2, h2),
                p2[self.n - self.net.H:]]
        return jnp.concatenate(out)

    def dirac_parts(self, z, t):
        G = jax.jacfwd(self.constraints)(z, t)
        dH = jax.grad(self.hamiltonian)(z, t)
        Om = self.Om
        C = G @ Om @ G.T
        return G, dH, C

    def vector_field(self, z, t, explicit_time=True):
        """Dirac evolution; for constraints with explicit time dependence (the look-ahead constraint with
        time-dependent mu, gamma) the term -Gamma(z, Theta) C^{-1} dTheta/dt keeps them satisfied."""
        G, dH, C = self.dirac_parts(z, t)
        Om = self.Om
        rhs = G @ Om @ dH
        if explicit_time:
            rhs = rhs + jax.jacfwd(self.constraints, argnums=1)(z, jnp.float64(t))
        return Om @ dH - Om @ G.T @ jnp.linalg.solve(C, rhs)

    def poisson_field(self, z, t):
        return self.Om @ jax.grad(self.hamiltonian)(z, t)

    def dirac_matrix(self, z, t):
        G, _, C = self.dirac_parts(z, t)
        Om = self.Om
        return Om - Om @ G.T @ jnp.linalg.solve(C, G @ Om)

    def multipliers(self, z, t):
        """u = -C^{-1} Gamma(Theta, H): H_T = H + u.Theta preserves the constraints."""
        G, dH, C = self.dirac_parts(z, t)
        return -jnp.linalg.solve(C, G @ self.Om @ dH)

    def surface_point(self, x1, p1, x2, p2, t):
        """Point of the constraint surface over (x_i, p_i): activations from the forward pass, constrained
        momenta zero, look-ahead coordinates y_i = x_i + h vbar."""
        net = self.net
        zeros_h = np.zeros(net.H)
        if self.la:
            vb = np.asarray(vbar(jnp.asarray(x1), jnp.asarray(p1), jnp.asarray(x2), jnp.asarray(p2), t, self.opt))
            y1, y2 = x1 + self.h_la * vb, x2 + self.h_la * vb
            q1 = np.concatenate([x1, y1, net.forward(y1)]); q2 = np.concatenate([x2, y2, net.forward(y2)])
            pp1 = np.concatenate([p1, np.zeros(net.P), zeros_h]); pp2 = np.concatenate([p2, np.zeros(net.P), zeros_h])
        else:
            q1 = np.concatenate([x1, net.forward(x1)]); q2 = np.concatenate([x2, net.forward(x2)])
            pp1 = np.concatenate([p1, zeros_h]); pp2 = np.concatenate([p2, zeros_h])
        return jnp.asarray(np.concatenate([q1, pp1, q2, pp2]))


def reference_field(net, opt, x1, p1, x2, p2, t, lookahead=False, h_la=H_LOOKAHEAD):
    """Ordinary training: Hamilton's equations of H0 + L(x1) - L(x2) (or of H_LA), grad L by reverse-mode AD."""
    def Hred(x1, p1, x2, p2):
        if lookahead:
            vb = vbar(x1, p1, x2, p2, t, opt)
            return H0(x1, p1, x2, p2, t, opt) + net.loss(x1 + h_la * vb) - net.loss(x2 + h_la * vb)
        return H0(x1, p1, x2, p2, t, opt) + net.loss(x1) - net.loss(x2)
    g = jax.grad(Hred, argnums=(0, 1, 2, 3))(x1, p1, x2, p2)
    return g[1], -g[0], -g[3], g[2]          # x1', p1', x2', p2'


# --------------------------------------------------------------------------- checks
def lookahead_checks(sysm, z, t, zd, upd, cost, opt_name):
    """8. Statements of the look-ahead appendix (App. C), off the physical limit:
    Dirac brackets (Eq. (C9)), multipliers (Eqs. (C5)
    and (C6)), the g_- terms of Eq. (C7) and the
    constrained evolution (Eq. (C8)). Constraint order of `Constrained`:
    Theta = (psi_1, rho_1, psi_2, rho_2, phi_1, pi_1, phi_2, pi_2)."""
    net, o, hla, n = sysm.net, sysm.opt, sysm.h_la, sysm.n
    P, H = net.P, net.H
    eps = {1: 1.0, 2: -1.0}
    mu, cdamp = o["mu"](t), o["gamma"](t) - o["mud"](t)
    off = {1: 0, 2: 2 * n}                                    # start of q_i in z; p_i starts n later
    ix = {i: off[i] + np.arange(P) for i in (1, 2)}
    iy = {i: off[i] + P + np.arange(P) for i in (1, 2)}
    ih = {i: off[i] + 2 * P + np.arange(H) for i in (1, 2)}
    ip = {i: off[i] + n + np.arange(P) for i in (1, 2)}
    x = {i: z[ix[i]] for i in (1, 2)}; p = {i: z[ip[i]] for i in (1, 2)}; y = {i: z[iy[i]] for i in (1, 2)}
    I = np.eye(P)
    Dm = sysm.dirac_matrix(z, t)
    D = lambda a, b: Dm[np.ix_(a, b)]
    tag = f"{opt_name}, {cost}"
    for i in (1, 2):
        Jy = jax.jacfwd(net.forward)(y[i])
        for j in (1, 2):
            upd(f"8: Delta(y_i, p_j) == [eps_i delta_ij - eps_j h(gamma - mudot)/4mu] I ({tag})",
                jnp.max(jnp.abs(D(iy[i], ip[j]) - ((eps[i] if i == j else 0.0) - eps[j] * hla * cdamp / (4 * mu)) * I)))
            upd(f"8: Delta(y_i, x_j) == -eps_j h/(2mu) I ({tag})",
                jnp.max(jnp.abs(D(iy[i], ix[j]) + eps[j] * hla / (2 * mu) * I)))
            for nm, idx in (("x", ix[j]), ("p", ip[j]), ("y", iy[j])):
                upd(f"8: Delta(h_i, .) == (dh/dy at y_i) Delta(y_i, .), for . = x_j, p_j, y_j ({tag})",
                    jnp.max(jnp.abs(D(ih[i], idx) - Jy @ D(iy[i], idx))))
    upd(f"8: Delta(y_1, y_2) == (h/mu) I: the look-ahead points of the two copies do not commute ({tag})",
        jnp.max(jnp.abs(D(iy[1], iy[2]) - hla / mu * I)))
    # size of the cross-copy brackets of the activations (information: nonzero for h != 0)
    upd(f"8: |Delta(h_1, h_2)| (max entry, nonzero off the physical limit) ({tag})", -float(jnp.max(jnp.abs(D(ih[1], ih[2])))))
    # multipliers, with the explicit time dependence of the constraints
    G, dH, C = sysm.dirac_parts(z, t)
    rhs = G @ sysm.Om @ dH + jax.jacfwd(sysm.constraints, argnums=1)(z, jnp.float64(t))
    u = -jnp.linalg.solve(C, rhs)
    u_psi = {1: u[:P], 2: u[2 * P:3 * P]}; u_rho = {1: u[P:2 * P], 2: u[3 * P:4 * P]}
    b = 4 * P
    u_phi = {1: u[b:b + H], 2: u[b + 2 * H:b + 3 * H]}; u_pi = {1: u[b + H:b + 2 * H], 2: u[b + 3 * H:b + 4 * H]}
    gL = {i: jax.grad(net.loss)(y[i]) for i in (1, 2)}
    for i in (1, 2):
        upd(f"8: u_psi,i == -eps_i grad L(y_i) ({tag})", jnp.max(jnp.abs(u_psi[i] + eps[i] * gL[i])))
        upd(f"8: u_phi,i == -eps_i delta(y_i), errors at the look-ahead point ({tag})",
            jnp.max(jnp.abs(u_phi[i] + eps[i] * net.errors(y[i]))))
        upd(f"8: dy_i/dt == eps_i u_rho,i ({tag})", jnp.max(jnp.abs(zd[iy[i]] - eps[i] * u_rho[i])))
        upd(f"8: dh_i/dt == eps_i u_pi,i == (dh/dy) dy_i/dt ({tag})", max(
            jnp.max(jnp.abs(zd[ih[i]] - eps[i] * u_pi[i])),
            jnp.max(jnp.abs(zd[ih[i]] - jax.jvp(net.forward, (y[i],), (zd[iy[i]],))[1]))))
    # eps_i u_rho,i == v_i + h[Gamma(vbar, H0) + d vbar/dt] - (h/mu) grad L(y_ibar)
    vfun = lambda a, b_, c, d, s: vbar(a, b_, c, d, s, o)
    v = {1: jax.grad(H0, argnums=1)(x[1], p[1], x[2], p[2], t, o), 2: -jax.grad(H0, argnums=3)(x[1], p[1], x[2], p[2], t, o)}
    gH0 = jax.grad(H0, argnums=(0, 1, 2, 3))(x[1], p[1], x[2], p[2], t, o)
    flow0 = (gH0[1], -gH0[0], -gH0[3], gH0[2])                # Hamilton's equations of H0
    tt = jnp.float64(t)
    vb_H0 = jax.jvp(vfun, (x[1], p[1], x[2], p[2], tt), (*flow0, jnp.float64(0.0)))[1]
    vb_t = jax.jvp(vfun, (x[1], p[1], x[2], p[2], tt),
                   (jnp.zeros(P), jnp.zeros(P), jnp.zeros(P), jnp.zeros(P), jnp.float64(1.0)))[1]
    for i in (1, 2):
        target = v[i] + hla * (vb_H0 + vb_t) - hla / mu * gL[3 - i]
        upd(f"8: eps_i u_rho,i == v_i + h[Gamma(vbar, H0) + dvbar/dt] - (h/mu) grad L(y_ibar) ({tag})",
            jnp.max(jnp.abs(eps[i] * u_rho[i] - target)))
    # equations of motion: g_- terms, and dy/dt == dx/dt + h dvbar/dt along the motion
    gm = gL[1] - gL[2]
    for i in (1, 2):
        upd(f"8: dx_i/dt == v_i + eps_i (h/2mu) g_- ({tag})", jnp.max(jnp.abs(zd[ix[i]] - v[i] - eps[i] * hla / (2 * mu) * gm)))
        upd(f"8: dp_i/dt == -d^i H0 - grad L(y_i) + eps_i h(gamma - mudot)/(4mu) g_- ({tag})",
            jnp.max(jnp.abs(zd[ip[i]] - flow0[2 * i - 1] + gL[i] - eps[i] * hla * cdamp / (4 * mu) * gm)))
    upd(f"8: |g_-| = |grad L(y_1) - grad L(y_2)| (nonzero off the physical limit) ({tag})", -float(jnp.max(jnp.abs(gm))))
    vb_dot = jax.jvp(vfun, (x[1], p[1], x[2], p[2], tt), (zd[ix[1]], zd[ip[1]], zd[ix[2]], zd[ip[2]], jnp.float64(1.0)))[1]
    for i in (1, 2):
        upd(f"8: dy_i/dt == dx_i/dt + h dvbar/dt ({tag})", jnp.max(jnp.abs(zd[iy[i]] - zd[ix[i]] - hla * vb_dot)))


def pointwise_checks(results):
    rng = np.random.default_rng(SEED + 10)
    worst = {}
    def upd(key, val):
        val = float(val)
        if val < 0:      # entries recorded as -value are minima over the points (should be large)
            worst[key] = max(worst.get(key, -np.inf), val)
        else:
            worst[key] = max(worst.get(key, 0.0), val)
    for cost in ("lsq", "xent"):
        net = Network(cost)
        P, H = net.P, net.H
        for opt_name in ("heavy ball", "gamma = 3/t"):
            opt = optimizer(opt_name)
            for la in (False, True):
                sysm = Constrained(net, opt, lookahead=la)
                n = sysm.n
                for trial in range(3):
                    t = 2.0 + trial
                    x1, x2 = net.x_init + 0.3 * rng.normal(size=P), net.x_init + 0.3 * rng.normal(size=P)
                    p1, p2 = rng.normal(size=P), rng.normal(size=P)
                    z = sysm.surface_point(x1, p1, x2, p2, t)
                    zd = sysm.vector_field(z, t)
                    q1d, p1d, q2d, p2d = sysm.split(zd)
                    ref = reference_field(net, opt, *map(jnp.asarray, (x1, p1, x2, p2)), t, lookahead=la)
                    got = (q1d[:P], p1d[:P], q2d[:P], p2d[:P])
                    scale = max(float(jnp.max(jnp.abs(r))) for r in ref)
                    err = max(float(jnp.max(jnp.abs(g - r))) for g, r in zip(got, ref)) / scale
                    upd(f"{'5' if la else '1'}: Dirac (x', p') == reference, {'Nesterov look-ahead, ' if la else ''}{opt_name}, {cost}", err)
                    if la and opt_name != "heavy ball":
                        zd0 = sysm.vector_field(z, t, explicit_time=False)
                        a1, b1, a2, b2 = sysm.split(zd0)
                        err0 = max(float(jnp.max(jnp.abs(g - r))) for g, r in zip((a1[:P], b1[:P], a2[:P], b2[:P]), ref)) / scale
                        upd(f"5: gamma = 3/t: (x', p') without the dTheta/dt term, relative error ({cost})", err0)
                    # constraints are preserved: d Theta/dt = dTheta . z' = 0
                    G = jax.jacfwd(sysm.constraints)(z, t)
                    upd("1/5: constraints preserved, max |dTheta/dt|",
                        jnp.max(jnp.abs(G @ zd + jax.jacfwd(sysm.constraints, argnums=1)(z, jnp.float64(t)))))
                    if la:
                        lookahead_checks(sysm, z, t, zd, upd, cost, opt_name)
                        continue
                    # 2. activations: forward-mode JVP along x'
                    hd1 = q1d[P:]; hd2 = q2d[P:]
                    jvp1 = jax.jvp(net.forward, (jnp.asarray(x1),), (q1d[:P],))[1]
                    jvp2 = jax.jvp(net.forward, (jnp.asarray(x2),), (q2d[:P],))[1]
                    upd("2: Dirac h' == forward-mode JVP dh/dx . x'", max(jnp.max(jnp.abs(hd1 - jvp1)), jnp.max(jnp.abs(hd2 - jvp2))))
                    upd("2: momenta of the activations stay zero, max |pi'|", max(jnp.max(jnp.abs(p1d[P:])), jnp.max(jnp.abs(p2d[P:]))))
                    # 6. negative control
                    pz = sysm.poisson_field(z, t)
                    _, pp1d, _, _ = sysm.split(pz)
                    upd(f"6: negative control: Poisson p1' misses the gradient, |error|/|grad L| ({cost})",
                        -float(jnp.max(jnp.abs(pp1d[:P] - ref[1]))) / float(jnp.max(jnp.abs(jax.grad(net.loss)(jnp.asarray(x1))))))
                    if opt_name != "heavy ball":
                        continue
                    # 3. Dirac brackets of activations with x and p
                    Dm = sysm.dirac_matrix(z, t)
                    i_h1 = np.arange(P, n); i_x1 = np.arange(P); i_p1 = n + np.arange(P)
                    i_h2 = 2 * n + np.arange(P, n); i_x2 = 2 * n + np.arange(P); i_p2 = 3 * n + np.arange(P)
                    jac1 = jax.jacfwd(net.forward)(jnp.asarray(x1)); jac2 = jax.jacfwd(net.forward)(jnp.asarray(x2))
                    upd("3: Delta(h_1, p_1) == +dh/dx (copy 1)", jnp.max(jnp.abs(Dm[np.ix_(i_h1, i_p1)] - jac1)))
                    upd("3: Delta(h_2, p_2) == -dh/dx (copy 2)", jnp.max(jnp.abs(Dm[np.ix_(i_h2, i_p2)] + jac2)))
                    upd("3: Delta(h, x) == 0, Delta(h_1, p_2) == 0", max(
                        jnp.max(jnp.abs(Dm[np.ix_(i_h1, i_x1)])), jnp.max(jnp.abs(Dm[np.ix_(i_h2, i_x2)])),
                        jnp.max(jnp.abs(Dm[np.ix_(i_h1, i_p2)]))))
                    gc1 = jax.grad(net.cost)(jnp.asarray(z[P:n]))
                    upd(f"3: Delta(c(h_L1), p_1) == grad L(x1) ({cost})",
                        jnp.max(jnp.abs(gc1 @ Dm[np.ix_(i_h1, i_p1)] - jax.grad(net.loss)(jnp.asarray(x1)))))
                    # 4. multipliers
                    u = sysm.multipliers(z, t)
                    u_phi1, u_phi2 = u[:H], u[2 * H:3 * H]
                    upd("4: multipliers of the layer constraints: u_phi1 == -delta(x1)", jnp.max(jnp.abs(u_phi1 + net.errors(jnp.asarray(x1)))))
                    upd("4: multipliers of the layer constraints: u_phi2 == +delta(x2)", jnp.max(jnp.abs(u_phi2 - net.errors(jnp.asarray(x2)))))
                    upd("4: multipliers of the momentum constraints: u_pi1 == h1'", jnp.max(jnp.abs(u[H:2 * H] - hd1)))
    results["pointwise"] = {}
    for k, v in sorted(worst.items()):
        if k.startswith("6") or "nonzero" in k:
            results["pointwise"][k] = report(k + " (min over points)", -v)
        else:
            results["pointwise"][k] = report(k, v, 1e-10)
    return Network("lsq").P, Network("lsq").H


def trajectory_runs(results):
    net = Network("xent")
    P = net.P
    x0 = net.x_init
    runs = {}
    for name, opt_name, la, t0, t1 in [("heavy ball", "heavy ball", False, 0.0, 30.0),
                                       (r"$\gamma = 3/t$", "gamma = 3/t", False, 1.0, 31.0),
                                       ("Nesterov look-ahead", "heavy ball", True, 0.0, 30.0)]:
        opt = optimizer(opt_name)
        sysm = Constrained(net, opt, lookahead=la)
        # physical limit, at rest: p = mu x' + (gamma - mu') x / 2 = (gamma - mu') x / 2
        p0 = (opt["gamma"](t0) - opt["mud"](t0)) / 2 * x0
        z0 = sysm.surface_point(x0, p0, x0, p0, t0)
        vf = jax.jit(sysm.vector_field)
        t, zs = integrate(lambda t, z: np.asarray(vf(jnp.asarray(z), t)), np.asarray(z0), t0, t1, n_eval=200)
        cons = jax.jit(sysm.constraints)
        viol = np.array([float(jnp.max(jnp.abs(cons(jnp.asarray(z), tt)))) for tt, z in zip(t, zs)])
        n = sysm.n
        x_dirac = zs[:, :P]
        x_avg = (zs[:, :P] + zs[:, 2 * n:2 * n + P]) / 2
        sep = np.max(np.abs(zs[:, :P] - zs[:, 2 * n:2 * n + P]), axis=1)
        # reference: ordinary second-order ODE in parameter space, gradient by reverse-mode AD
        grad = jax.jit(jax.grad(net.loss))
        mu, gam = opt["mu"], opt["gamma"]

        def rhs(t, s):
            x, v = s[:P], s[P:]
            g = grad(jnp.asarray(x + H_LOOKAHEAD * v)) if la else grad(jnp.asarray(x))
            return np.concatenate([v, -(gam(t) * v + np.asarray(g)) / mu(t)])
        _, sref = integrate(rhs, np.concatenate([x0, np.zeros(P)]), t0, t1, n_eval=200)
        diff = np.max(np.abs(x_avg - sref[:, :P]), axis=1)
        diff1 = np.max(np.abs(x_dirac - sref[:, :P]), axis=1)
        lossf = jax.jit(net.loss)
        L_d = np.array([float(lossf(jnp.asarray(x))) for x in x_dirac])
        L_r = np.array([float(lossf(jnp.asarray(x))) for x in sref[:, :P]])
        runs[name] = dict(t=t, L_d=L_d, L_r=L_r, diff=diff, viol=viol, sep=sep)
        results[f"trajectory[{name}]"] = dict(
            max_param_diff_copy_average_close=report(
                f"7: {name}: max_t |(x_1 + x_2)/2 (Dirac) - x_ref| while |x_1 - x_2| < 1e-6",
                diff[sep < 1e-6].max(), 1e-9),
            max_param_diff_copy_average=report(
                f"7: {name}: max_t |(x_1 + x_2)/2 (Dirac) - x_ref| (second order in the separation)", diff.max()),
            final_diff_over_sep_squared=(report(
                f"7: {name}: final |(x_1 + x_2)/2 - x_ref| / |x_1 - x_2|^2", diff[-1] / sep[-1]**2)
                if sep[-1] > 1e-6 else None),
            max_param_diff_copy1=report(f"7: {name}: max_t |x_1 (Dirac) - x_ref|", diff1.max()),
            max_copy_separation=report(
                f"7: {name}: max_t |x_1 - x_2| (round-off amplified by the anti-damped mirror mode)", sep.max()),
            copy_separation_growth_rate=(float(np.polyfit(t[sep > 1e-13], np.log(sep[sep > 1e-13]), 1)[0])
                                         if np.sum(sep > 1e-13) > 10 else None),
            max_constraint_violation=report(f"7: {name}: max_t |Theta(z(t))|", viol.max(), 1e-7),
            loss_initial=L_r[0], loss_final=L_r[-1])
    return runs


def main() -> None:
    print("Numerical test: backpropagation as Dirac evolution")
    setup_style()
    results: dict = {"setup": dict(sizes=SIZES, N=N, seed=SEED, h_lookahead=H_LOOKAHEAD)}
    P, H = pointwise_checks(results)
    results["setup"].update(n_params=P, n_activations=H)
    runs = trajectory_runs(results)

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4))
    for (name, r), c in zip(runs.items(), ["C0", "C3", "C2"]):
        tt = r["t"] - r["t"][0]
        axes[0].plot(tt, r["L_r"], color=c, label=name)
        axes[0].plot(tt[::8], r["L_d"][::8], "o", ms=2.8, mfc="none", color="k", mew=0.5)
        axes[1].plot(tt[1:], r["diff"][1:], color=c, label=name)
        axes[1].plot(tt[1:], r["viol"][1:], "--", color=c)
        axes[1].plot(tt[1:], r["sep"][1:], ":", color=c)
    axes[0].plot([], [], "o", ms=2.8, mfc="none", color="k", mew=0.5, label="Dirac flow")
    axes[0].set_xlabel(r"$t - t_0$"); axes[0].set_ylabel("training loss")
    axes[0].set_title("(a) training loss")
    axes[0].legend(frameon=False)
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$t - t_0$")
    axes[1].set_title("(b) deviations")
    axes[1].plot([], [], "-", color="k", label=r"$|\bar{\mathbf{x}}_{\rm Dirac} - \mathbf{x}_{\rm ref}|$")
    axes[1].plot([], [], "--", color="k", label=r"$|\Theta|$")
    axes[1].plot([], [], ":", color="k", label=r"$|\mathbf{x}_1 - \mathbf{x}_2|$")
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles[3:], labels[3:], frameon=False, loc="upper left")
    axes[1].set_ylim(1e-16, 1e-1)
    fig.tight_layout()
    save_figure(fig, "dirac_backprop")
    save_results("dirac_backprop", results)


if __name__ == "__main__":
    main()
