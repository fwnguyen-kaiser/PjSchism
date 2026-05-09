"""
IOHMM — Input-Output Hidden Markov Model (Bengio & Frasconi 1996).

Spec (FinancialFrameworkV1.5, §5):
  Transition : P(S_t=j | S_{t-1}=i, U_t) = softmax_j(α_{ij} + β_{ij}^T U_t)
  Emission   : O_t | S_t=s ~ N(μ_s, Σ_s),  Tr(Σ_s) ≤ τ
  Regulariser: J(θ) = -log L(θ) + λ Σ_{ij} ||β_{ij}||²

This module is the public coordinator. Computation is delegated to:
  initialise  — KMeans init + bootstrap τ
  emissions   — log N(·;μ,Σ) + M-step
  transitions — softmax log A + L-BFGS M-step
  inference   — forward/backward, E-step, Viterbi, filter
"""

from __future__ import annotations

import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.special import logsumexp

from schism.models import emissions as em
from schism.models import inference as inf
from schism.models import initialise as ini
from schism.models import transitions as tr
from schism.utils.logger import ingestion_logger

_LOG = ingestion_logger
_EPS = 1e-10

_DEFAULT_LABELS = [f"state_{k}" for k in range(16)]


class IOHMM:
    """
    Input-Output Hidden Markov Model with Gaussian emissions.

    Parameters
    ----------
    n_states        : K — number of latent regimes
    n_obs           : D — dim(O_t)
    n_exog          : M — dim(U_t)
    n_iter          : max EM iterations per run  (model_config → max_iter)
    tol             : ΔLL convergence threshold  (model_config → tol)
    lambda_reg      : L2 penalty on β_{ij}       (model_config → lambda_reg)
    tau_percentile  : bootstrap percentile for τ (model_config → tau_percentile)
    covariance_floor: diagonal floor on Σ_k      (model_config → covariance_floor)
    n_em_runs       : multi-start EM restarts    (model_config → n_em_runs)
    sticky_kappa    : self-transition bias; 1=none, 7=~70% self-transition init (model_config → sticky_kappa)
    random_state    : base reproducibility seed
    rv_col          : index of f7_rv_ratio in O_t; states ordered by this after fit (§2)
    """

    def __init__(
        self,
        n_states: int = 4,
        n_obs: int = 10,
        n_exog: int = 4,
        n_iter: int = 200,
        tol: float = 1e-5,
        lambda_reg: float = 0.01,
        tau_percentile: float = 95.0,
        covariance_floor: float = 0.1,
        n_em_runs: int = 1,
        sticky_kappa: float = 1.0,
        random_state: int = 42,
        rv_col: int = 6,
    ) -> None:
        self.K = n_states
        self.D = n_obs
        self.M = n_exog
        self.n_iter = n_iter
        self.tol = tol
        self.lambda_reg = lambda_reg
        self.tau_percentile = tau_percentile
        self.covariance_floor = covariance_floor
        self.n_em_runs = n_em_runs
        self.sticky_kappa = sticky_kappa
        self.random_state = random_state
        self.rv_col = rv_col

        # Learned parameters
        self.pi: np.ndarray | None = None       # (K,)
        self.mu: np.ndarray | None = None       # (K, D)
        self.sigma: np.ndarray | None = None    # (K, D, D)
        self.alpha: np.ndarray | None = None    # (K, K)
        self.beta: np.ndarray | None = None     # (K, K, M)
        self.tau: float | None = None           # trace ceiling

        self.labels: list[str] = _DEFAULT_LABELS[: self.K]
        self.model_ver: str = ""
        self.ll_history: list[float] = []
        self._fitted: bool = False

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg: dict, **overrides) -> "IOHMM":
        """
        Build an IOHMM from a model_config dict (e.g. loaded from model_config.yaml).

        Expected keys (all optional — fall back to __init__ defaults):
            K, Dim, lambda_reg, tau_percentile, covariance_floor,
            n_em_runs, max_iter, tol

        `overrides` are passed as-is and take priority over the YAML values.
        """
        kwargs: dict = {
            "n_states":        cfg.get("K",                4),
            "n_obs":           cfg.get("Dim",              10),
            "n_iter":          cfg.get("max_iter",         200),
            "tol":             cfg.get("tol",              1e-5),
            "lambda_reg":      cfg.get("lambda_reg",       0.01),
            "tau_percentile":  cfg.get("tau_percentile",   95.0),
            "covariance_floor":cfg.get("covariance_floor", 0.1),
            "n_em_runs":       cfg.get("n_em_runs",        1),
            "sticky_kappa":    cfg.get("sticky_kappa",     1.0),
        }
        kwargs.update(overrides)
        return cls(**kwargs)

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(self, O: np.ndarray, U: np.ndarray) -> "IOHMM":
        """
        Fit via Baum-Welch EM.

        Rows where any O_t component is NaN are dropped before training.
        NaN in U_t is replaced with 0 per component (no exogenous influence).
        When n_em_runs > 1 the EM is restarted with seeds (random_state + run)
        and the run with the highest final LL is kept.
        After convergence, states are reordered by ascending μ_s[rv_col] (§2).
        """
        assert O.shape[0] == U.shape[0]

        valid = ~np.isnan(O).any(axis=1)
        O = O[valid]
        U_safe = inf.safe_U(U[valid])
        T = len(O)

        _LOG.info(
            "iohmm_fit_start",
            T=T, K=self.K, D=self.D, M=self.M,
            n_em_runs=self.n_em_runs,
        )

        best_ll: float = -np.inf
        best_params: dict | None = None
        best_history: list[float] = []

        for run in range(max(1, self.n_em_runs)):
            seed = self.random_state + run
            params = ini.init_params(
                O, self.K, self.M, seed,
                tau_percentile=self.tau_percentile,
                sticky_kappa=self.sticky_kappa,
            )
            pi    = params["pi"]
            mu    = params["mu"]
            sigma = params["sigma"]
            tau   = params["tau"]
            alpha = params["alpha"]
            beta  = params["beta"]

            ll_history: list[float] = []
            prev_ll = -np.inf
            ll_per_bar = -np.inf

            for iteration in range(self.n_iter):
                log_b  = em.log_emission(O, mu, sigma)
                log_A  = tr.log_transition(U_safe, alpha, beta)
                log_pi = np.log(pi + _EPS)

                gamma, xi, log_ll = inf.e_step(log_b, log_A, log_pi)

                mu, sigma, pi = em.m_step_emission(
                    O, gamma, tau, cov_floor=self.covariance_floor
                )
                alpha, beta = tr.m_step_transition(
                    U_safe, xi, alpha, beta, self.lambda_reg,
                    sticky_kappa=self.sticky_kappa,
                )

                ll_per_bar = log_ll / T
                ll_history.append(ll_per_bar)

                if iteration % 10 == 0:
                    _LOG.info(
                        "iohmm_fit_iter",
                        run=run, iteration=iteration,
                        ll_per_bar=round(ll_per_bar, 6),
                    )

                if abs(log_ll - prev_ll) < self.tol * T:
                    _LOG.info(
                        "iohmm_fit_converged",
                        run=run, iteration=iteration,
                        ll_per_bar=round(ll_per_bar, 6),
                    )
                    break
                prev_ll = log_ll

            if ll_per_bar > best_ll:
                best_ll = ll_per_bar
                best_params = {
                    "pi": pi, "mu": mu, "sigma": sigma,
                    "tau": tau, "alpha": alpha, "beta": beta,
                }
                best_history = ll_history

        assert best_params is not None
        self.pi    = best_params["pi"]
        self.mu    = best_params["mu"]
        self.sigma = best_params["sigma"]
        self.tau   = best_params["tau"]
        self.alpha = best_params["alpha"]
        self.beta  = best_params["beta"]
        self.ll_history = best_history
        self._fitted = True
        self.model_ver = datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M%S")

        # §2: order states by ascending mean RV ratio for identifiability
        rv_order = np.argsort(self.mu[:, self.rv_col])
        self._apply_order(rv_order)

        _LOG.info(
            "iohmm_fit_done",
            model_ver=self.model_ver,
            ll_per_bar=round(best_ll, 6),
            rv_order=rv_order.tolist(),
            tau=round(self.tau, 6),
            n_em_runs=self.n_em_runs,
        )
        return self

    # ── Decoding ─────────────────────────────────────────────────────────────

    def decode(self, O: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Viterbi: most-likely state sequence. Returns (T,) int array."""
        U_safe = inf.safe_U(U)
        log_b = em.log_emission(O, self.mu, self.sigma)
        log_A = tr.log_transition(U_safe, self.alpha, self.beta)
        log_pi = np.log(self.pi + _EPS)
        return inf.viterbi(log_b, log_A, log_pi)

    def filter(self, O: np.ndarray, U: np.ndarray) -> np.ndarray:
        """
        Forward-only filter for a full sequence.
        Returns γ_t = P(S_t | O_{1:t}, U_{1:t}) as (T, K).
        """
        U_safe = inf.safe_U(U)
        log_b = em.log_emission(O, self.mu, self.sigma)
        log_A = tr.log_transition(U_safe, self.alpha, self.beta)
        log_pi = np.log(self.pi + _EPS)
        return inf.filter_sequence(log_b, log_A, log_pi)

    def filter_step(
        self,
        log_alpha_prev: np.ndarray,
        o_t: np.ndarray,
        u_t: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        One-step online forward update.

        Parameters
        ----------
        log_alpha_prev : (K,) log forward variable from previous step
        o_t            : (D,) observation at time t
        u_t            : (M,) exogenous at time t (NaN → 0)

        Returns
        -------
        log_alpha_t : (K,) updated log forward variable
        gamma_t     : (K,) P(S_t | O_{1:t}, U_{1:t})
        """
        u_safe = inf.safe_U(u_t[None, :])[0]
        log_b_t = em.log_emission(o_t[None, :], self.mu, self.sigma)[0]
        logits = self.alpha + self.beta @ u_safe           # (K, K)
        log_A_t = logits - logsumexp(logits, axis=1, keepdims=True)
        return inf.filter_step(log_alpha_prev, log_b_t, log_A_t)

    # ── Prediction ───────────────────────────────────────────────────────────

    def predict_state(
        self,
        gamma_T: np.ndarray,
        u_next: Optional[np.ndarray] = None,
        steps: int = 1,
    ) -> np.ndarray:
        """
        Predict state distribution `steps` bars ahead.
        u_next : (M,) exogenous for next bar; None → marginal (zeros).
        Returns (K,) distribution.
        """
        u = inf.safe_U((u_next if u_next is not None else np.zeros(self.M))[None, :])[0]
        logits = self.alpha + self.beta @ u
        A = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
        p = gamma_T.copy()
        for _ in range(steps):
            p = p @ A
        return p

    def predict_obs(self, state_probs: np.ndarray) -> np.ndarray:
        """Expected observation E[O_{t+h}] = Σ_k p_k μ_k. Returns (D,)."""
        return state_probs @ self.mu

    # ── Scoring ──────────────────────────────────────────────────────────────

    def score(self, O: np.ndarray, U: np.ndarray) -> float:
        """Average log-likelihood per bar on held-out data."""
        valid = ~np.isnan(O).any(axis=1)
        O, U = O[valid], U[valid]
        U_safe = inf.safe_U(U)
        log_b = em.log_emission(O, self.mu, self.sigma)
        log_A = tr.log_transition(U_safe, self.alpha, self.beta)
        log_pi = np.log(self.pi + _EPS)
        _, log_ll = inf.forward(log_b, log_A, log_pi)
        return log_ll / len(O)

    def bic(self, O: np.ndarray, U: np.ndarray) -> float:
        """Bayesian Information Criterion — lower is better."""
        T = int((~np.isnan(O).any(axis=1)).sum())
        K, D, M = self.K, self.D, self.M
        n_params = (K - 1) + K * D + K * D * (D + 1) // 2 + K * K + K * K * M
        ll = self.score(O, U) * T
        return -2 * ll + n_params * np.log(T)

    # ── Evaluation criteria (§6.2) ────────────────────────────────────────────

    def log_eval_criteria(self, gamma: np.ndarray) -> None:
        """
        Log §6.2 criteria after fit. Emits WARNING for violations; never raises.
        Checks: state frequency > 5%, mean sojourn ∈ [3, 100] bars.
        """
        states = gamma.argmax(axis=1)

        # Frequency
        for k in range(self.K):
            freq = float((states == k).mean())
            if freq < 0.05:
                _LOG.warning(
                    "eval_state_low_frequency",
                    state=k, label=self.labels[k], freq=round(freq, 4),
                )

        # Sojourn
        runs: dict[int, list[int]] = {k: [] for k in range(self.K)}
        cur, run = int(states[0]), 1
        for s in states[1:]:
            s = int(s)
            if s == cur:
                run += 1
            else:
                runs[cur].append(run)
                cur, run = s, 1
        runs[cur].append(run)

        for k in range(self.K):
            if not runs[k]:
                _LOG.warning("eval_state_never_visited", state=k, label=self.labels[k])
                continue
            mean_soj = float(np.mean(runs[k]))
            if 3 <= mean_soj <= 100:
                _LOG.info("eval_sojourn_ok", state=k, label=self.labels[k], mean_sojourn=round(mean_soj, 2))
            else:
                _LOG.warning("eval_sojourn_out_of_range", state=k, label=self.labels[k], mean_sojourn=round(mean_soj, 2))

    # ── Private convenience wrappers (used by runtime engine) ────────────────

    def _log_emission(self, O: np.ndarray) -> np.ndarray:
        return em.log_emission(O, self.mu, self.sigma)

    def _log_transition(self, U_safe: np.ndarray) -> np.ndarray:
        return tr.log_transition(U_safe, self.alpha, self.beta)

    def _safe_U(self, U: np.ndarray) -> np.ndarray:
        return inf.safe_U(U)

    def _forward(
        self, log_b: np.ndarray, log_A: np.ndarray
    ) -> tuple[np.ndarray, float]:
        return inf.forward(log_b, log_A, np.log(self.pi + _EPS))

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _apply_order(self, order: np.ndarray) -> None:
        """Reorder all state-indexed parameters in-place."""
        self.pi = self.pi[order]
        self.mu = self.mu[order]
        self.sigma = self.sigma[order]
        self.labels = [self.labels[i] for i in order]
        self.alpha = self.alpha[np.ix_(order, order)]
        self.beta = self.beta[np.ix_(order, order)]

    # ── Serialisation ────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)
        _LOG.info("iohmm_saved", path=str(path), model_ver=self.model_ver)

    @classmethod
    def load(cls, path: str | Path) -> "IOHMM":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls.__new__(cls)
        obj.__dict__.update(state)
        _LOG.info("iohmm_loaded", path=str(path), model_ver=obj.model_ver)
        return obj
