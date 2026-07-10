"""
Tests for the amortized BayesFlow posterior (src/amortized.py).

These are skipped automatically when the heavy TF/BayesFlow stack is not installed,
so the core suite still runs in a minimal environment. When the stack IS present we
do a *tiny* smoke-train (a few iterations) to exercise the full plumbing --
prior -> simulator -> flow -> sample -> SBC -> save/load -- without asserting on
calibration quality (that needs a full training run).
"""
import numpy as np
import pytest

# Skip the whole module unless TensorFlow + BayesFlow import cleanly.
pytest.importorskip("tensorflow")
pytest.importorskip("bayesflow")

from src.amortized import AmortizedPosteriorModel  # noqa: E402
from src.generative_ga import PARAM_NAMES  # noqa: E402


@pytest.fixture(scope="module")
def tiny_model():
    m = AmortizedPosteriorModel(num_coupling_layers=3)
    m.train(epochs=2, iterations_per_epoch=15, batch_size=32)
    return m


def test_sample_shape_and_envelope(tiny_model):
    s = tiny_model.sample(45.0, n_samples=128)
    assert s.shape == (128, len(PARAM_NAMES))
    assert (s >= tiny_model.bounds[:, 0] - 1e-6).all()
    assert (s <= tiny_model.bounds[:, 1] + 1e-6).all()


def test_fixed_age_is_honored(tiny_model):
    """R7.2: conditioning on a design age pins every sample's age exactly."""
    age_idx = PARAM_NAMES.index("age")
    s = tiny_model.sample(45.0, n_samples=64, age=28.0)
    assert np.allclose(s[:, age_idx], 28.0)


def test_age_none_marginalises(tiny_model):
    """R7.2: age=None draws age from the prior per sample (recovers the marginal)."""
    age_idx = PARAM_NAMES.index("age")
    s = tiny_model.sample(45.0, n_samples=256, age=None)
    ages = s[:, age_idx]
    assert ages.std() > 1.0  # genuinely varying, not a single pinned value
    assert (ages >= tiny_model.bounds[age_idx, 0] - 1e-6).all()
    assert (ages <= tiny_model.bounds[age_idx, 1] + 1e-6).all()


def test_sbc_ranks_shape_and_bounds(tiny_model):
    """SBC covers the 7 SAMPLED params (age is a condition, not sampled)."""
    n_post = 80
    ranks = tiny_model.sbc_ranks(n_datasets=40, n_post=n_post)
    assert ranks.shape == (40, tiny_model.n_theta)
    assert tiny_model.n_theta == len(PARAM_NAMES) - 1
    assert ranks.min() >= 0 and ranks.max() <= n_post


def test_save_load_roundtrip(tmp_path, tiny_model):
    prefix = str(tmp_path / "ckpt")
    tiny_model.save(prefix)
    assert AmortizedPosteriorModel.weights_exist(prefix)

    reloaded = AmortizedPosteriorModel(num_coupling_layers=3)
    assert reloaded.load(prefix)
    s = reloaded.sample(45.0, n_samples=32)
    assert s.shape == (32, len(PARAM_NAMES))


def test_load_detects_stale_dataset(tmp_path, tiny_model, monkeypatch):
    """A flow trained on a different dataset size must be refused as stale."""
    import src.amortized as A
    prefix = str(tmp_path / "ckpt")
    tiny_model.save(prefix)  # records the current dataset fingerprint
    # Simulate calibration changing the dataset (more rows).
    monkeypatch.setattr(A, "dataset_fingerprint", lambda: 10_000_000)
    reloaded = A.AmortizedPosteriorModel(num_coupling_layers=3)
    assert reloaded.load(prefix) is False


def test_load_detects_stale_model(tmp_path, tiny_model, monkeypatch):
    """R7.1 guard: a flow trained against a different forward model is refused."""
    import src.amortized as A
    prefix = str(tmp_path / "ckpt")
    tiny_model.save(prefix)
    monkeypatch.setattr(A, "model_fingerprint", lambda: "deadbeefdeadbeef")
    reloaded = A.AmortizedPosteriorModel(num_coupling_layers=3)
    assert reloaded.load(prefix) is False


def test_load_detects_incompatible_schema(tmp_path, tiny_model):
    """R7.2 guard: a checkpoint from an older conditioning schema (e.g. the
    strength-only flow) must be rejected, not loaded into the wrong-shaped network."""
    prefix = str(tmp_path / "ckpt")
    tiny_model.save(prefix)
    # Rewrite the norm file with a legacy schema tag, preserving every other field.
    d = dict(np.load(prefix + "_norm.npz"))
    d["cond_schema"] = "strength_only_v1"
    np.savez(prefix + "_norm.npz", **d)
    reloaded = AmortizedPosteriorModel(num_coupling_layers=3)
    assert reloaded.load(prefix) is False


def test_explorer_prefers_ga_when_no_weights(monkeypatch):
    """With no trained weights, BayesFlowExplorer must fall back to the GA designer."""
    from src.amortized import AmortizedPosteriorModel as APM
    monkeypatch.setattr(APM, "weights_exist", staticmethod(lambda prefix=None: False))
    from src.bayesian import BayesFlowExplorer
    explorer = BayesFlowExplorer()
    assert explorer.amortized is None
    s = explorer.sample_posterior(45.0, n_samples=50)
    assert s.shape == (50, len(PARAM_NAMES))
    # method="flow" must raise when no flow is trained, like "amortized".
    with pytest.raises(RuntimeError):
        explorer.sample_posterior(45.0, n_samples=50, method="flow")


def test_flow_method_routes_to_amortizer():
    """method='flow' must actually invoke the trained flow, not silently use the GA."""
    from src.bayesian import BayesFlowExplorer
    explorer = BayesFlowExplorer()
    if explorer.amortized is None:
        pytest.skip("no trained amortizer weights available")
    called = {"flow": 0, "ga": 0}
    orig_flow = explorer.amortized.sample
    explorer.amortized.sample = lambda *a, **k: (called.__setitem__("flow", called["flow"] + 1), orig_flow(*a, **k))[1]
    explorer.designer.sample = lambda *a, **k: called.__setitem__("ga", called["ga"] + 1) or np.zeros((k.get("n_samples", 1), len(PARAM_NAMES)))
    explorer.sample_posterior(45.0, n_samples=40, method="flow")
    assert called["flow"] == 1 and called["ga"] == 0


def test_pinned_age_uses_flow_and_honors_age():
    """R7.2 gate: a fixed design age now routes to the flow (not the GA) and the
    returned cloud is conditioned on that exact age."""
    from src.bayesian import BayesFlowExplorer
    explorer = BayesFlowExplorer()
    if explorer.amortized is None:
        pytest.skip("no trained amortizer weights available")
    age_idx = PARAM_NAMES.index("age")
    ga_called = {"n": 0}
    explorer.designer.sample = lambda *a, **k: ga_called.__setitem__("n", ga_called["n"] + 1) or np.zeros((k.get("n_samples", 1), len(PARAM_NAMES)))
    s = explorer.sample_posterior(45.0, n_samples=200, method="auto", age=28.0)
    assert ga_called["n"] == 0, "pinned age should use the flow, not the GA"
    assert np.allclose(s[:, age_idx], 28.0)
