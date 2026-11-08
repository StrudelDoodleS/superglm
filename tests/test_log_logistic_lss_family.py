import pickle
import subprocess
import sys
from importlib import import_module, util

import numpy as np
import pandas as pd
import pytest

from superglm import SuperLSS
from superglm.distributional.family import (
    COMPLETE_OBSERVATION,
    DefaultPredictionFamily,
    DistributionFunctionFamily,
    ExpectedInformationFamily,
    ExpectedShortfallFamily,
    PredictorCurvatureDirectionalFamily,
    PredictorCurvatureSecondDirectionalFamily,
    ResponseWindowFamily,
    bounded_observation,
)
from superglm.distributional.observation import ResponseBounds
from tests.test_declared_scalar_adapter import weights


@pytest.fixture
def family():
    name = "superglm.distributional.families.log_logistic"
    assert util.find_spec(name) is not None, "public LogLogisticLSS is missing"
    return import_module(name).LogLogisticLSS()


def test_log_logistic_reuses_adapter_without_false_optional_capabilities(family):
    from superglm import LogLogisticLSS, LomaxLSS
    from superglm.distributional.families._declared import DeclaredScalarFamily

    assert (
        LogLogisticLSS.evaluate_natural
        is LomaxLSS.evaluate_natural
        is DeclaredScalarFamily.evaluate_natural
    )
    for protocol in (
        DistributionFunctionFamily,
        DefaultPredictionFamily,
        ResponseWindowFamily,
        PredictorCurvatureDirectionalFamily,
        PredictorCurvatureSecondDirectionalFamily,
    ):
        assert isinstance(family, protocol)
    assert not isinstance(family, ExpectedInformationFamily)
    assert not isinstance(family, ExpectedShortfallFamily)
    assert not family.capabilities.expected_information
    with pytest.raises(ValueError, match="support"):
        family.bind_likelihood(np.zeros(1), weights([1]), COMPLETE_OBSERVATION)
    # Right censoring at zero is whole support, and never an exact zero density.
    obs = bounded_observation(
        ResponseBounds.from_arrays(np.zeros(1), np.full(1, np.inf), np.ones(1, int))
    )
    plan = family.bind_likelihood(np.zeros(1), weights([2]), obs)
    actual = family.evaluate_natural(np.zeros(1), np.ones((1, 2)), plan)
    assert actual.valid.all() and actual.reported_log_likelihood[0] == 0.0


def test_log_logistic_bounded_carriers_and_all_derivatives(family):
    import mpmath as mp

    from tests._declared_scalar_oracles import assert_scaled_jets, jets, log_density, log_window

    y = np.array([0.5, 1.0, 1.5, 0.7])
    lo, hi = np.array([0.0, 0.2, 0.3, 0.0]), np.array([np.inf, 3.0, np.inf, 2.0])
    side = np.array([0, 1, -1, 0])
    mass = np.array([1, 2, 5, 2])
    plan = family.bind_likelihood(
        y, weights(mass), bounded_observation(ResponseBounds.from_arrays(lo, hi, side))
    )
    theta = np.tile([2.0, 1.5], (len(y), 1))
    out = family.evaluate_natural(y, theta, plan)
    third, fourth = family._higher(y, theta, plan, 4)
    for i in range(len(y)):

        def law(s, a):
            observed = (
                log_density("log_logistic", mp.mpf(y[i]), s, a)
                if side[i] == 0
                else log_window(
                    "log_logistic",
                    mp.mpf(y[i] if side[i] > 0 else lo[i]),
                    mp.mpf(hi[i] if side[i] > 0 else y[i]),
                    s,
                    a,
                )
            )
            return int(mass[i]) * (
                observed - log_window("log_logistic", mp.mpf(lo[i]), mp.mpf(hi[i]), s, a)
            )

        ref = jets(law, theta[i], 4)
        assert_scaled_jets(
            (
                out.reported_log_likelihood[i : i + 1],
                out.score[i],
                out.hessian_packed[i],
                third[i],
                fourth[i],
            ),
            ref,
            theta[i],
            factor=8192,
        )
    np.testing.assert_allclose(
        out.parameter_independent_carrier, np.where(side == 0, -mass * np.log(y), 0.0)
    )
    unit = family.bind_likelihood(y, weights(np.ones(len(y)), "prior"), plan.observation)
    single = family.evaluate_natural(y, theta, unit)
    np.testing.assert_allclose(out.reported_log_likelihood, mass * single.reported_log_likelihood)


def test_carrier_rounding_is_separate_from_exact_plan_identity(family, monkeypatch):
    from dataclasses import replace

    from superglm.distributional.families._base import immutable
    from superglm.distributional.likelihood_identity import rebound_plan_mismatch

    y = np.array([0.2, 3.0, 4.0])
    mass = weights([1, 2, 5])
    original = family.bind_likelihood(y, mass, COMPLETE_OBSERVATION)
    stored_sum = float(original.parameter_independent_carrier.sum())
    carrier = family._kernel.carrier
    # A faithful different libm can change a log-derived carrier by an ulp.
    monkeypatch.setattr(family._kernel, "carrier", lambda y, m: np.nextafter(carrier(y, m), np.inf))
    live = family.bind_likelihood(y, mass, COMPLETE_OBSERVATION)
    assert live.carrier_digest != original.carrier_digest
    assert live.plan_identifier == original.plan_identifier
    assert rebound_plan_mismatch(live, original.plan_identifier, stored_sum) is None
    altered = replace(
        live, parameter_independent_carrier=immutable(live.parameter_independent_carrier + 0.01)
    )
    assert rebound_plan_mismatch(altered, original.plan_identifier, stored_sum) == "carrier"
    # Rounding tolerance does not permit any response or weight change.
    other = family.bind_likelihood(y + 0.01, mass, COMPLETE_OBSERVATION)
    assert rebound_plan_mismatch(other, original.plan_identifier, stored_sum) == "identifier"


@pytest.mark.parametrize("bounded", [False, True])
def test_public_families_fresh_process_serialization_without_development_imports(
    tmp_path, family, bounded
):
    import joblib

    from superglm import LomaxLSS

    rng = np.random.default_rng(401)
    frame = pd.DataFrame({"unused": np.zeros(160)})
    for f in (family, LomaxLSS()):
        y = f.quantile(rng.uniform(0.01, 0.99, len(frame)), np.tile([2.0, 3.0], (len(frame), 1)))
        options = {}
        if bounded:
            options = {"truncation": (0.0, 8.0), "censored": y > 3.0}
            y = np.minimum(y, 3.0)
        model = SuperLSS(f, f.scale(), f.shape()).fit(frame, y, **options)
        artifact = tmp_path / (type(f).__name__ + ".pkl")
        artifact.write_bytes(pickle.dumps(model))
        jobfile = artifact.with_suffix(".joblib")
        joblib.dump(model, jobfile)
        plan = f.bind_likelihood(y, weights(np.ones(len(y))), COMPLETE_OBSERVATION)
        restored = pickle.loads(pickle.dumps(plan))
        assert restored.plan_identifier == plan.plan_identifier
        f.validate_likelihood_plan(y, restored)
        np.testing.assert_array_equal(joblib.load(jobfile).predict(frame), model.predict(frame))
        script = """
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('sympy','mpmath','tools','notes'):
            raise AssertionError('runtime imported development dependency: '+fullname)
sys.meta_path.insert(0, Block())
import pickle, joblib, numpy as np, pandas as pd
from superglm import LomaxLSS, LogLogisticLSS
m=pickle.load(open(sys.argv[1],'rb'))
other=joblib.load(sys.argv[2])
frame=pd.DataFrame({'unused':np.zeros(2)})
np.testing.assert_array_equal(m.predict(frame),other.predict(frame))
assert m.family.to_config()==other.family.to_config()
assert np.isfinite(m.predict_quantile(frame,.9)).all()
"""
        subprocess.run(
            [sys.executable, "-c", script, str(artifact), str(jobfile)],
            check=True,
            capture_output=True,
            text=True,
        )
