"""Real-inference smoke coverage for the supported foundation forecast stack."""

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

pytest.importorskip("torch", reason="foundation smoke requires real torch")
pytest.importorskip("chronos", reason="foundation smoke requires chronos-forecasting")


def test_chronos_tiny_runs_real_cpu_inference():
    """Exercise model loading, tensor preparation, inference, and reconstruction."""
    script = textwrap.dedent(
        """
        import json
        import numpy as np
        import pandas as pd
        import torch

        from mtdata.forecast.methods.pretrained import ChronosBoltMethod

        torch.manual_seed(0)
        result = ChronosBoltMethod().forecast(
            pd.Series(np.linspace(100.0, 110.0, 32)),
            horizon=2,
            seasonality=1,
            params={
                "model_name": "amazon/chronos-t5-tiny",
                "device_map": "cpu",
                "context_length": 32,
                "quantiles": [0.5],
            },
        )
        print(json.dumps({
            "forecast": result.forecast.tolist(),
            "pipeline": result.params_used["pipeline"],
            "torch_version": torch.__version__,
        }))
        """
    )
    env = os.environ.copy()
    env.update(
        {
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=False,
        env=env,
        text=True,
        timeout=180,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    forecast = payload["forecast"]

    assert len(forecast) == 2
    assert all(np.isfinite(value) for value in forecast)
    assert any(not np.isclose(value, 0.0) for value in forecast)
    assert payload["pipeline"] == "ChronosPipeline"
    assert payload["torch_version"]
