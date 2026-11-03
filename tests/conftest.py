import pytest

from sfr.config import load_config


@pytest.fixture
def cfg(tmp_path):
    """Real configs, but synthetic data and all outputs inside tmp_path."""
    return load_config(overrides=[
        "data.default_source=synthetic",
        "data.assets={SP500: {symbol: SPX}, AAPL: {symbol: AAPL}}",
        'settings.sample.start="2015-01-01"',
        'data.sources.synthetic.end="2021-12-31"',
        'settings.sample.train_end="2019-12-31"',
        f"settings.paths.raw={tmp_path / 'raw'}",
        f"settings.paths.processed={tmp_path / 'processed'}",
        f"settings.paths.external={tmp_path / 'external'}",
        f"settings.paths.reports={tmp_path / 'reports'}",
    ])
