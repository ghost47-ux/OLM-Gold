"""Ingestion planning is network-free and gates remain closed by default."""

import pandas as pd
import pytest

from olm.data.credentials import AuthorizationGates
from olm.data.ingestion import IngestionService


def test_plan_needs_no_client_or_gate(tmp_path):
    service = IngestionService(None, data_root=tmp_path / "data")
    plan = service.plan(
        symbol="XAU/USD",
        interval="5min",
        start=pd.Timestamp("2025-01-01T00:00:00Z"),
        end=pd.Timestamp("2025-01-03T00:00:00Z"),
    )
    assert plan["network_calls"] == 0
    assert plan["planned_windows"] >= 1
    assert plan["pending_windows"] == plan["planned_windows"]
    assert plan["first_window"].startswith("2025-01-01")


def test_download_still_requires_retention_gate(tmp_path):
    service = IngestionService(None, data_root=tmp_path / "data")
    with pytest.raises(PermissionError, match="TD_DATA_RETENTION_APPROVED"):
        service.download(
            symbol="XAU/USD",
            interval="5min",
            start=pd.Timestamp("2025-01-01T00:00:00Z"),
            end=pd.Timestamp("2025-01-02T00:00:00Z"),
            gates=AuthorizationGates(),
        )
