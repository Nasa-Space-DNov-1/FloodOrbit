import os
import json
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_run_flash_flood_pipeline():
    payload = {
        "place_name": "Thore Valley, Gilgit-Baltistan",
        "pre_event_start": "2026-05-01",
        "pre_event_end": "2026-06-01",
        "post_event_start": "2026-07-01",
        "post_event_end": "2026-08-01",
        "buffer_radius_meters": 5000.0,
        "use_mock_generation": True
    }

    response = client.post("/api/v1/run-pipeline", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "SUCCESS"
    # 25x25 flood patch = 625 pixels * 400 m2 = 250,000 m2 = 0.25 km2
    assert data["total_candidate_area_km2"] == 0.25
    assert data["quality_report"]["orbital_check"] == "PASSED"
    assert data["quality_report"]["speckle_pixels_filtered"] == 2
    assert "HHHV" in data["quality_report"]["rejected_complex_terms"]

    # Verify all required artifacts were written to disk
    expected_files = [
        "config.json",
        "data/raw/nisar_pre_event.h5",
        "data/raw/nisar_post_event.h5",
        "outputs/aoi.geojson",
        "outputs/scene_inventory.csv",
        "outputs/candidate_flood_extent.geojson",
        "outputs/flood_area_summary.json",
        "outputs/quality_report.json",
        "outputs/archive.zip"
    ]
    for filepath in expected_files:
        assert os.path.exists(filepath), f"Missing expected output file: {filepath}"

    with open("outputs/flood_area_summary.json", "r") as f:
        summary = json.load(f)
    assert summary["utm_epsg"] == 32643
    assert summary["total_flood_pixels"] == 625
    assert summary["total_debris_pixels"] == 100