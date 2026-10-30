from pydantic import BaseModel, Field
from typing import Dict, Any

class PipelineRunRequest(BaseModel):
    place_name: str = Field(
        default="Thore Valley, Gilgit-Baltistan",
        examples=["Thore Valley, Gilgit-Baltistan"]
    )
    pre_event_start: str = Field(default="2026-05-01", examples=["2026-05-01"])
    pre_event_end: str = Field(default="2026-06-01", examples=["2026-06-01"])
    post_event_start: str = Field(default="2026-07-01", examples=["2026-07-01"])
    post_event_end: str = Field(default="2026-08-01", examples=["2026-08-01"])
    buffer_radius_meters: float = Field(
        default=5000.0,
        description="Metric AOI buffer radius in metres"
    )
    use_mock_generation: bool = Field(
        default=True,
        description="Use synthetic NISAR L2 GCOV HDF5 generator if offline"
    )

class PipelineResponse(BaseModel):
    status: str
    message: str
    total_candidate_area_km2: float
    output_archive_path: str
    quality_report: Dict[str, Any]