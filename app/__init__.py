"""Integration & UI layer for Argus Panoptes.

Planned (Day 4-5): a FastAPI service (`/infer`, `/batch`) emitting structured
JSON payloads for the downstream cost/nesting optimizer (cycle-time factor,
quality score, recommendations), a ``StreamingPerceptor`` for real-time chunk
processing, mock PLC/OPC-UA tags, and a Streamlit dashboard for live simulation
control and monitoring.

Scaffold only in v1.
"""

__version__ = "0.0.0"
