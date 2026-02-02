"""
In-memory cache for AI-generated medical reports.

Purpose:
- Store the latest generated report
- Track whether report generation is complete
- Prevent stale or placeholder reports from leaking
"""
import threading


_latest_report: str | None = None
_report_ready: bool = False
_report_lock = threading.Lock()


def reset_report():
    """
    Call this BEFORE starting a new report generation.
    Clears old report and marks status as not ready.
    """
    global _latest_report, _report_ready
    _latest_report = None
    _report_ready = False


def save_report(report: str):
    global _latest_report, _report_ready
    with _report_lock:
        _latest_report = report
        _report_ready = True



def get_report() -> str | None:
    """
    Returns the latest report text if available, else None.
    """
    return _latest_report


def is_report_ready() -> bool:
    with _report_lock:
        return _report_ready

