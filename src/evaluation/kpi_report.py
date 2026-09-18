"""KPI analysis of simulation results."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import numpy as np



def analyze(
    process_log: np.ndarray,
    order_log: List[Dict],
    *,
    sim_duration: float,
    num_machines: int,
    rework_station_id: str,
) -> Dict[str, float]:
    """KPI panel for one run.

    ``rework_station_id`` names the optional stage (e.g. M5, a class-specific
    task or audit rather than classic rework). ``rework_rate`` is the
    percentage of orders that visit it at least once and ``multi_rework_rate``
    the percentage visiting it twice or more.
    """
    completed = [o for o in order_log if o.get("timestamp_completion") is not None]
    n_valid = len(completed)
    if n_valid == 0:
        raise ValueError("Fail fast: no completed orders; the KPI panel is undefined.")

    events_by_order = _group_events_by_order(process_log)
    cycle_times, buffer_waits, system_times = [], [], []
    processing_times, inline_waits, total_buffer_times = [], [], []
    rework_count = 0
    multi_rework_count = 0

    for order in completed:
        oid = order["order_id"]
        t_creation = float(order["timestamp_creation"])
        t_completion = float(order["timestamp_completion"])
        order_events = events_by_order.get(oid, [])
        if not order_events:
            continue
        machine_events = [e for e in order_events if e["station_type"] == "machine"]
        buffer_events = [e for e in order_events if e["station_type"] == "buffer"]
        if not machine_events:
            continue
        first_machine_start = min(e["timestamp_event_start"] for e in machine_events)
        total_processing = sum(e["net_process_time"] for e in machine_events)
        total_buffer_time = sum(e["time_processing"] for e in buffer_events)
        system_time = t_completion - t_creation
        buffer_wait = first_machine_start - t_creation
        cycle_time = t_completion - first_machine_start
        inline_wait = max(0.0, cycle_time - total_processing)
        system_times.append(system_time)
        cycle_times.append(cycle_time)
        buffer_waits.append(buffer_wait)
        processing_times.append(total_processing)
        inline_waits.append(inline_wait)
        total_buffer_times.append(total_buffer_time)

        n_rework_visits = sum(1 for e in machine_events if e["station"] == rework_station_id)
        if n_rework_visits >= 1:
            rework_count += 1
        if n_rework_visits >= 2:
            multi_rework_count += 1

    ct = np.array(cycle_times)
    bw = np.array(buffer_waits)
    st = np.array(system_times)
    pt = np.array(processing_times)
    iw = np.array(inline_waits)
    bt = np.array(total_buffer_times)

    fe_cycle = (np.mean(pt) / np.mean(ct) * 100) if np.mean(ct) > 0 else 0.0
    fe_system = (np.mean(pt) / np.mean(st) * 100) if np.mean(st) > 0 else 0.0
    rework_rate = rework_count / n_valid * 100
    multi_rework_rate = multi_rework_count / n_valid * 100

    machine_records = [r for r in _iter_records(process_log) if r["station_type"] == "machine"]
    breakdown_events = [r for r in machine_records if r["is_breakdown"]]
    breakdown_repair_times = [r["repair_time"] for r in breakdown_events]
    n_breakdowns = len(breakdown_events)
    avg_breakdown_repair = np.mean(breakdown_repair_times) if breakdown_repair_times else 0.0
    total_breakdown_time = sum(breakdown_repair_times)
    tech_availability = _compute_availability(total_breakdown_time, sim_duration, num_machines)
    n_buffer_events = sum(1 for r in _iter_records(process_log) if r["station_type"] == "buffer")

    return {
        "n_valid": n_valid,
        "cycle_time_mean": float(np.mean(ct)), "cycle_time_std": float(np.std(ct)),
        "buffer_wait_mean": float(np.mean(bw)), "system_time_mean": float(np.mean(st)),
        "processing_mean": float(np.mean(pt)), "inline_wait_mean": float(np.mean(iw)),
        "buffer_time_mean": float(np.mean(bt)),
        "fe_cycle": fe_cycle, "fe_system": fe_system,
        "rework_rate": rework_rate,
        "multi_rework_rate": multi_rework_rate,
        "n_breakdowns": n_breakdowns,
        "avg_breakdown_repair": avg_breakdown_repair,
        "tech_availability": tech_availability,
        "n_buffer_events": n_buffer_events,
    }


def _group_events_by_order(process_log: np.ndarray) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in process_log:
        grouped[str(row["order_id"])].append({
            "timestamp_event_start": float(row["timestamp_event_start"]),
            "station": str(row["station"]), "station_type": str(row["station_type"]),
            "time_processing": float(row["time_processing"]),
            "net_process_time": float(row["net_process_time"]),
        })
    return grouped


def _iter_records(process_log: np.ndarray):
    for row in process_log:
        yield {
            "station_type": str(row["station_type"]),
            "is_breakdown": bool(row["is_breakdown"]),
            "repair_time": float(row["repair_time"]),
        }


def _compute_availability(total_breakdown_time: float, sim_duration: float,
                          num_machines: int) -> float:
    total_machine_time = sim_duration * num_machines
    if total_machine_time <= 0:
        return 100.0
    return (total_machine_time - total_breakdown_time) / total_machine_time * 100


def extract_cycle_times(
    process_log: np.ndarray, order_log: list[dict],
) -> np.ndarray:
    """CT = t_completion - first machine start, per completed order.

    Same definition as the cycle_time in ``analyze``, which computes it
    alongside its other per-order metrics.
    """
    completed = [o for o in order_log if o.get("timestamp_completion") is not None]
    events_by_order: dict[str, list] = defaultdict(list)
    for row in process_log:
        if str(row["station_type"]) == "machine":
            events_by_order[str(row["order_id"])].append(
                float(row["timestamp_event_start"])
            )
    cts = []
    for order in completed:
        starts = events_by_order.get(order["order_id"], [])
        if not starts:
            continue
        cts.append(float(order["timestamp_completion"]) - min(starts))
    return np.array(cts)
