"""KPI analysis of simulation results."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np



def analyze(
    process_log: np.ndarray,
    order_log: List[Dict],
    label: str = "simulation",
    sim_duration: Optional[float] = None,
    num_machines: Optional[int] = None,
    rework_station_id: Optional[str] = None,
    *,
    report: bool = False,
) -> Dict[str, float]:
    """KPI panel for one run; prints a formatted report when report=True.

    ``rework_station_id`` names the optional stage (e.g. M5, a class-specific
    task or audit rather than classic rework). ``rework_rate`` is then the
    percentage of orders that visit it at least once and ``multi_rework_rate``
    the percentage visiting it twice or more; None leaves both at 0.0.
    """
    completed = [o for o in order_log if o.get("timestamp_completion") is not None]
    n_valid = len(completed)
    if n_valid == 0:
        print(f"Results Analysis for {label}: no completed orders.")
        return {}

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

        if rework_station_id is not None:
            n_rework_visits = sum(
                1 for e in machine_events if e["station"] == rework_station_id
            )
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
    n_productive = len([r for r in machine_records if not r["is_breakdown"]])
    bd_rate = (n_breakdowns / n_productive * 100) if n_productive > 0 else 0.0
    n_buffer_events = sum(1 for r in _iter_records(process_log) if r["station_type"] == "buffer")

    if report:
        _print_report(
            label=label, n_valid=n_valid, ct=ct, bw=bw, st=st, pt=pt, iw=iw, bt=bt,
            fe_cycle=fe_cycle, fe_system=fe_system,
            rework_rate=rework_rate, multi_rework_rate=multi_rework_rate,
            rework_station_id=rework_station_id,
            n_breakdowns=n_breakdowns, bd_rate=bd_rate, avg_breakdown_repair=avg_breakdown_repair,
            tech_availability=tech_availability, n_buffer_events=n_buffer_events,
        )

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


def _compute_availability(
    total_breakdown_time: float,
    sim_duration: Optional[float],
    num_machines: Optional[int],
) -> float:
    if sim_duration is None or num_machines is None or num_machines == 0:
        return 100.0
    total_machine_time = sim_duration * num_machines
    if total_machine_time <= 0:
        return 100.0
    return (total_machine_time - total_breakdown_time) / total_machine_time * 100


def _print_report(
    label, n_valid, ct, bw, st, pt, iw, bt,
    fe_cycle, fe_system,
    rework_rate, multi_rework_rate, rework_station_id,
    n_breakdowns, bd_rate, avg_breakdown_repair,
    tech_availability, n_buffer_events,
) -> None:
    sep = "-" * 60
    print(f"\nResults Analysis for {label} ({n_valid:,} valid jobs):")
    print(sep)
    print(f"{'Metric':<22}| {'Mean':>10} | {'Std Dev':>10}")
    print(sep)
    print(f"{'Cycle Time':<22}| {np.mean(ct):>10.2f} | {np.std(ct):>10.2f}   (FE: {fe_cycle:.1f}%)")
    print(f"{'Buffer Wait':<22}| {np.mean(bw):>10.2f} | {np.std(bw):>10.2f}")
    print(f"{'System Time':<22}| {np.mean(st):>10.2f} | {np.std(st):>10.2f}   (FE: {fe_system:.1f}%)")
    print(f"{'Processing':<22}| {np.mean(pt):>10.2f} | {np.std(pt):>10.2f}")
    print(f"{'Inline Wait':<22}| {np.mean(iw):>10.2f} | {np.std(iw):>10.2f}")
    print(f"{'Buffer Time (total)':<22}| {np.mean(bt):>10.2f} | {np.std(bt):>10.2f}")
    print(sep)
    if rework_station_id is not None:
        print(f"Rework Rate ({rework_station_id} ≥1×):  {rework_rate:.2f}%")
        print(f"Multi-Rework Rate ({rework_station_id} ≥2×): {multi_rework_rate:.2f}%")
    else:
        print("Rework Rate: n/a (no rework_station_id provided)")
    print(f"Buffer Events: {n_buffer_events:,}")
    print(f"{'Reliability':<22}| {'Value':>10}")
    print(sep)
    print(f"{'Breakdowns':<22}| {n_breakdowns:>10}   (Rate: {bd_rate:.2f}%)")
    print(f"{'Avg Repair Time':<22}| {avg_breakdown_repair:>10.2f}")
    print(f"{'Tech. Availability':<22}| {tech_availability:>9.2f} %")
    print(sep)


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
