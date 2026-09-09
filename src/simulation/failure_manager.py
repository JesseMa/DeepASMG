"""FailureManager – TTF-decrement failure control.

One TTF is sampled per cycle and then decremented deterministically by each
job's processing time, so all stochasticity of the failure process lives in
that single sample. Survival and repair are independent strategy slots with
separable random streams.
"""

from __future__ import annotations

from typing import Callable, Dict, TYPE_CHECKING

from src.simulation.event import EventType

if TYPE_CHECKING:
    from src.dynamics.foundation_dynamics import SurvivalStrategy, RepairStrategy
    from src.recording.recorder import Recorder
    from src.simulation.event import EventQueue
    from src.simulation.station import Station


class FailureManager:
    def __init__(
        self,
        survival_strategy: "SurvivalStrategy",
        repair_strategy: "RepairStrategy",
    ) -> None:
        self._survival = survival_strategy
        self._repair = repair_strategy

    def initialize(self, stations: Dict[str, "Station"]) -> None:
        """Sample initial TTFs for all stations (after station reset)."""
        for station in stations.values():
            self._start_new_cycle(station, current_time=0.0)

    def check_after_job(
        self,
        station: "Station",
        process_time: float,
        current_time: float,
        events: "EventQueue",
        allow_breakdown: bool = True,
    ) -> None:
        """Decrement TTF and account wear after a completed job.

        allow_breakdown=False is used for jobs that finish while the machine is
        already under repair: their wear belongs to the closing cycle (the cycle
        snapshot happens at repair end), but no second breakdown may be
        scheduled.
        """
        if station.ttf_remaining is None:
            return

        station.ttf_remaining -= process_time
        station.accumulated_op_time += process_time
        station.jobs_in_cycle += 1

        if station.ttf_remaining <= 0 and allow_breakdown:
            self._schedule_breakdown(station, current_time, events)

    def handle_repaired(
        self,
        event,
        station: "Station",
        recorder: "Recorder",
        current_time: float,
        on_slot_freed_fn: Callable[["Station"], None],
    ) -> None:
        station.is_down = False

        recorder.record_process_step(
            order_id="BREAKDOWN",
            timestamp_start=event.data["breakdown_start"],
            station_id=station.id,
            station_type=station.station_type,
            process_time=event.data["repair_time"],
            is_breakdown=True,
            repair_time=event.data["repair_time"],
        )

        # Update cycle history BEFORE sampling the new TTF
        self._survival.notify_cycle_end(
            station_id=station.id,
            ttf=station.accumulated_op_time,
            n_jobs=station.jobs_in_cycle,
            repair_time=event.data["repair_time"],
        )
        self._repair.notify_cycle_end(
            station_id=station.id,
            ttf=station.accumulated_op_time,
            n_jobs=station.jobs_in_cycle,
            repair_time=event.data["repair_time"],
        )

        self._start_new_cycle(station, current_time=current_time)

        on_slot_freed_fn(station)

    def _start_new_cycle(self, station: "Station", current_time: float) -> None:
        ttf = self._survival.sample_time_to_failure(station.id, current_time=current_time)
        station.ttf_remaining = ttf
        station.accumulated_op_time = 0.0
        station.cycle_start_wall_time = current_time
        station.jobs_in_cycle = 0

    def _schedule_breakdown(
        self,
        station: "Station",
        current_time: float,
        events: "EventQueue",
    ) -> None:
        station.is_down = True

        utilization = station.get_utilization(current_time)
        repair_time = self._repair.predict_repair_time(
            station_id=station.id,
            operating_time_since_last=station.accumulated_op_time,
            utilization=utilization,
            current_time=current_time,
        )

        events.schedule(
            time=current_time + repair_time,
            event_type=EventType.MACHINE_REPAIRED,
            station_id=station.id,
            data={
                "repair_time": repair_time,
                "breakdown_start": current_time,
            },
        )