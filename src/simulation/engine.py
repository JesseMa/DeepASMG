"""SimulationEngine – time-discrete simulation at one-second resolution.

Departure discipline: strict FIFO — only the front order of a departure
queue may leave; all others wait even if their target station is free.

Pull model: the start station is filled initially and replenished whenever a
slot is freed.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm

from src.config.routing_keys import station_admissible_targets
from src.config.schema import ProcessConfig, StationConfig
from src.config.simulation_config import SimulationConfig
from src.recording.recorder import Recorder
from src.simulation.deadlock_manager import DeadlockManager
from src.simulation.event import EventQueue, EventType
from src.simulation.failure_manager import FailureManager
from src.simulation.order import Order
from src.simulation.station import Station


class SimulationEngine:
    def __init__(self, process_config: ProcessConfig, sim_config: SimulationConfig) -> None:
        self._process_config, self._sim_config = process_config, sim_config
        self._process_time = sim_config.process_time_strategy
        self._transition = sim_config.transition_strategy
        self._product = sim_config.product_strategy

        self._deadlock = DeadlockManager()
        self._availability = FailureManager(
            survival_strategy=sim_config.survival_strategy,
            repair_strategy=sim_config.repair_strategy,
        )

        self._stations: Dict[str, Station] = {}
        self._events = EventQueue()
        self._recorder = Recorder()
        self._current_time: int = 0
        self._order_counter: int = 0
        self._is_done: bool = False

        self._cascade_queue: deque[Station] = deque()
        self._cascade_in_queue: Set[str] = set()
        self._is_cascading: bool = False

        self._routing_targets: Dict[str, Set[str]] = self._build_routing_targets()

    def _build_routing_targets(self) -> Dict[str, Set[str]]:
        return {sc.id: station_admissible_targets(sc)
                for sc in self._process_config.stations}

    def reset(self) -> None:
        self._current_time = 0
        self._order_counter = 0
        self._is_done = False
        self._events.clear()
        self._recorder.reset()

        self._deadlock.reset()
        self._cascade_queue.clear()
        self._cascade_in_queue.clear()
        self._is_cascading = False

        self._stations.clear()
        for sc in self._process_config.stations:
            self._stations[sc.id] = Station(sc)

        # One dict with run-long lifetime: the strategies keep the reference.
        station_configs: Dict[str, StationConfig] = {
            sc.id: sc for sc in self._process_config.stations
        }
        self._process_time.initialize(station_configs)
        self._transition.initialize(station_configs)
        self._sim_config.survival_strategy.initialize(station_configs)
        self._sim_config.repair_strategy.initialize(station_configs)
        self._product.initialize(
            self._process_config.product_features,
            temporal_modulation=self._process_config.temporal_modulation,
            markov_alphas=self._process_config.markov_alphas,
        )

        self._availability.initialize(self._stations)

        self._create_initial_orders()

    def step(self) -> None:
        """Advance the simulation by one second."""
        if self._is_done:
            raise RuntimeError("Simulation has ended. Please call reset().")

        self._deadlock.reset_step_counter()

        while self._events and self._events.peek_time() <= self._current_time:
            event = self._events.pop()
            self._process_event(event)

        self._current_time += 1

        if self._current_time >= self._sim_config.max_time:
            self._is_done = True

    def run(self, label: str = "simulation") -> Tuple[np.ndarray, List[Dict]]:
        self.reset()

        with tqdm(total=self._sim_config.total_steps, desc=label, unit="step") as pbar:
            while not self._is_done:
                self.step()
                pbar.update(1)

        if self._deadlock.deadlock_count > 0:
            print(
                f"  Deadlocks resolved: {self._deadlock.deadlock_count}, "
                f"Remaining in overflow: {self._deadlock.overflow_size}"
            )

        return (
            self._recorder.get_process_log_array(warmup_time=self._sim_config.warmup_time),
            self._recorder.get_order_log(warmup_time=self._sim_config.warmup_time),
        )

    def _process_event(self, event) -> None:
        if event.type == EventType.PROCESS_COMPLETE:
            self._handle_process_complete(event)
        elif event.type == EventType.MACHINE_REPAIRED:
            self._availability.handle_repaired(
                event,
                station=self._stations[event.station_id],
                recorder=self._recorder,
                current_time=float(self._current_time),
                on_slot_freed_fn=self._on_slot_freed,
            )

    def _accept_order(self, station: Station, order: Order) -> None:
        order.visits[station.id] = order.visits.get(station.id, 0) + 1
        process_time = self._get_process_time(station, order)
        station.start_processing(order)

        self._events.schedule(
            time=self._current_time + process_time,
            event_type=EventType.PROCESS_COMPLETE,
            station_id=station.id,
            order_id=order.id,
            data={
                "start_time": float(self._current_time),
                "process_time": process_time,
            },
        )

    def _handle_process_complete(self, event) -> None:
        station = self._stations[event.station_id]
        order = station.finish_processing(event.order_id)
        config = station.config

        self._recorder.record_process_step(
            order_id=order.id,
            timestamp_start=event.data["start_time"],
            station_id=station.id,
            station_type=station.station_type,
            process_time=event.data["process_time"],
        )

        # Must run BEFORE the freed-slot cascade: a machine whose TTF is
        # exhausted by this job has to be marked down before any waitlisted
        # order can be admitted.
        if config.mttr > 0:
            self._availability.check_after_job(
                station,
                process_time=event.data["process_time"],
                current_time=float(self._current_time),
                events=self._events,
                allow_breakdown=not station.is_down,
            )

        target_id = self._transition.predict(
            station.id, order,
            available_targets=self._routing_targets.get(station.id),
            current_time=float(self._current_time),
        )

        if target_id is None:
            self._recorder.complete_order(order.id, float(self._current_time))
            self._on_slot_freed(station)
        else:
            station.add_to_departure(order, target_id)
            freed_slots = self._drain_departure_fifo(station)
            if freed_slots > 0:
                self._on_slot_freed(station)

    def _get_process_time(self, station: Station, order: Order) -> float:
        config = station.config
        if config.process_times:
            return self._process_time.predict(station.id, order, float(self._current_time))
        if config.is_machine:
            raise ValueError(
                f"CRITICAL ERROR: Station '{station.id}' is marked as a machine "
                f"but has no entries in 'process_times'. \n"
                f"This would lead to an invalid process time of 0s."
            )
        return float(config.transit_time)

    def _drain_departure_fifo(self, station: Station) -> int:
        """Dispatch from the departure queue, FIFO: only the front order may leave.

        Returns the number of slots freed, counting overflow displacements.
        """
        freed_slots = 0
        while station.has_departure:
            order, target_id = station.peek_departure()
            target = self._stations.get(target_id)

            if target is None:
                raise RuntimeError(
                    f"Drain-FIFO: target station '{target_id}' for order "
                    f"'{order.id}' does not exist in self._stations (source: "
                    f"'{station.id}'). Topology or TransitionStrategy bug "
                    f"— predict() should have returned target_id=None when "
                    f"the target is unreachable. Known stations: "
                    f"{sorted(self._stations.keys())}."
                )

            if target.is_available:
                station.pop_departure()
                self._accept_order(target, order)
                freed_slots += 1
            else:
                cycle = self._deadlock.detect_cycle(self._stations, station.id, target_id)
                if cycle is not None:
                    resolved = self._deadlock.resolve_deadlock(
                        station, self._recorder, float(self._current_time),
                    )
                    if resolved:
                        freed_slots += 1
                        continue

                target.add_to_waitlist(station.id)
                break

        return freed_slots


    def _on_slot_freed(self, station: Station) -> None:
        """Trigger the freed-slot cascade iteratively with deduplication (avoids recursion)."""
        if station.id not in self._cascade_in_queue:
            self._cascade_queue.append(station)
            self._cascade_in_queue.add(station.id)

        if self._is_cascading:
            return

        self._is_cascading = True
        try:
            while self._cascade_queue:
                curr = self._cascade_queue.popleft()
                self._cascade_in_queue.discard(curr.id)
                self._process_freed_slot(curr)
        finally:
            self._is_cascading = False

    def _process_freed_slot(self, station: Station) -> None:
        self._drain_departure_fifo(station)

        if station.config.is_start_station:
            # One replacement per freed slot: a single cascade pass can free
            # several start-station slots, so replenish until capacity is
            # restored (bounded by capacity as a hard stop).
            for _ in range(station.config.capacity):
                if not station.is_available:
                    break
                self._replenish_start_station(station)

        self._drain_waitlist(station)
        self._deadlock.drain_overflow(station, self._accept_order)

    def _drain_waitlist(self, station: Station) -> None:
        while station.is_available and station.has_waiting:
            source_id = station.pop_waitlist()
            source = self._stations.get(source_id)
            if source is None:
                continue
            freed = self._drain_departure_fifo(source)
            if freed > 0:
                self._on_slot_freed(source)

    def _create_order(self, features: Dict[str, str]) -> Order:
        order_id = f"R{self._sim_config.run_id}_J{self._order_counter:06d}"
        self._order_counter += 1
        return Order(
            id=order_id,
            features=features,
            timestamp_creation=float(self._current_time),
        )

    def _create_initial_orders(self) -> None:
        start = self._find_start_station()
        if start is None:
            raise RuntimeError("No start station configured!")

        cap = start.config.capacity
        capacity_limit = cap if cap > 0 else float("inf")
        fill_count = int(min(capacity_limit, self._sim_config.initial_orders))

        for _ in range(fill_count):
            self._replenish_start_station(start)

    def _find_start_station(self) -> Optional[Station]:
        for station in self._stations.values():
            if station.config.is_start_station:
                return station
        return None

    def _replenish_start_station(self, station: Station) -> None:
        features = self._product.sample_features(current_time=float(self._current_time))
        order = self._create_order(features)
        self._recorder.register_order(order)
        self._accept_order(station, order)
