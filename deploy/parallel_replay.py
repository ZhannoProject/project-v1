"""Persistent channel workers: one single-thread ORT session per CPU core.

The coordinator releases each acquired hop to all workers, then waits for
every channel before timestamping output availability. IPC and synchronization
are included in hop latency. Each channel owns independent filter/model state.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
import traceback

import numpy as np


class ChannelProcessor:
    def __init__(self, model, channels, sigma, raw_input, reset_samples=()):
        from deploy.preprocessing import CausalFilter
        self.model, self.sigma = model, sigma
        self.raw_input = raw_input
        self.resets = set(reset_samples)
        self.states = [model.initial_state() for _ in range(channels)]
        self.filters = [CausalFilter(model.sample_rate) if raw_input else None for _ in range(channels)]
        self.segment_start = 0

    def hop(self, signal, first, end):
        from deploy.preprocessing import CausalFilter
        cuts = [first, *[p for p in range(first+1, end) if p in self.resets], end]
        output = np.empty((len(self.states), end-first), dtype=np.float32)
        infer_s = 0.
        for channel in range(len(self.states)):
            for lo, hi in zip(cuts, cuts[1:]):
                if lo in self.resets:
                    self.states[channel] = self.model.initial_state()
                    self.filters[channel] = CausalFilter(self.model.sample_rate) if self.raw_input else None
                    self.segment_start = lo
                chunk = signal[lo:hi]
                if self.filters[channel] is not None:
                    chunk = self.filters[channel].process(chunk)
                for index, sample in zip(range(lo, hi), chunk):
                    value, self.states[channel] = self.model.step(float(sample)/self.sigma, self.states[channel])
                    output[channel, index-first] = value*self.sigma
                    infer_s += self.model.last_run_s
        return output, infer_s, self.segment_start


def _worker(connection, model_path, input_path, shared, shape, channel_start, channel_stop,
            cpu, sigma, raw_input, resets):
    try:
        from deploy import preflight
        from deploy.memory import peak_ram_mib
        from deploy.raspi_infer import IncrementalDenoiser, normalization_scale
        if cpu is not None:
            preflight.pin(str(cpu))
        model = IncrementalDenoiser(model_path, 1, spin=False)
        sigma = normalization_scale(model.meta, sigma=sigma, online=True)
        signal = np.load(input_path, allow_pickle=False, mmap_mode='r')
        output = np.frombuffer(shared, dtype=np.float32).reshape(shape)
        processor = ChannelProcessor(model, channel_stop-channel_start, sigma, raw_input, resets)
        ready = dict(pid=os.getpid(), cpu=cpu, channels=list(range(channel_start,channel_stop)),
                     threads=model.threads, affinity=sorted(os.sched_getaffinity(0))
                     if hasattr(os,'sched_getaffinity') else None)
        connection.send(('ready', ready))
        cpu_start = time.process_time()
        while True:
            command = connection.recv()
            if command is None:
                connection.send(('finished', dict(**ready, cpu_time_s=time.process_time()-cpu_start,
                                                  peak_ram_mib=peak_ram_mib())))
                break
            first, end = command
            begin = time.perf_counter()
            values, infer_s, segment = processor.hop(signal, first, end)
            output[channel_start:channel_stop, first:end] = values
            connection.send(('hop', dict(infer_s=infer_s, segment_start=segment,
                                         processing_ms=1000*(time.perf_counter()-begin))))
    except (EOFError, BrokenPipeError):
        pass  # Coordinator stopped; do not leave an orphan inference process.
    except BaseException:
        try:
            connection.send(('error', traceback.format_exc()))
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        connection.close()


class ParallelChannels:
    def __init__(self, model_path, input_path, n_samples, channels, workers, cpus,
                 sigma, raw_input, reset_samples):
        self.context = mp.get_context('spawn')
        self.shape = (channels, n_samples)
        self.shared = self.context.RawArray('f', channels*n_samples)
        self.output = np.frombuffer(self.shared, dtype=np.float32).reshape(self.shape)
        self.processes, self.connections, self.ready, self.finished = [], [], [], []
        self.closed = False
        count = min(workers, channels)
        offset = 0
        try:
            for i in range(count):
                size = channels//count + (i < channels % count)
                parent, child = self.context.Pipe()
                process = self.context.Process(target=_worker,
                    args=(child, str(model_path), str(input_path), self.shared, self.shape,
                          offset, offset+size, cpus[i] if cpus else None, sigma, raw_input, reset_samples),
                    name=f'EEG-channel-worker-{i+1}', daemon=True)
                process.start()
                child.close()
                self.processes.append(process)
                self.connections.append(parent)
                offset += size
            self.ready = [self._receive(i, 'ready', timeout=90) for i in range(count)]
        except BaseException:
            self.close()
            raise

    def _receive(self, index, expected, timeout=30):
        connection, process = self.connections[index], self.processes[index]
        deadline = time.monotonic()+timeout
        while not connection.poll(.1):
            if not process.is_alive():
                raise RuntimeError(f'Channel worker {index+1} exited ({process.exitcode})')
            if time.monotonic() >= deadline:
                raise RuntimeError(f'Channel worker {index+1} timed out')
        try:
            kind, payload = connection.recv()
        except EOFError as exc:
            raise RuntimeError(f'Channel worker {index+1} disconnected') from exc
        if kind != expected:
            raise RuntimeError(f'Channel worker {index+1}: {payload}')
        return payload

    @property
    def runtime(self):
        return dict(mode='channel_processes', process_count=len(self.ready), threads_per_process=1,
                    allow_spinning=False, worker_settings=self.ready)

    def hop(self, first, end):
        for connection in self.connections:
            connection.send((first,end))
        return [self._receive(i,'hop') for i in range(len(self.connections))]

    def finish(self):
        for connection in self.connections:
            connection.send(None)
        self.finished = [self._receive(i,'finished') for i in range(len(self.connections))]
        self.close(terminate=False)

    def close(self, terminate=True):
        if self.closed:
            return
        self.closed = True
        for connection in self.connections:
            connection.close()
        for process in self.processes:
            if terminate and process.is_alive():
                process.terminate()
            process.join(timeout=3)
            if process.is_alive():
                process.kill()
                process.join(timeout=3)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
