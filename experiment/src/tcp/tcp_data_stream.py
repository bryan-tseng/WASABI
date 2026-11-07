"""Stream wideband data from Intan RHX over TCP into a ring buffer.

RHX must have Remote TCP Control enabled (Network > Remote TCP Control),
with the command server (default 5010) and waveform output server (default 5001) connected.
RHX keeps recording the full .rhs to disk as usual; this only taps the live data.
"""
import socket
import threading
import time

import numpy as np

MAGIC = 0x2EF07A08
FRAMES_PER_BLOCK = 128
UV_PER_BIT = 0.195


class RingBuffer:
    """Fixed-size (n_samples, n_ch) float32 buffer indexed by absolute sample count."""

    def __init__(self, n_samples, n_ch):
        self.buf = np.zeros((n_samples, n_ch), dtype=np.float32)
        self.n = n_samples
        self.total = 0  # samples written since start
        self.t0 = None  # RHX timestamp of the first sample received
        self.lock = threading.Lock()

    def write(self, x):
        k = x.shape[0]
        i = self.total % self.n
        with self.lock:
            if i + k <= self.n:
                self.buf[i:i + k] = x
            else:
                m = self.n - i
                self.buf[i:] = x[:m]
                self.buf[:k - m] = x[m:]
            self.total += k

    def read(self, start, stop):
        """Copy samples [start, stop) in absolute sample index. Returns None if not yet available or overwritten."""
        with self.lock:
            if stop > self.total or start < self.total - self.n:
                return None
            idx = np.arange(start, stop) % self.n
            return self.buf[idx].copy()


class RHXStream:
    def __init__(self, amp_channels, digital_in=False, host="127.0.0.1",
                 cmd_port=5010, data_port=5001, buffer_s=10.0):
        # amp_channels: RHX names, e.g. ["A-000", "A-001", ...]
        self.amp_channels = list(amp_channels)
        self.digital_in = digital_in
        self.cmd = socket.create_connection((host, cmd_port))
        self.data = socket.create_connection((host, data_port))
        self.data.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 23)
        self.data.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        self.fs = float(self.query("get sampleratehertz"))
        self.n_amp = len(self.amp_channels)
        n_words = self.n_amp + (1 if digital_in else 0)
        # One TCP data block: magic, then 128 frames of [int32 timestamp, uint16 x n_words]
        frame = np.dtype([("ts", "<i4"), ("w", "<u2", (n_words,))])
        self.block_dtype = np.dtype([("magic", "<u4"), ("frames", frame, (FRAMES_PER_BLOCK,))])

        self.ring = RingBuffer(int(buffer_s * self.fs), self.n_amp)
        self.dig = np.zeros(self.ring.n, dtype=np.uint16) if digital_in else None
        self.dropped = 0
        self._stop = threading.Event()
        self._thread = None

    # Command port helpers
    def send(self, msg):
        self.cmd.sendall(msg.encode())
        time.sleep(0.02)  # RHX processes commands sequentially; small gap avoids merged commands

    def query(self, msg):
        self.cmd.sendall(msg.encode())
        resp = self.cmd.recv(1024).decode().strip()
        return resp.split()[-1]  # e.g. "Return: SampleRateHertz 30000"

    def configure(self, blocks_per_write=1):
        self.send("set runmode stop")
        self.send("execute clearalldataoutputs")
        self.send(f"set TCPNumberDataBlocksPerWrite {blocks_per_write}")  # 1 block = 128 samples = 4.3 ms at 30 kHz
        for ch in self.amp_channels:
            self.send(f"set {ch}.tcpdataoutputenabled true")
        if self.digital_in:
            self.send("set digital-in-01.tcpdataoutputenabled true")  # verify name/order against your RHX version

    def start(self, record=True):
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        self.send("set runmode record" if record else "set runmode run")

    def stop(self):
        self.send("set runmode stop")
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    # Reader thread
    def _reader(self):
        bsize = self.block_dtype.itemsize
        raw = bytearray(bsize * 512)
        view = memoryview(raw)
        have = 0
        last_ts = None
        while not self._stop.is_set():
            n = self.data.recv_into(view[have:])
            if n == 0:
                break
            have += n
            n_blocks = have // bsize
            if n_blocks == 0:
                continue
            blocks = np.frombuffer(raw, dtype=self.block_dtype, count=n_blocks)
            if not np.all(blocks["magic"] == MAGIC):
                raise RuntimeError("Lost alignment with RHX data stream (bad magic number)")

            ts = blocks["frames"]["ts"].reshape(-1)
            w = blocks["frames"]["w"].reshape(-1, blocks["frames"]["w"].shape[-1])
            amp = (w[:, :self.n_amp].astype(np.float32) - 32768.0) * UV_PER_BIT

            if last_ts is None:
                self.ring.t0 = int(ts[0])
            elif ts[0] != last_ts + 1:
                self.dropped += int(ts[0] - last_ts - 1)
            last_ts = int(ts[-1])

            if self.digital_in:
                i = self.ring.total % self.ring.n
                k = w.shape[0]
                idx = (np.arange(k) + i) % self.ring.n
                self.dig[idx] = w[:, -1]
            self.ring.write(amp)

            used = n_blocks * bsize
            raw[:have - used] = raw[used:have]
            have -= used

    # Convenience
    def latest(self, n_samples):
        t = self.ring.total
        return self.ring.read(t - n_samples, t), t

    def window(self, center_sample, pre_s, post_s, timeout_s=1.0):
        """Block until [center - pre, center + post) is available, then return it as (n_samples, n_ch)."""
        a = center_sample - int(pre_s * self.fs)
        b = center_sample + int(post_s * self.fs)
        t_end = time.perf_counter() + timeout_s
        while self.ring.total < b:
            if time.perf_counter() > t_end:
                return None
            time.sleep(0.001)
        return self.ring.read(a, b)


if __name__ == "__main__":
    # chans = [f"a-{i:03d}" for i in range(64)] + [f"b-{i:03d}" for i in range(16)]
    chans = [f"A-{i:03d}" for i in range(32)]
    rhx = RHXStream(chans)
    rhx.configure(blocks_per_write=1)
    rhx.start(record=False)
    try:
        while True:
            time.sleep(1.0)
            x, t = rhx.latest(int(0.1 * rhx.fs))
            if x is not None:
                print(f"t={t / rhx.fs:7.2f} s  rms={x.std(0).mean():6.1f} uV  dropped={rhx.dropped}")
    except KeyboardInterrupt:
        rhx.stop()