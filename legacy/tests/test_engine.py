import numpy as np

from openmixing.engine.ringbuffer import RingBuffer


def test_ring_buffer_wraps_and_zero_fills():
    rb = RingBuffer(10)
    x = np.arange(16, dtype=np.float32).reshape(8, 2)
    assert rb.write(x) == 8
    out = np.empty((6, 2), np.float32)
    assert rb.read_into(out) == 6 and (out == x[:6]).all()
    assert rb.write(x) == 8  # wraps around the end
    out = np.empty((12, 2), np.float32)
    assert rb.read_into(out) == 10
    assert (out[:2] == x[6:]).all() and (out[2:10] == x).all() and (out[10:] == 0).all()
