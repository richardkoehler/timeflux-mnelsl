"""Tests for lsl.py"""

import time
import uuid

import numpy as np
import pandas as pd
import timeflux.helpers.clock
import timeflux_mnelsl
from timeflux.core.io import Port


def test_lsl() -> None:
    source_id = uuid.uuid4().hex
    sender = timeflux_mnelsl.nodes.lsl.Send(
        name="test", type="Signal", format="float64", rate=0.0, source=source_id
    )
    sender.i = Port()
    receiver = timeflux_mnelsl.nodes.lsl.Receive(prop="source_id", value=source_id, clocksync=False)
    receiver.o = Port()
    for ind in range(2):  # The first sample is lost, as the receiver is not yet connected to LSL
        now = timeflux.helpers.clock.now()
        expected = pd.DataFrame(data=[[ind]], index=(now,), columns=["A"], dtype=float)
        sender.i.data = expected.copy()
        sender.update()
        time.sleep(0.01)
        receiver.update()
    print(receiver.o.data)
    if receiver._inlet is not None:
        receiver._inlet.close_stream()
        del receiver._inlet
    expected = expected
    output = receiver.o.data
    print(output.index.to_numpy(dtype=float))
    print(expected.index.to_numpy(dtype=float))
    np.testing.assert_array_almost_equal(
        output.index.to_numpy(dtype=float), expected.index.to_numpy(dtype=float), decimal=6
    )
    np.testing.assert_array_equal(output.to_numpy(), expected.to_numpy())


if __name__ == "__main__":
    test_lsl()
