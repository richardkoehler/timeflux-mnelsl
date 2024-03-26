"""Lab streaming layer nodes

The lab streaming layer provides a set of functions to make instrument data
accessible in real time within a lab network. From there, streams can be
picked up by recording programs, viewing programs or custom experiment
applications that access data streams in real time.

"""

import os
import uuid
from time import time
from typing import Literal

import mne_lsl
import numpy as np
import pandas as pd
from timeflux.core.node import Node


class Send(Node):
    """Send data to an LSL stream.

    Attributes:
        i (Port): Default data input, expects DataFrame.

    Args:
        name (string): The name of the stream.
        type (string): The content type of the stream, e.g. "EEG" or "Markers. See
            https://github.com/sccn/xdf/wiki/Meta-Data for specifications of types of streams.
            Default is the custom type "Signal".
        format (string): The type that the data will be converted to. Supported types: ``float32``,
            ``float64``, ``int8``, ``int16``, ``int32``, ``int64``, ``string``. Default is
            ``float64``.
        rate (float): The sampling rate. Set to ``0.0`` to indicate a variable sampling rate.
            Default is 0.0.
        source (string, None): The unique identifier for the stream. If ``None``, it will be
            auto-generated. Default is ``None``.
        config_path (string, None): Optional path to an LSL config file. Default is ``None``.
    """

    def __init__(
        self,
        name: str,
        type: str = "Signal",
        format: Literal[
            "string", "float32", "float64", "int8", "int16", "int32", "int64"
        ] = "float64",
        rate: float = 0.0,
        source: str | None = None,
        config_path: str | None = None,
    ) -> None:
        self._name = name
        self._type = type
        self._include: list = [object if format != "string" else np.number]
        self._format = format
        self._rate = rate
        if not source:
            source = str(uuid.uuid4())
        self._source = source
        self._outlet: mne_lsl.lsl.StreamOutlet | None = None
        if config_path is not None:
            os.environ["LSLAPICFG"] = config_path

    def update(self) -> None:
        if isinstance(self.i.data, pd.core.frame.DataFrame):
            data: pd.DataFrame = self.i.data.select_dtypes(include=self._include)
            if self._outlet is None:
                labels: list[str] = data.columns.tolist()
                sinfo = mne_lsl.lsl.StreamInfo(
                    name=self._name,
                    stype=self._type,
                    n_channels=len(labels),
                    sfreq=self._rate,
                    dtype=self._format,
                    source_id=self._source,
                )
                sinfo.set_channel_names(labels)
                self._outlet = mne_lsl.lsl.StreamOutlet(sinfo)
            values = data.to_numpy(dtype=self._format)
            stamps = self.i.data.index.values.astype(np.float64) / 1e9
            for row, stamp in zip(values, stamps, strict=True):
                self._outlet.push_sample(row, stamp)


class Receive(Node):
    """Receive data from an LSL stream.

    Attributes:
        o (Port): Default output, provides DataFrame and meta.

    Args:
        prop (string): The property to look for during stream resolution. One of ``name``, ``type``,
            ``stype`` (type and stype are equal) or ``source_id``. Preferably use ``source_id``.
        value (string): The value that the property should have (e.g., ``EEG`` for the ``type``
            property).
        timeout (float): The resolution timeout, in seconds.
        channels (list, None): Override the channel names. If ``None``, the names defined in the LSL
            stream will be used.
        max_samples (int): The maximum number of samples to return per call. Default is 1024.
        clocksync (bool): Perform automatic clock synchronization if the stream has a LSL timestamp
            and not a Timeflux timestamp. Default is ``True``.
        dejitter (bool): Remove jitter from timestamps using a smoothing algorithm to the received
            timestamps. Default is ``False``.
        monotonize (bool): Force the timestamps to be monotonically ascending. Only makes sense if
            timestamps are dejittered. Default is ``False``.
        threadsafe (bool): Same inlet can be read from by multiple threads. Default is ``True``.
        config_path (string, None): Optional path to an LSL config file. Default is ``None``.

    """

    def __init__(
        self,
        prop: Literal["name", "type", "stype", "source_id"] = "name",
        value: str | None = None,
        timeout: float = 1.0,
        channels: list[str] | None = None,
        max_samples: int = 1024,
        clocksync: bool = True,
        dejitter: bool = False,
        monotonize: bool = False,
        threadsafe: bool = True,
        config_path: str | None = None,
    ) -> None:
        if not value:
            raise ValueError("Please specify a stream name or a property and value.")
        self._prop = prop
        if self._prop == "type":
            self._prop = "stype"  # MNE-LSL takes the argument stype not type
        self._value = value
        self._inlet: mne_lsl.lsl.StreamInlet | None = None
        self._labels: list[str] | None = None
        self._channels = channels
        self._timeout = timeout
        self._max_samples = max_samples
        self._flags: list[str] | None = []
        self._offset = time() - mne_lsl.lsl.local_clock() if clocksync else 0
        for flag_descr, flag in zip(
            ("clocksync", "dejitter", "monotonize", "threadsafe"),
            (clocksync, dejitter, monotonize, threadsafe),
            strict=True,
        ):
            if flag:
                self._flags.append(flag_descr)
        if not self._flags:
            self._flags = None
        if config_path is not None:
            os.environ["LSLAPICFG"] = config_path

    def update(self) -> None:
        if self._inlet is None:
            self.logger.debug(f"Resolving stream with {self._prop} {self._value}")

            streams = mne_lsl.lsl.resolve_streams(
                timeout=self._timeout, **{self._prop: self._value}
            )

            if not streams:
                return
            self.logger.debug("Stream acquired")
            self._inlet = mne_lsl.lsl.StreamInlet(sinfo=streams[0], processing_flags=self._flags)
            self._inlet.open_stream()
            sinfo = self._inlet.get_sinfo()  # retrieve stream information with all properties
            self._meta = {
                "name": sinfo.name,
                "type": sinfo.stype,
                "rate": sinfo.sfreq,
                "info": sinfo.as_xml.replace("\n", "").replace("\t", ""),
            }
            if isinstance(self._channels, list):
                self._labels = self._channels
            else:
                self._labels = sinfo.get_channel_names()

        values, stamps = self._inlet.pull_chunk(max_samples=self._max_samples)
        if stamps.size > 0:
            if self._offset != 0:
                stamps = stamps + self._offset
            stamps = pd.to_datetime(stamps, format=None, unit="s")
            self.o.set(values, stamps, self._labels, self._meta)
