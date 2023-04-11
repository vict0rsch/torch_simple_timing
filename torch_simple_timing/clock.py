"""
Base class to time Pytorch code.

This clock is the base module reused in :class:`torch_simple_timing.timer.Timer`.

To use stand-alone:

.. code-block:: python
    import torch
    from torch_simple_timing.clock import Clock

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # enable GPU timing or CPU timing
    gpu = device.type == "cuda"

    # data structure to store multiple kinds of timings
    dict_store = {}

    # Create a clock timing the rest of this example.
    # Notice you can chain the instantiation and the start method.
    # (but you don't have to)
    full_clock = Clock(store=dict_store, name="full", gpu=gpu).start()

    # create a random tensor for the sake of this demo
    # and time its creation with an existing store
    with Clock(store=dict_store, name="tensor-init", gpu=gpu):
        t = torch.randn(2000, 2000, device=device)


    # create a one-time clock
    clock_no_store = Clock(gpu=gpu)

    # Create a new clock re-using the same store for its timings.
    clock_dict_store = Clock(store=dict_store, name="mult/inv", gpu=gpu)

    # start clocks
    clock_no_store.start()
    clock_dict_store.start()

    # Create a new clock and using it as a context-manager
    # storing its times in a list
    with Clock(gpu=gpu, store=[]) as clock_list_store:
        torch.inverse(t @ t.T)

    # re-use the same clock as a context-manager
    with clock_list_store:
        torch.inverse(t @ t.T)

    # stop clocks
    clock_no_store.stop()
    clock_dict_store.stop()
    full_clock.stop()

    # print results
    print(clock_no_store.duration)
    print(clock_list_store.store)
    print(full_clock.store)


"""

from time import time
from typing import Dict, List, Optional, Hashable
from torch_simple_timing.utils import synchronize
import torch


class Clock:
    """
    A utility class for timing Pytorch code.

    Args:
        name (str): The name of the timer.
        store (Dict[str, List], optional): A dictionary for storing timer results.
            Defaults to {}.
        gpu (bool, optional): Indicates if GPU timing should be used. Defaults to False.
        ignore (bool, optional): If True, the timer does not record any results.
            Defaults to False.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        store: Optional[Dict[str, List]] = None,
        gpu: Optional[bool] = False,
        ignore: Optional[bool] = False,
    ):
        self.store = store
        self.name = name
        self.gpu = gpu
        self.ignore = ignore

        self.duration = None

        if self.store is not None:
            assert isinstance(
                self.store, (dict, list)
            ), "store must be a dictionary or a list"
            if isinstance(self.store, dict):
                if self.name is not None and not isinstance(self.name, Hashable):
                    raise TypeError(
                        "`name` must be hashable because it is "
                        + "used as a key in the store dict"
                    )
                if self.name not in self.store:
                    self.store[self.name] = []

    def __enter__(self):
        """
        Starts the timer.

        Returns:
            self: The Clock instance.
        """
        return self.start()

    def start(self):
        if self.ignore:
            return
        if self.gpu:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        else:
            self.start_time = time()
        return self

    def __exit__(self, *args):
        """
        Stops the timer and records the duration.

        Args:
            *args: Any exception raised during the timed code.

        Returns:
            None
        """
        self.stop()

    def stop(self):
        if self.ignore:
            return

        if self.gpu:
            self.end_event.record()
            synchronize()
            self.duration = self.start_event.elapsed_time(self.end) / 1000
        else:
            self.end_time = time()
            self.duration = self.end_time - self.start_time

        if self.store is not None:
            if isinstance(self.store, list):
                self.store.append(self.duration)
            elif isinstance(self.store, dict):
                if isinstance(self.store[self.name], list):
                    self.store[self.name].append(self.duration)
                else:
                    self.store[self.name] = self.duration
            else:
                raise TypeError("store must be a dictionary or a list")

    def __repr__(self) -> str:
        """
        Describes the clock from its constructor's arguments.
        If the clock has been used, it also includes its latest duration.

        .. code-block:: python
            # Example outputs
            Clock(store=<NoneType>, name=None, gpu=False, ignore=False)
            Clock(store=<list[2]>, name=None, gpu=False, ignore=False | duration=0.303)
            Clock(store={'full': '<list[1]>', 'tensor-init': '<list[1]>', 'mult/inv': '<list[1]>'}, name=full, gpu=False, ignore=False | duration=0.629)

        Returns:
            str: Clock's description.
        """
        d = {**self.__dict__, "store": f"<{type(self.store).__name__}>"}
        d = {
            k: v if not isinstance(v, float) else float(f"{v:.3f}")
            for k, v in d.items()
            if k in {"name", "store", "gpu", "ignore"}
        }
        if isinstance(self.store, dict):

            d["store"] = {
                k: f"<{type(v).__name__}>"
                if not hasattr(v, "__len__")
                else f"<{type(v).__name__}[{len(v)}]>"
                for k, v in self.store.items()
            }
        elif isinstance(self.store, list):
            d["store"] = f"<{type(self.store).__name__}[{len(self.store)}]>"

        d = ", ".join(f"{k}={v}" for k, v in d.items())
        d += f" | duration={self.duration:.3f}" if self.duration is not None else ""
        d += f" | id={id(self)}"
        return f"{self.__class__.__name__}({d})"