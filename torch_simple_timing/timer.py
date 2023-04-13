"""
This class enables timing of (PyTorch) code blocks. It internally
leverages the :class:`~torch_simple_timing.clock.Clock` class to measure
execution times.

When the constructor argument ``gpu`` is set to ``True``, the timer's clocks
will use :class:`torch.cuda.Event` to time GPU code. For timings to be meaningful,
:func:`torch.cuda.synchronize()` must be called before and after the code block.
In the case of distributed training, :func:`torch.distributed.barrier()` will also
be called.

This ^ is taken care of by the :class:`~torch_simple_timing.clock.Clock` class,
but be aware that this may slow-down your code.

.. note::

    Wait, what?? Timing slows code down?? Yes, it does. But it's not as bad as
    you might think. It mainly means that you should be careful when you define
    :class:`~torch_simple_timing.clock.Clock` and
    :class:`~torch_simple_timing.timer.Timer`
    objects. For example, if you want to time a ``forward``
    function of a model, the fact that the *overall* epoch is slower does not matter,
    you want to accurately measure the time spent in the ``forward`` function.

.. warning::

    Because of the :func:`torch.cuda.synchronize()` calls, the
    :class:`~torch_simple_timing.timer.Timer` class should
    be carefully used in the context of training. For instance, if you want to time
    **epochs** the synchronization overhead will be negligible. However, if you want
    to time training **iterations**, you should be careful to only do that for 1 (or
    a few epochs) and not for the whole training. Otherwise, the overhead may become
    significant. Use the ``ignore`` argument to disable timing for a specific clock.

Example:

.. code-block:: python

    import torch
    from torch.nn import Sequential, Linear, ReLU
    from torch_simple_timing import Timer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = device.type == "cuda"
    timer = Timer(gpu=gpu)

    # manual start
    timer.clock("init").start()

    batches = 32
    bs = 64
    n = batches * bs
    dim = 64
    labels = 10
    hidden = 1024
    epochs = 5

    t = torch.randn(n, dim, device=device)
    y = torch.randint(0, labels, (n,), device=device)

    model = Sequential(
        Linear(dim, hidden),
        ReLU(),
        Linear(hidden, hidden),
        ReLU(),
        Linear(hidden, labels),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    loss_func = torch.nn.CrossEntropyLoss()

    timer.clock("init").stop()

    with timer.clock("train-loop"):
        for epoch in range(epochs):
            with timer.clock("train-epoch"):
                for batch in range(batches):
                    optimizer.zero_grad()
                    # only time the first 2 epochs
                    with timer.clock("train-batch", ignore=epoch > 2):
                        with timer.clock("forward"):
                            pred = model(t[batch * bs : (batch + 1) * bs])
                        with timer.clock("loss", ignore=epoch > 2):
                            loss = loss_func(pred, y[batch * bs : (batch + 1) * bs])
                        with timer.clock("backward", ignore=epoch > 2):
                            loss.backward()
                        optimizer.step()

                    if batch % 10 == 0:
                        print(f"Epoch {epoch}, batch {batch}, loss {loss.item():.3f}")

    # compute mean/std stats for each clock in the timer
    stats = timer.stats()

    # stats will be computed internally if not provided
    print(timer.display(stats=stats, precision=5))

.. code-block:: text

    init        : 0.01141           (n=  1)
    train-loop  : 1.97650           (n=  1)
    train-epoch : 0.39529 ± 0.07439 (n=  5)
    train-batch : 0.01087 ± 0.00189 (n= 96)
    forward     : 0.00327 ± 0.00703 (n=160)
    loss        : 0.00010 ± 0.00023 (n= 96)
    backward    : 0.00438 ± 0.00074 (n= 96)

"""

import torch
from torch_simple_timing.clock import Clock
from typing import Dict, List, Optional, Union, Callable


class Timer:
    def __init__(self, gpu: bool = False, ignore: bool = False):
        """
        ``Clock`` manager. Store and display timing statistics.

        .. warning::

            In order to accurately measure GPU timings, :func:`torch.cuda.synchronize()`
            will be called before and after each clock's ``start`` and ``stop``

        Args:
            gpu (bool, optional): Whether or not to use GPU timing
                using CUDA events. Defaults to ``False``.
            ignore (bool, optional): Whether to disable this timer. Can be useful
                when the same piece of code is used in various contexts, for instance
                in training or validation modes you may want to disable timing.
                Defaults to ``False``.
        """
        self.times = {}
        self.clocks = {}
        self.gpu = gpu
        self.ignore = ignore

    def __repr__(self) -> str:
        t = {k: len(v) for k, v in self.times.items()}
        r = f"Timer(gpu={self.gpu}, ignore={self.ignore}, times={t})"
        return r

    def reset(self, keys: Optional[Union[str, List[str]]] = None) -> None:
        """
        Deletes specified ``keys``.
        If ``keys`` is None, resets all timers.

        Args:
            keys (Union[str, List[str]], optional): Specific named timers to reset,
                or all of them if ``keys`` is ``None`` . Defaults to ``None``.
        """
        if isinstance(keys, str):
            keys = [keys]

        if keys is None:
            self.times = {}
            self.clocks = {}
        else:
            for k in keys:
                self.times.pop(k, None)
                self.clocks.pop(k, None)

    def clock(
        self,
        name: str,
        ignore: Optional[bool] = None,
        gpu: Optional[bool] = None,
    ) -> Clock:
        """
        Create a new ``Clock`` object with name ``name`` and add it to the ``Timer``.
        If the ``Clock`` already exists, it will be returned.

        .. note::

            If ``ignore`` is ``None``, the ``Timer``'s ``ignore`` attribute will be
            used.

        .. note::

            If ``ignore`` is not ``None``, the ``Clock`` 's ``ignore`` attribute will be
            updated.

        .. warning::

            Don't forget to call ``.start()`` and ``.stop()`` on the returned ``Clock``
            if you're not using ``timer.clock()`` as a context manager.

        Args:
            name (str): A name for the requested clock.
            ignore (Optional[bool], optional): Whether to ignore this clock and don't
                time anything. This is useful in case timing slows you down (because of
                :func:`torch.cuda.synchronize()` and :func:`torch.distributed.barrier()`) and
                you only want to time the first epoch for instance. Defaults to
                ``None``.
            gpu (Optional[bool], optional): Whether to enable GPU timing with CUDA
                events. Defaults to ``None``.

        Returns:
            Clock: The requested ``Clock`` object.
        """
        if name not in self.clocks:
            if ignore is None:
                ignore = self.ignore
            self.clocks[name] = Clock(
                name, self.times, self.gpu if gpu is None else gpu, ignore
            )
        if ignore != self.clocks[name].ignore:
            self.clocks[name].ignore = ignore
        return self.clocks[name]

    def disable(self, clock_names: Optional[List[str]] = None) -> None:
        """
        Disable the specified clocks based on their names.

        Args:
            clock_names (Optional[List[str]], optional): The list of clock names
                to disable. If ``None``, all clocks in this timer are disabled.
                Defaults to ``None``.
        """
        if clock_names is None:
            clock_names = self.clocks.keys()
        for k in clock_names:
            if k in self.clocks:
                self.clocks[k].ignore = True

    def stats(
        self,
        clock_names: Optional[List[str]] = None,
        map_funcs: Optional[Dict[str, callable]] = None,
    ) -> Dict[str, Dict[str, Union[int, float]]]:
        """
        Computes the mean and standard deviation of the times for each clock.
        Returns a dictionary of dictionaries with the following structure:

        .. code-block:: python

            {
                "clock_name": {
                    "mean": float,
                    "std": float,
                    "n": int
                }
            }

        Optionally, you can provide a dictionary of functions to apply to the
        list of times for each clock. If a clock name is not in the dictionary,
        no function will be applied (equivalent to ``lambda t: t``).

        .. code-block:: python

                throughput = timer.stats(
                    map_funcs={"forward": lambda t: batch_size / t}
                )

        This method will be called internally by ``timer.display()``
        or you can provide it there if you want to do something else with the
        stats (log them for instance).

        Args:
            clock_names (List[str], optional): List of clock names to compute
                the stats for, or all of them if ``None`` . Defaults to ``None``.
            map_funcs (Dict[str, callable], optional): Dictionary of functions
                to pre-process the list of times for each clock. Defaults to ``None``.

        Returns:
            Dict[str, Dict[str, Union[int, float]]]: A dictionary of dictionaries,
                mapping clock names to a dictionary of statistics.
        """
        if clock_names is None:
            clock_names = self.times.keys()
        if map_funcs is None:
            map_funcs = {}
        clock_names = set(clock_names)
        stats = {}
        for k, v in self.times.items():
            if k in clock_names:
                t = torch.tensor(list(map(map_funcs.get(k, lambda t: t), v))).float()
                m = torch.mean(t).item()
                s = torch.std(t).item()
                n = len(v)
                stats[k] = {"mean": m, "std": s, "n": n}
        return stats

    def display(
        self,
        clock_names: Optional[List[str]] = None,
        precision: int = 3,
        sort_keys_func: Callable = None,
        stats: Dict[str, Dict[str, Union[int, float]]] = None,
    ):
        """
        Display the mean, standard deviation and support of the times
        for each clock.

        :meth:`Timer.stats` is called internally to compute the stats. You can
        pre-compute stats independently and pass them to this method with the
        ``stats=`` argument.

        Optionally, you can provide a function ``sort_keys_func`` to sort the
        clocks by a specific key. For instance, you can sort them alphabetically
        with ``sort_keys_func=lambda k: k``. By default, they will be displayed
        according to their creation order.

        .. code-block:: python

            >>> print(timer.display())
            epoch    : 0.251 ± 0.027 (n=10)
            forward  : 0.002 ± 0.005 (n=50)
            backward : 0.002 ± 0.004 (n=50)

        Args:
            clock_names (Optional[List[str]], optional): The list of clock names
                to display. If ``None``, all clocks in this timer are displayed.
                Defaults to ``None``.
            precision (int, optional): The number of digits to display after the
                decimal point. Defaults to ``3``.
            sort_keys_func (Callable, optional): A function to use to sort the
                displayed clocks. Defaults to ``None``, *i.e.* creation order.
            stats (Dict[str, Dict[str, Union[int, float]]], optional): The stats
                to display. If ``None``, the stats will be computed internally.
                Defaults to ``None``.

        Returns:
            str: A string representation of the stats.
        """

        if stats is None:
            stats = self.stats(clock_names)
        if sort_keys_func is not None:
            keys = sorted(stats.keys(), key=sort_keys_func)
        else:
            keys = list(stats.keys())
        max_key_len = max(len(k) for k in keys)

        mean_strs = [f"{stats[k]['mean']:.{precision}f}" for k in keys]
        std_strs = [f"{stats[k]['std']:.{precision}f}" for k in keys]
        n_strs = [f"{stats[k]['n']}" for k in keys]

        max_std_len = max(len(s) for s in std_strs)
        max_mean_len = max(len(s) for s in mean_strs)
        max_n_len = max(len(s) for s in n_strs)

        outs = []

        for i, k in enumerate(keys):
            v = stats[k]
            mean = mean_strs[i]
            mean_s = f"{mean:>{max_mean_len}}"
            n = n_strs[i]
            n_str = f"(n={n:>{max_n_len}})"
            std = std_strs[i]
            std_s = (
                f" ± {std:>{max_std_len}}" if v["n"] > 1 else " " * (max_std_len + 3)
            )
            outs.append(f"{k:<{max_key_len+1}}: {mean_s}{std_s} {n_str}")
        return "\n".join(outs)
