"""
This class enables timing of (PyTorch) code blocks. It internally
leverages the :class:`torch_simple_timing.clock.Clock` class to measure
execution times.

When the constructor argument ``gpu`` is set to ``True``, the timer's clocks
will use ``torch.cuda.Event`` to time GPU code. For timings to be meaningful,
``torch.cuda.synchronize()`` must be called before and after the code block.
This is taken care of by the :class:`torch_simple_timing.clock.Clock` class,
but be aware that this may slow-down your code.

.. note::

    Wait, what?? Timing slows code down?? Yes, it does. But it's not as bad as
    you might think. It mainly means that you should be careful when you define
    ``Clock``and ``Timer`` objects. For example, if you want to time a ``forward``
    function of a model, the fact that the *overall* epoch is slower does not matter,
    you want to accurately measure the time spent in the ``forward`` function.

.. warning::

    Because of the ``torch.cuda.synchronize()`` calls, the ``Timer`` class should
    be carefully used in the context of training. For instance, if you want to time
    **epochs** the synchronization overhead will be negligible. However, if you want
    to time training **iterations**, you should be careful to only do that for 1 (or
    a few epochs) and not for the whole training. Otherwise, the overhead may become
    significant. Use the ``ignore`` argument to disable timing for a specific clock.

Example:

..code-block:: python

    import torch
    from torch.nn import Sequential, Linear, ReLU
    from torch_simple_timing import Timer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = device.type == "cuda"
    timer = Timer(gpu=gpu)

    # manual start
    timer.mark("init").start()

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

    timer.mark("init").stop()

    with timer.mark("train-loop"):
        for epoch in range(epochs):
            with timer.mark("train-epoch"):
                for batch in range(batches):
                    optimizer.zero_grad()
                    # only time the first 2 epochs
                    with timer.mark("train-batch", ignore=epoch > 2):
                        with timer.mark("forward"):
                            pred = model(t[batch * bs : (batch + 1) * bs])
                        with timer.mark("loss", ignore=epoch > 2):
                            loss = loss_func(pred, y[batch * bs : (batch + 1) * bs])
                        with timer.mark("backward", ignore=epoch > 2):
                            loss.backward()
                        optimizer.step()

                    if batch % 10 == 0:
                        print(f"Epoch {epoch}, batch {batch}, loss {loss.item():.3f}")

    # compute mean/std stats for each clock in the timer
    stats = timer.stats()

    # stats will be computed internally if not provided
    print(timer.display(stats=stats, precision=5))

.. code-block:: text


"""

import torch
from torch_simple_timing.clock import Clock


class Timer:
    def __init__(self, gpu=False, ignore=False):
        self.times = {}
        self.clocks = {}
        self.gpu = gpu
        self.ignore = ignore

    def __repr__(self):
        t = {k: len(v) for k, v in self.times.items()}
        r = f"Timer(gpu={self.gpu}, ignore={self.ignore}, times={t})"
        return r

    def reset(self, keys=None):
        """
        Resets timers as per ``keys``.
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

    def mark(self, name, ignore=None, gpu=None):
        if name not in self.clocks:
            if ignore is None:
                ignore = self.ignore
            self.clocks[name] = Clock(
                name, self.times, self.gpu if gpu is None else gpu, ignore
            )
        if ignore != self.clocks[name].ignore:
            self.clocks[name].ignore = ignore
        return self.clocks[name]

    def disable(self, clock_names=None):
        if clock_names is None:
            clock_names = self.clocks.keys()
        for k in clock_names:
            if k in self.clocks:
                self.clocks[k].ignore = True

    def stats(self, clock_names=None):
        if clock_names is None:
            clock_names = self.times.keys()
        clock_names = set(clock_names)
        stats = {}
        for k, v in self.times.items():
            if k in clock_names:
                t = torch.tensor(v)
                m = torch.mean(t).item()
                s = torch.std(t).item()
                n = len(v)
                stats[k] = {"mean": m, "std": s, "n": n}
        return stats

    def display(self, clock_names=None, precision=3, sort_keys_func=None, stats=None):
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
