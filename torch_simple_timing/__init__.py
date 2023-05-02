"""
1. "*I want to time operations once*"

   * That's what a :class:`~torch_simple_timing.clock.Clock` is for

2. "*I want to time the same operations multiple times*"

   * That's what a :class:`~torch_simple_timing.timer.Timer` is for

In simple terms:

* A :class:`~torch_simple_timing.clock.Clock` is an object (and context-manager) that
    will compute the elapsed
    time between its :meth:`~torch_simple_timing.clock.Clock.start()`
    (or :meth:`~torch_simple_timing.clock.Clock.__enter__`) and
    :meth:`~torch_simple_timing.clock.Clock.stop()`
    (or :meth:`~torch_simple_timing.clock.Clock.__exit__`)
* A :class:`~torch_simple_timing.timer.Timer` will internally manage clocks so that you
    can focus on readability and not data structures

You can also decorate your functions with :func:`~torch_simple_timing.timeit` to time
their execution with little code overhead. Then use
:func:`~torch_simple_timing.get_global_timer` to access the timer it uses and access
its :meth:`~torch_simple_timing.timer.Timer.stats` or use it manually to time code.

.. code-block:: python
    :caption: Example of using :func:`~torch_simple_timing.timeit` to time a function
              (pseudo-code)

    from torch_simple_timing import timeit, get_global_timer

    @timeit(gpu=True)
    def train(*args, **kwargs):
        # do stuff

    @timeit(gpu=True)
    def test(*args, **kwargs):
        # some other stuff

    def main():
        for _ in range(epochs):
            train(epochs, model, optimizer, loss_fn)

        restults = test(model, loss_fn)

        timer = get_global_timer()

        with timer.clock("logging"):
            logger.log(results)

        logger.log(timer.stats())
"""
from .clock import Clock  # noqa: F401
from .timer import Timer
from pathlib import Path

__version__ = (
    [
        line.split("=")[-1].strip()
        for line in (Path(__file__).resolve().parent.parent / "pyproject.toml")
        .read_text()
        .splitlines()
        if line.startswith("version")
    ][0]
    .replace("'", "")
    .replace('"', "")
)
"""The package version string."""

TIMER = Timer()
"""
The global :class:`~torch_simple_timing.timer.Timer` instance
used by :func:`~torch_simple_timing.timeit` to time functions.
"""


def timeit(name=None, gpu=False, timer=None):
    """
    Decorator to time a function call.
    Example:

    .. code-block:: python

        from torch_simple_timing import timeit, get_global_timer, reset_global_timer
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpu = device.type == "cuda"

        # Use the function name as the timer name
        @timeit(gpu=gpu)
        def train():
            x = torch.rand(1000, 1000, device=)
            return torch.inverse(x @ x)

        # Use a custom name
        @timeit("test")
        def test_cpu():
            return torch.inverse(torch.rand(1000, 1000) @ torch.rand(1000, 1000))

        if __name__ == "__main__":
            for _ in range((epochs := 10)):
                train()
            test_cpu()

            timer = get_global_timer()
            stats = timer.stats()
            print(timer.display(stats=stats))

            reset_global_timer()

    Args:
        name (str, optional): The name of the timer to use. Defaults to the decorated
            function's name.
        gpu (bool, optional): Whether to use GPU timing. Defaults to False.
        timer (:class:`~torch_simple_timing.timer.Timer`, optional): The timer instance
            to use. Defaults to ``None``, in which case the global timer instance is
            used.

    """
    if timer is None:
        global TIMER
        timer = TIMER

    def decorator(func):
        def wrapper(*args, **kwargs):
            with timer.clock(name if name is not None else func.__name__, gpu=gpu):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def set_global_timer(timer) -> None:
    """
    Set the global :class:`~torch_simple_timing.timer.Timer` instance
    to a user-provided new one.

    Args:
        timer (:class:`~torch_simple_timing.timer.Timer`): The new timer instance
            to use globally.
    """
    global TIMER
    TIMER = timer


def get_global_timer() -> Timer:
    """
    Get the global :class:`~torch_simple_timing.timer.Timer` instance.

    Returns:
        :class:`~torch_simple_timing.timer.Timer`: The global timer instance.
    """
    return TIMER


def reset_global_timer() -> None:
    """
    Sets the global :class:`~torch_simple_timing.timer.Timer` instance to a new one.
    """
    global TIMER
    TIMER = Timer()
