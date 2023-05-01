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

"""
from .clock import Clock  # noqa: F401
from .timer import Timer


TIMER = Timer()


def timeit(name, gpu=False, timer=None):
    """
    Decorator to time a function call.
    Example:

    .. code-block:: python

        from torch_simple_timing import timeit, get_global_timer, reset_global_timer
        import torch

        @timeit("train", gpu=True)
        def train():
            x = torch.rand(1000, 1000, device="cuda" if torch.cuda.is_available() else "cpu")
            return torch.inverse(x @ x)

        @timeit("test")
        def test_cpu():
            return torch.inverse(torch.rand(1000, 1000) @ torch.rand(1000, 1000))

        train()
        test()

        timer = get_global_timer()
        stats = timer.stats()
        print(timer.display(stats=stats))

        reset_global_timer()


    """
    if timer is None:
        global TIMER
        timer = TIMER

    def decorator(func):
        def wrapper(*args, **kwargs):
            with timer.clock(name, gpu=gpu):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def set_global_timer(timer):
    global TIMER
    TIMER = timer


def get_global_timer():
    return TIMER


def reset_global_timer():
    global TIMER
    TIMER = Timer()
