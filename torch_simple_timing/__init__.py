"""
1. "*I want to time operations once*"

   * That's what a :class:`~torch_simple_timing.clock.Clock` is for

2. "*I want to time the same operations multiple times*"

   * That's what a :class:`~torch_simple_timing.timer.Timer` is for

In simple terms:

* A :class:`~torch_simple_timing.clock.Clock` is an object (and context-manager) that will compute the elapsed
    time between its :meth:`~torch_simple_timing.clock.Clock.start()` (or :meth:`~torch_simple_timing.clock.Clock.__enter__`) and :meth:`~torch_simple_timing.clock.Clock.stop()` (or :meth:`~torch_simple_timing.clock.Clock.__exit__`)
* A :class:`~torch_simple_timing.timer.Timer` will internally manage clocks so that you can focus on readability
    and not data structures

"""
from .clock import Clock  # noqa: F401
from .timer import Timer  # noqa: F401
