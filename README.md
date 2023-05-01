<p align="center">
<strong><a href="https://github.com/vict0rsch/torch_simple_timing" target="_blank">💻&nbsp;&nbsp;Code</a></strong>
<strong>&nbsp;&nbsp;•&nbsp;&nbsp;</strong>
<strong><a href="https://torch-simple-timing.readthedocs.io/" target="_blank">Docs&nbsp;&nbsp;📑</a></strong>
</p>

<p align="center">
    <a>
	    <img src='https://img.shields.io/badge/python-3.8%2B-blue' alt='Python' />
	</a>
	<a href='https://torch-simple-timing.readthedocs.io/en/latest/?badge=latest'>
    	<img src='https://readthedocs.org/projects/torch-simple-timing/badge/?version=latest' alt='Documentation Status' />
	</a>
    <a href="https://github.com/psf/black">
	    <img src='https://img.shields.io/badge/code%20style-black-black' />
	</a>
    <a href="https://pytorch.org">
        <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white"/>
    </a>
    <a href="https://pypi.org/project/torch-simple-timing/">
        <img src="https://badge.fury.io/py/torch_simple_timing.svg" alt="PyPI version" height="20">
    </a>
</p>
<br/>


# Torch Simple Timing

A simple yet versatile package to time CPU/GPU/Multi-GPU ops.

1. "*I want to time operations once*"
   1. That's what a `Clock` is for
2. "*I want to time the same operations multiple times*"
   1. That's what a `Timer` is for

In simple terms:

* A `Clock` is an object (and context-manager) that will compute the ellapsed time between its `start()` (or `__enter__`) and `stop()` (or `__exit__`)
* A `Timer` will internally manage clocks so that you can focus on readability and not data structures

## Installation

```
pip install torch_simple_timing
```

## How to use

### A `Clock`

```python
from torch_simple_parsing import Clock
import torch

t = torch.rand(2000, 2000)
gpu = torch.cuda.is_available()

with Clock(gpu=gpu) as context_clock:
    torch.inverse(t @ t.T)

clock = Clock(gpu=gpu).start()
torch.inverse(t @ t.T)
clock.stop()

print(context_clock.duration) # 0.29688501358032227
print(clock.duration)         # 0.292896032333374
```

More examples, including bout how to easily share data structures using a `store` can be found in the [documentation](https://torch-simple-timing.readthedocs.io/en/latest/autoapi/torch_simple_timing/clock/index.html).

### A `Timer`

```python
from torch_simple_timing import Timer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X = torch.rand(5000, 5000, device=device)
Y = torch.rand(5000, 100, device=device)
model = torch.nn.Linear(5000, 100).to(device)
optimizer = torch.optim.Adam(model.parameters())

gpu = device.type == "cuda"
timer = Timer(gpu=gpu)

for epoch in range(10):
    timer.clock("epoch").start()
    for b in range(50):
        x = X[b*100: (b+1)*100]
        y = Y[b*100: (b+1)*100]
        optimizer.zero_grad()
        with timer.clock("forward", ignore=epoch>0):
            p = model(x)
        loss = torch.nn.functional.cross_entropy(p, y)
        with timer.clock("backward", ignore=epoch>0):
            loss.backward()
        optimizer.step()
    timer.clock("epoch").stop()

stats = timer.stats()
# use stats for display and/or logging
# wandb.summary.update(stats)
print(timer.display(stats=stats, precision=5))
```

```
epoch    : 0.25064 ± 0.02728 (n=10)
forward  : 0.00226 ± 0.00526 (n=50)
backward : 0.00209 ± 0.00387 (n=50)
```

### A decorator

You can also use a decorator to time functions without much overhead in your code:

```python
from torch_simple_timing import timeit, get_global_timer, reset_global_timer
import torch

# Use the function name as the timer name
@timeit(gpu=True)
def train():
    x = torch.rand(1000, 1000, device="cuda" if torch.cuda.is_available() else "cpu")
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
    print(timer.display())

    reset_global_timer()
```

Prints:

```text
train : 0.045 ± 0.007 (n=10)
test  : 0.046         (n= 1)
```

By default the `@timeit` decodrator takes at least a `name`, will use `gpu=False` and use the global timer (`torch_simple_timing.TIMER`). You can pass your own timer with `@timeit(name, timer=timer)`.

See [in the docs]([https://](https://torch-simple-timing.readthedocs.io/en/latest/autoapi/torch_simple_timing/index.html)).
