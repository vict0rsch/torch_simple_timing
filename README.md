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

More examples, including bout how to easily share data structures using a `store` can be found in the [documentation]().

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
    timer.mark("epoch").start()
    for b in range(50):
        x = X[b*100: (b+1)*100]
        y = Y[b*100: (b+1)*100]
        optimizer.zero_grad()
        with timer.mark("forward", ignore=epoch>0):
            p = model(x)
        loss = torch.nn.functional.cross_entropy(p, y)
        with timer.mark("backward", ignore=epoch>0):
            loss.backward()
        optimizer.step()
    timer.mark("epoch").stop()

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
