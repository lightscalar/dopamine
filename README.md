# DOPAMINE: A Tensorflow Implementation of Trust Region Policy Optimization

This is a Tensorflow-based implementation of [Trust Region Policy
Optimization](https://arxiv.org/abs/1502.05477), a deep reinforcement learning
technique. This code has been tested on Python 3.5+.

This implementation is intended to be more pedagogical than optimal. We focus
here on understanding the algorithm via clear code. That said, when run on a
GPU, it performs just fine on moderately large problems.

## Installation

Installation is easy. Just add the root directory to your local Python path in
your `.profile`.

```unix
export PYTHONPATH="${PYTHONPATH}:/Users/username/path/to/dopamine"
```

## Tests

Tests can be executed using [nose](http://nose.readthedocs.io/en/latest/). Just
run it from the ```tests/``` directory.

