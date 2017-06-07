# DOPAMINE: A Tensorflow Implementation of Trust Region Policy Optimization

This is a Tensorflow-based implementation of [Trust Region Policy
Optimization](https://arxiv.org/abs/1502.05477), a deep reinforcement learning
technique. This code has been tested on Python 3.5+.

This implementation is intended to be more pedagogical than optimal. We focus
here on understanding the algorithm via clear code. That said, when run on a
GPU, it performs just fine on moderately large problems.

This code is associated with a forthcoming blog post detailing the algorithm
and discussing how to efficiently implement it using tensorflow. Check back
here for that link.

## Installation

Installation is easy. Just add the root directory to your local Python path in
your `.profile`.

```unix
export PYTHONPATH="${PYTHONPATH}:/Users/username/path/to/dopamine"
```

## Run an Experiment

To get started, run one of the simple experiments in the `/scripts` directory.
I recommend starting with `/scripts/lineworld_experiments.py` to see get a
quick sense of how this works.

## Tests

Tests can be executed using [nose](http://nose.readthedocs.io/en/latest/). Just
run it from the ```tests/``` directory.

