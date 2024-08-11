# Reinforce

[![C++ CI](https://github.com/maichmueller/reinforce/actions/workflows/cpp.yml/badge.svg)](https://github.com/maichmueller/reinforce/actions/workflows/cpp.yml)

`Reinforce` is a C++ port of the [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) interfaces, designed for those who want to dive into Reinforcement Learning (RL) with a performance boost. It offers Gymnasium-like spaces and environments, so you can wring every last ounce of speed from your training routines.

Keep in mind, that this is very early work in progress and that any API may be subject to major change.

## Features

- **Gym Spaces:** Reinforce provides C++ implementations of common Gymnasium spaces whenever meaningful. For a seamless
  transition from the Python-based standard set by Gym their API and behaviour is replicated as closely as possible.
  All spaces implemented so far provide the public API:
    - (batch) `sampling`
    - `contains` checks
    - `repr`
    - `equality` comparison

  As of now, the following spaces are available:
    - `Discrete`
    - `Box`
    - `Graph`
    - `MultiDiscrete`
    - `MultiBinary`
    - `OneOf`
    - `Text`
    - `Tuple`
    - `Sequence`

- **XTensor:** Reinforce uses [xtensor](https://github.com/xtensor-stack/xtensor) as its tensor operations backbone akin
  to `NumPy` for `Gymnasium`. This allows for speedy generation and processing of sampled data from spaces. The
  `xtensor` API is designed to resemble `NumPy`'s API (see [xtensor-docs](https://xtensor.readthedocs.io/en/latest/)).
- **Environments:** Reinforce plans to offer a small selection of environments for training RL models.
  Currently, only a version of `gridworld` of arbitrary dimensions is included.
- **Python Bindings**: In the future, I hope to export Reinforce to Python. Given the heavy template reliance, this is
  still being evaluated for feasibility. If you have advice or wish to share the workload on this, feel free to open an
  issue
  or discussion item!

## Examples

### Sampling From Box Space.

<table>
<tr>
<th>Gymnasium</th>
<th>Reinforce</th>
</tr>
<tr>
<td>

```python
from gymnasium import spaces
from gymnasium.vector.utils import concatenate
import numpy as np

# high and low boundaries for samples
# of shape (2,3). We have that
# low[i,j] <= samples[:, i,j] <= high[i,j]
low = np.array([[-np.inf, 0.0, -4.0],
                [4.0, 16.0, 64.0]])
high = np.array([[np.inf, 2.0, -2.0],
                 [8.0, 32.0, np.inf]])
space = spaces.Box(low, high)

with np.printoptions(precision=1):
    # sample either a single sample
    print(f"Sample:\n{space.sample()}")
    # or a batch of samples
    out = np.empty((100,) + space.shape, dtype=space.dtype)
    samples = tuple(space.sample() for _ in range(100))
    print(f"Batch:\n{concatenate(space, samples, out)}")
```

</td>
<td>

```cpp
#include <reinforce/spaces/box.hpp>
#include <reinforce/utils/math.hpp>
using namespace force;

// high and low boundaries for samples
// of shape (2,3). We have that
// low[i,j] <= samples[:, i,j] <= high[i,j]
xarray< double > low {{-inf<>,  0,    -4},
                      {     4, 16,    64}};
xarray< double > high{{ inf<>,  2,    -2},
                      {     8, 32, inf<>}};
auto space = BoxSpace{low, high};

xt::print_options::set_precision(1);
// sample either a single sample
fmt::println("Sample:\n{}", space.sample());
// or a batch of samples
fmt::println("Batch:\n{}", space.sample(100));



```

</td>
</tr>

<tr>
<td>

```cpp
Sample:
[[-1.2  1.6 -2.1]
 [ 5.6 28.1 64.9]]
Batch:
[[[ 0.4  1.5 -3.4]
  [ 7.7 29.1 64.3]]

 [[-1.   0.4 -3.8]
  [ 4.8 24.  64.1]]

 [[-0.5  1.9 -2.2]
  [ 5.1 22.3 65.3]]

 ...

 [[ 0.5  1.3 -3.4]
  [ 4.2 25.3 64.8]]

 [[-1.6  0.5 -3.8]
  [ 5.9 23.4 64. ]]

 [[ 1.1  1.8 -2.3]
  [ 5.2 29.2 64. ]]]
```

</td>
<td>

```cpp
Sample:
{{  1.3,   0.1,  -3. },
 {  6.9,  22.6,  64.4}}
Batch:
{{{ -1.6,   0.6,  -2.3},
  {  7.8,  29.3,  64.7}},
  
 {{ -0.3,   0.7,  -3.6},
  {  6.8,  27.4,  64.1}},
  
 {{  1.3,   1. ,  -3.7},
  {  5.3,  23.4,  64.2}},
  
 ...,
 
 {{ -0.1,   0.5,  -2.2},
  {  4.9,  24.8,  64. }},
  
 {{  0. ,   0.7,  -2. },
  {  7.5,  16.3,  64.8}},
  
 {{ -1. ,   0.9,  -3.4},
  {  5.5,  25.7,  64.1}}}
```

</td>
</tr>
</table>

## Getting Started

This project relies on several dependencies to function correctly:

- **CMake**, a cross-platform open-source build system generator.

- **Conan**, a C/C++ package manager.

The specific dependencies for the code to function are listed in the `conandata.yml` file. Conan will handle the
installation of these.

To install cmake/conan, you can use the following commands:

```bash
sudo apt-get install cmake
pip install conan
```

Please note that these commands are for Ubuntu. If you're using a different operating system, please adjust the commands
accordingly.

After installing Conan, you can install the project-specific dependencies by running:

```bash
conan install .
```

This command needs to be run in the project directory and will install all the necessary dependencies listed in
the `conandata.yml` file, and partially in `conanfile.py`.
The provided `configure.sh` and `build.sh` scripts will handle the dependency installation from conan and subsequent
cmake configuration automatically; by default, it will configure a build folder named `build` in the project root.

To get started with this, clone the repository and build it by running the following commands:

```bash
git clone https://github.com/maichmueller/reinforce.git
cd reinforce
./configure.sh --output build
./build.sh build
```

After building the project, you can install it by running (WIP):

```bash
./install.sh build
```

## Documentation

As of now, the documentation is still a work in progress. However, the test files under `tests` showcase basic usage.

## Contributing

Contributions to Reinforce are welcome.

## License

Reinforce is licensed under the MIT License. See `LICENSE` for more information.

## Contact

If you have any questions or suggestions, please open an issue on GitHub.
