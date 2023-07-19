# RNNTools.jl

This repository contains code to accompany the paper: Trainability, Expressivity and Interpretability in Gated Neural ODEs, ICML 2023 ([arXiv](https://arxiv.org/abs/2307.06398)). `RNNTools.jl` is a Julia package for training recurrent neural networks (RNNs) and neural ordinary differential equations (nODEs).

## Installation
The package was tested on Julia v1.6, but other versions may also work. To install `RNNTools.jl`, follow these steps:

```
$ module load julia/1.6.1
$ julia
julia> ]
(@v1.6) pkg> add Flux
(@v1.6) pkg> add <the address where you put the RNNTools repository>/RNNTools.jl
```

### Getting Started
- See `src/RNNTools.jl` and `experiments/basic/basic.jl` to get started on using the package.

### N-Bit Flip-Flop Task
- `run_flipflop_array.sh`: shell script used to run the flip-flop experiments.
#### Fixed-Amplitude 3-Bit Flip-Flop Task
- `n_bit_flipflop/three_bit/original/n_bit_flipflop_disc_time_array.jl`: julia code used to train various networks on the fixed-amplitude 3-bit flip-flop task (Sussillo & Barak, 2013).
#### Variable-Amplitude 3-Bit Flip-Flop Task
- `n_bit_flipflop/three_bit/variable/n_bit_flipflop_varying_amp_disc_time_array_gnode.jl`: julia code used to train gnODE on the variable-amplitude 3-bit flip-flop task.
- `n_bit_flipflop/three_bit/variable/n_bit_flipflop_varying_amp_disc_time_array_gru.jl`: julia code used to train GRU on the variable-amplitude 3-bit flip-flop task.
- `n_bit_flipflop/three_bit/variable/n_bit_flipflop_varying_amp_disc_time_array_mgru.jl`: julia code used to train mGRU on the variable-amplitude 3-bit flip-flop task.
- `n_bit_flipflop/three_bit/variable/n_bit_flipflop_varying_amp_disc_time_array_node.jl`: julia code used to train nODE on the variable-amplitude 3-bit flip-flop task.
- `n_bit_flipflop/three_bit/variable/n_bit_flipflop_varying_amp_disc_time_array_rnn.jl`: julia code used to train vanilla RNN on the variable-amplitude 3-bit flip-flop task.
#### Disk 2-Bit Flip-Flop Task
- `n_bit_flipflop/two_bit/disk/two_bit_flipflop_disk_disc_time_gnode.jl`: julia code used to train gnODE on the disk 2-bit flip-flop task.
#### Fixed-Amplitude (4 Stable Fixed Points) 2-Bit Flip-Flop Task
- `n_bit_flipflop/two_bit/original/two_bit_flipflop_disc_time_gnode.jl`: julia code used to train gnODE on the fixed-amplitude 2-bit flip-flop task (Sussillo & Barak, 2013).
#### Rectangle 2-Bit Flip-Flop Task
- `n_bit_flipflop/two_bit/rectangle/two_bit_flipflop_rectangle_disc_time_gnode.jl`: julia code used to train gnODE on the rectangle 2-bit flip-flop task.
#### Ring 2-Bit Flip-Flop Task
- `n_bit_flipflop/two_bit/ring/two_bit_flipflop_ring_disc_time_gnode.jl`: julia code used to train gnODE on the ring 2-bit flip-flop task.
#### Square 2-Bit Flip-Flop Task
- `n_bit_flipflop/two_bit/square/two_bit_flipflop_square_disc_time_gnode.jl`: julia code used to train gnODE on the square 2-bit flip-flop task.
- `n_bit_flipflop/two_bit/square/two_bit_flipflop_square_disc_time_gru.jl`: julia code used to train GRU on the square 2-bit flip-flop task.
- `n_bit_flipflop/two_bit/square/two_bit_flipflop_square_disc_time_mgru.jl`: julia code used to train mGRU on the square 2-bit flip-flop task.
- `n_bit_flipflop/two_bit/square/two_bit_flipflop_square_disc_time_node.jl`: julia code used to train nODE on the square 2-bit flip-flop task.
- `n_bit_flipflop/two_bit/square/two_bit_flipflop_square_disc_time_rnn.jl`: julia code used to train vanilla RNN on the square 2-bit flip-flop task.

## Citation

```bibtex
@article{kim2023gnode,
 title     = {{T}rainability, {E}xpressivity and {I}nterpretability in {G}ated {N}eural {ODE}s},
 author    = {Kim, Timothy Doyeon and Can, Tankut and Krishnamurthy, Kamesh}, 
 journal   = {Proceedings of the 40th International Conference on Machine Learning},
 year      = {2023}
}
```
