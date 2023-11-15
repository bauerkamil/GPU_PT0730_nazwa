# GPU_PT0730_sito_Eratostenesa_sito_Sundarama

### Table of contents

- [General info](#general-info)
- [Authors](#authors)
- [Instructions](#instructions)
  - [Prerequisites](#prerequisites)
  - [Make commands](#make-commands)

### General info

This is a project for Acceleration of calculations in data processing (_Akceleracja obliczeń w przetwarzaniu danych_).
It's main goal is to implement two algorithms: the sieve of Eratosthenes and the sieve of Sundaram with an appropriate distribution of calculations on multithreaded CPU and on GPU.

### Authors

- Kacper Aleks, 259086
- Kamil Bauer, 259102
- Damian Gnieciak, 259065
- Szymon Leja, 259047
- Patryk Uzarowski, 259105

### Instructions

#### Prerequisites

- g++ installed
- Make installed
- NVIDIA CUDA Toolkit installed

#### Make commands

The project's Makefile includes Bash scripts (running on PowerShell or other interpreters might not work). To get a full list of make commands input `make` into command line. Underneath is a list of most important commands.

| Command            | Description                                                      |
| ------------------ | ---------------------------------------------------------------- |
| `make run_era`     | compiles and runs the sieve of Eratosthenes on multithreaded CPU |
| `make run_sun`     | compiles and runs the sieve of Sundaram on multithreaded CPU     |
| `make run_era_gpu` | compiles and runs the sieve of Eratosthenes on GPU               |
| `make run_sun_gpu` | compiles and runs the sieve of Sundaram on GPU                   |
| `make clean`       | removes bin folder                                               |
