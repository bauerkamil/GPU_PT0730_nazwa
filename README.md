# GPU_PT0730_sito_Eratostenesa_sito_Sundarama

### Table of contents
* [General info](#general-info)
* [Authors](#authors)
* [Instructions](#instructions)
    - [Prerequisites](#prerequisites)
    - [Make commands](#make-commands)

### General info

This is a project for Acceleration of calculations in data processing (*Akceleracja obliczeń w przetwarzaniu danych*).
It's main goal is to implement two algorithms: the sieve of Eratosthenes and the sieve of Sundaram with an appropriate distribution of calculations on multithreaded CPU and on GPU.

### Authors

* Kacper Aleks, 259086
* Kamil Bauer, 259102
* Damian Gnieciak, 259065
* Szymon Leja, 259047
* Patryk Uzarowski, 259105

### Instructions

#### Prerequisites

* g++ installed
* Make installed
* NVIDIA CUDA Toolkit installed

#### Make commands

The project's Makefile includes Bash scripts (running on PowerShell or other interpreters might not work). To get a full list of make commands input `make` into command line. Underneath is a list of most important commands.

|Command|Description|
|---|---|
|`make run1`|compiles and runs the sieve of Eratosthenes on multithreaded CPU|
|`make run2`|compiles and runs the sieve of Sundaram on multithreaded CPU|
|`make run3`|compiles and runs the sieve of Eratosthenes on GPU|
|`make run4`|compiles and runs the sieve of Sundaram on GPU|
|`make clean`|removes bin folder|
