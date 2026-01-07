[![CI C++ Std compliance](https://github.com/Minres/DBT-RISE-RISCV/actions/workflows/ci-compile.yml/badge.svg)](https://github.com/Minres/DBT-RISE-RISCV/actions/workflows/ci-compile.yml)

# DBT-RISE-RISCV
A instruction set simulator based on DBT-RISE implementing the RISC-V ISA.
The project is hosted at https://github.com/Minres/DBT-RISE-RISCV .

This repo contains only the code of the RISC-V ISS.
A complete VP using this ISS can be found at https://github.com/Minres/RISCV-VP which models some generic RISC-V µC.

## Contents
While this project works as a standalone appplication it is based on the [DBT-RISE](https://github.com/Minres/DBT-RISE-Core) Framework for developing ISS.
Currently part of the repository are the following implementations adhering to version 2.2 of the 'The RISC-V Instruction Set Manual Volume I: User-Level ISA':

* RV32I
* RV32IMAC
* RV32GC
* RC64I
* RV64GC
  
All possible selections and backends are achitectural compliant, passing the official [riscv-arch-test](https://github.com/riscv-non-isa/riscv-arch-test) suite for M-mode.

Along with the different ISA implementations there is a wrapper implementing the M/S/U modes including virtual memory management and CSRs as of privileged spec 1.10. The main.cpp in src allows to build a standalone ISS when integrated into a top-level project.
For an example take a look at [ https://github.com/Minres/RISCV-VP]( https://github.com/Minres/RISCV-VP).

Last but not least a SystemC wrapper is provided to allow easy integration into SystemC based virtual platforms (e.g. https://github.com/Minres/RISCV-VP).

Since DBT-RISE uses a generative approach other needed combinations or custom extension can be generated. For further information please do not hesitate to contact [info@minres.com](mailto:info@minres.com).

## Standalone build instructions

You need to have conan newer than version 2.0 available.
If you do not have it already it can be done in the following way (assuming you are using bash):

```
python3 -m venv .venv
. .venv/bin/activate
pip3 install conan
conan profile detect
```

Building the ISS is as simple as:

```
cmake -S . -B build/Release --preset Release && cmake --build build/Release -j24
```

Building a debug version is analogous:

```
cmake -S . -B build/Debug --preset Debug && cmake --build build/Debug -j24
```

To run a simple test one can use the MINRES Firmware examples:

```
git clone --recursive -b develop https://git.minres.com/Firmware/Firmwares.git build/Firmwares
make -C build/Firmwares/hello-world/ ISA=imc BOARD=iss
./build/Release/riscv-sim -f build/Firmwares/hello-world/hello.elf
```

## Highlights
- High level generative approach using C like syntax
- JIT-backends for higher execution speeds
- Built-in plugin support to inspect or manipulate architectural state before and after each instruction
- Full GDB-server
- Easy integration into SystemC or similar (eg. Platform Architect)