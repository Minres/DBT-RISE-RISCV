# DBT-RISE-RISCV
Core of an instruction set simulator based on DBT-RISE implementing the RISC-V ISA.
The project is hosted at https://git.minres.com/DBT-RISE/DBT-RISE-RISCV .

This repo contains only the code of the RISC-V ISS.
A complete VP using this ISS can be found at https://github.com/eyck/RISCV-VP which models some generic RISC-V µC.

This library provide the infrastructure to build RISC-V ISS.
Currently part of the library are the following implementations adhering to version 2.2 of the 'The RISC-V Instruction Set Manual Volume I: User-Level ISA':

* RV32I
* RV32IMAC
* RV32GC
* RC64I
  
Along with those ISA implementations there is a wrapper implementing the M/S/U modes including virtual memory management and CSRs as of privileged spec 1.10. The main.cpp in src allows to build a standalone ISS when integrated into a top-level project.
For further information please have a look at [https://git.minres.com/VP/RISCV-VP](https://git.minres.com/VP/RISCV-VP).

Last but not least a SystemC wrapper is provided to allow easy integration into SystemC based virtual platforms (e.g. https://github.com/eyck/RISCV-VP).

Since DBT-RISE uses a generative approach other needed combinations or custom extension can be generated. For further information please contact [info@minres.com](mailto:info@minres.com).

## Standalone build instructions

You need to have conan newer than version 2.0 available.
If you do not have it already it can be done in the following way (assuming you are using bash):

```
python3 -m venv .venv
. .venv/bin/activate
pip3 install conan
conan profile new default --detect
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
