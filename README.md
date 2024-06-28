# DBT-RISE-RISCV
Core of an instruction set simulator based on DBT-RISE implementing the RISC-V ISA. The project is hosted at https://git.minres.com/DBT-RISE/DBT-RISE-RISCV .

This repo contains only the code of the RISC-V ISS and can only be used with the DBT_RISE. A complete VP using this ISS can be found at https://git.minres.com/VP/RISCV-VP which models SiFives FE310 controlling a brushless DC (BLDC) motor.

This library provide the infrastructure to build RISC-V ISS. Currently part of the library are the following implementations adhering to version 2.2 of the 'The RISC-V Instruction Set Manual Volume I: User-Level ISA':

* RV32IMAC
* RV32GC
* RC64I
* RV64GC

All pass the respective compliance tests. Along with those ISA implementations there is a wrapper implementing the M/S/U modes inlcuding virtual memory management and CSRs as of privileged spec 1.10. The main.cpp in src allows to build a standalone ISS when integrated into a top-level project. For further information please have a look at [https://git.minres.com/VP/RISCV-VP](https://git.minres.com/VP/RISCV-VP).

Last but not least an SystemC wrapper is provided which allows easy integration into SystemC based virtual platforms.

Since DBT-RISE uses a generative approch other needed combinations or custom extension can be generated. For further information please contact [info@minres.com](mailto:info@minres.com).


conan install . --output-folder=. --build=missing -s compiler.cppstd=17
cmake -S . -B build/Release -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=build/Release/generators/conan_toolchain.cmake
cmake --preset conan-release
 