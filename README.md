# DBT-RISE-RiscV
Am instruction set simulator based on DBT-RISE implementing the Risc-V ISA

**DBT-RISE-RiscV README**

This is work in progress, so use at your own risk. Goal is to implement an open-source ISS which can easily embedded e.g. into SystemC Virtual Prototypes. It used code generation to allow easy extension and adaptation of the used instruction.
The Risc-V ISS reaches about 20MIPS at an Intel Core i7-2600K.

The implementation is based on LLVM 4.0. Eclipse CDT 4.7 (Oxygen) is recommended as IDE.

DBT-RISE-RiscV uses libGIS (https://github.com/vsergeev/libGIS) as well as ELFIO (http://elfio.sourceforge.net/), both under MIT license 

**What's missing**

* RV64I is only preliminary verified
* F & D standard extensions to be implemented

**Planned features**

* add platform peripherals to resemble E300 platform
  * PLIC
  * gpio
  * ...
* and more

**Quick start**

* you need to have a decent compiler, make and cmake installed
* install LLVM 4.0 according to http://apt.llvm.org/
* download and install SystemC from http://accellera.org/downloads/standards/systemc
  * optionally download and install SystemC Verification Library (SCV) from Accelera into the same location
* checkout source from git
* start an out-of-source build like so (e.g. when using LLVM 3.9 and bash)
```    
    cd DBT-RISE-RiscV
    mkdir build
    cd build
    cmake ..
    make
```
* if the SystemC installation is not found by cmake you can optionally specify the location by either setting the following environment variables pointing to the installation
  - SYSTEMC_HOME
  - SYSTEMC_PREFIX
  