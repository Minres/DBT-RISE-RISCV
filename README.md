# JIT-ISS
A versatile Just-in-time (JIT) compiling instruction set simulator (ISS)

**JIT-ISS README**

This is currently a proof of concept and work in progress, so use at your own risk. It is currently set-up as an Eclipse CDT project and based on LLVM. To build it you need latest LLVM and Eclipse CDT version 4.6 aka Neon.

To build with SystemC the define WITH_SYSTEMC needs to be set. Then a simple proof-of-concept system is created. Mainly missing are platform peripherals and interrupt handling. It reaches about 5 MIPS in lock-step mode on a MacBook Pro (Core i7-4870HQ@2.5GHz) running in a Docker container.

JIT-ISS uses libGIS (https://github.com/vsergeev/libGIS) as well as ELFIO (http://elfio.sourceforge.net/), both under MIT license 

**What's missing**

* only AVR instruction set implemented but not verified

**Planned features**

* add platform peripherals
  * timers
  * gpio
  * ext interrupt registers and functionality
* and more

**Quick start**

* you need to have a decent compiler, make and cmake installed
* install LLVM 3.9 or 4.0 according to http://apt.llvm.org/
* download and install SystemC from http://accellera.org/downloads/standards/systemc
  * optionally download and install SystemC Verification Library (SCV) from Accelera into the same location
* checkout source from git
* start an out-of-source build like so (e.g. when using LLVM 3.9 and bash)
```    
    cd JIT-ISS
    mkdir build
    cd build
    LLVM_HOME=/usr/lib/llvm-3.9 cmake ..
    make
```
* if the SystemC installation is not to be found be cmake you can optionally specify the location by either setting the following environment variables pointing to the installation
  - SYSTEMC_HOME
  - SYSTEMC_PREFIX
  