////////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, MINRES Technologies GmbH
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Contributors:
//       eyck@minres.com - initial API and implementation
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <iss/iss.h>

#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include <iss/arch/riscv_hart_msu_vp.h>
#include <iss/arch/rv32imac.h>
#include <iss/arch/rv64ia.h>
#include <iss/jit/MCJIThelper.h>
#include <iss/log_categories.h>

namespace po = boost::program_options;

int main(int argc, char *argv[]) {
    /*
     *  Define and parse the program options
     */
    po::variables_map clim;
    po::options_description desc("Options");
    // clang-format off
    desc.add_options()
        ("help,h", "Print help message")
        ("loglevel,l", po::value<int>()->implicit_value(2), "Sets logging verbosity")
        ("logfile,f", po::value<std::string>(), "Sets default log file.")
        ("disass,d", po::value<std::string>()->implicit_value(""), "Enables disassembly")
        ("elf", po::value<std::vector<std::string>>(), "ELF file(s) to load")
        ("gdb-port,g", po::value<unsigned>()->default_value(0), "enable gdb server and specify port to use")
        ("input,i", po::value<std::string>(), "the elf file to load (instead of hex files)")
        ("dump-ir", "dump the intermediate representation")
        ("cycles,c", po::value<int64_t>()->default_value(-1), "number of cycles to run")
        ("systemc,s", "Run as SystemC simulation")
        ("time", po::value<int>(), "SystemC simulation time in ms")
        ("reset,r", po::value<std::string>(), "reset address")
        ("trace", po::value<uint8_t>(), "enable tracing, or cmbintation of 1=signals and 2=TX text, 4=TX compressed text, 6=TX in SQLite")
        ("mem,m", po::value<std::string>(), "the memory input file")
        ("isa", po::value<std::string>()->default_value("rv32imac"), "isa to use for simulation");
    // clang-format on
    auto parsed = po::command_line_parser(argc, argv).options(desc).allow_unregistered().run();
    try {
        po::store(parsed, clim); // can throw
        // --help option
        if (clim.count("help")) {
            std::cout << "DBT-RISE-RiscV simulator for RISC-V" << std::endl << desc << std::endl;
            return 0;
        }
        po::notify(clim); // throws on error, so do after help in case
    } catch (po::error &e) {
        // there are problems
        std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
        std::cerr << desc << std::endl;
        return 1;
    }
    std::vector<std::string> args = collect_unrecognized(parsed.options, po::include_positional);

    if (clim.count("loglevel")) {
        auto l = logging::as_log_level(clim["loglevel"].as<int>());
        LOGGER(DEFAULT)::reporting_level() = l;
        LOGGER(connection)::reporting_level() = l;
    }
    if (clim.count("logfile")) {
        // configure the connection logger
        auto f = fopen(clim["logfile"].as<std::string>().c_str(), "w");
        LOG_OUTPUT(DEFAULT)::stream() = f;
        LOG_OUTPUT(connection)::stream() = f;
    }

    try {
        // application code comes here //
        iss::init_jit(argc, argv);
        bool dump = clim.count("dump-ir");
        // instantiate the simulator
        std::unique_ptr<iss::vm_if> vm{nullptr};
        if (clim["isa"].as<std::string>().substr(0, 4)=="rv64") {
            iss::arch::rv64ia* cpu = new iss::arch::riscv_hart_msu_vp<iss::arch::rv64ia>();
            vm = iss::create(cpu, clim["gdb-port"].as<unsigned>(), dump);
        } else if (clim["isa"].as<std::string>().substr(0, 4)=="rv32") {
            iss::arch::rv32imac* cpu = new iss::arch::riscv_hart_msu_vp<iss::arch::rv32imac>();
            vm = iss::create(cpu, clim["gdb-port"].as<unsigned>(), dump);
        } else {
            LOG(ERROR) << "Illegal argument value for '--isa': " << clim["isa"].as<std::string>() << std::endl;
            return 127;
        }
        if (clim.count("elf"))
            for (std::string input : clim["elf"].as<std::vector<std::string>>()) vm->get_arch()->load_file(input);
        if (clim.count("mem"))
            vm->get_arch()->load_file(clim["mem"].as<std::string>(), iss::arch::traits<iss::arch::rv32imac>::MEM);
        for (std::string input : args) vm->get_arch()->load_file(input);// treat remaining arguments as elf files
        if (clim.count("disass")) {
            vm->setDisassEnabled(true);
            LOGGER(disass)::reporting_level() = logging::INFO;
            auto file_name = clim["disass"].as<std::string>();
            if (file_name.length() > 0) {
                LOG_OUTPUT(disass)::stream() = fopen(file_name.c_str(), "w");
                LOGGER(disass)::print_time() = false;
                LOGGER(disass)::print_severity() = false;
            }
        }
        if (clim.count("reset")) {
            auto str = clim["reset"].as<std::string>();
            auto start_address = str.find("0x") == 0 ? std::stoull(str.substr(2), 0, 16) : std::stoull(str, 0, 10);
            vm->reset(start_address);
        } else {
            vm->reset();
        }
        int64_t cycles = -1;
        cycles = clim["cycles"].as<int64_t>();
        return vm->start(cycles);
    } catch (std::exception &e) {
        LOG(ERROR) << "Unhandled Exception reached the top of main: " << e.what() << ", application will now exit"
                   << std::endl;
        return 2;
    }
}
