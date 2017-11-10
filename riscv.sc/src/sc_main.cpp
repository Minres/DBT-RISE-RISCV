////////////////////////////////////////////////////////////////////////////////
// Copyright 2017 eyck@minres.com
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License.  You may obtain a copy
// of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
// License for the specific language governing permissions and limitations under
// the License.
////////////////////////////////////////////////////////////////////////////////
/*
 * sc_main.cpp
 *
 *  Created on: 17.09.2017
 *      Author: eyck@minres.com
 */

#include <boost/program_options.hpp>
#include <iss/jit/MCJIThelper.h>
#include <iss/log_categories.h>
#include <sstream>
#include <sysc/SiFive/platform.h>
#include "scc/configurer.h"
#include "scc/report.h"
#include "scc/scv_tr_db.h"
#include "scc/tracer.h"

using namespace sysc;
namespace po = boost::program_options;

namespace {
const size_t ERROR_IN_COMMAND_LINE = 1;
const size_t SUCCESS = 0;
const size_t ERROR_UNHANDLED_EXCEPTION = 2;
} // namespace

int sc_main(int argc, char *argv[]) {
    //    sc_report_handler::set_handler(my_report_handler);
    scc::Logger<>::reporting_level() = logging::ERROR;
    ///////////////////////////////////////////////////////////////////////////
    // CLI argument parsing
    ///////////////////////////////////////////////////////////////////////////
    po::options_description desc("Options");
    // clang-format off
    desc.add_options()
            ("help,h", "Print help message")
            ("verbose,v", po::value<int>()->implicit_value(0), "Sets logging verbosity")
            ("log-file", po::value<std::string>(), "Sets default log file.")
            ("disass,d", po::value<std::string>()->implicit_value(""), "Enables disassembly")
//            ("elf,l", po::value<std::vector<std::string>>(), "ELF file(s) to load")
            ("elf,l", po::value<std::string>(), "ELF file to load")
            ("gdb-port,g", po::value<unsigned short>()->default_value(0), "enable gdb server and specify port to use")
            ("dump-ir", "dump the intermediate representation")
            ("cycles", po::value<int64_t>()->default_value(-1), "number of cycles to run")
            ("quantum", po::value<unsigned>(), "SystemC quantum time in ns")
            ("reset,r", po::value<std::string>(), "reset address")
            ("trace,t", po::value<unsigned>()->default_value(0), "enable tracing, or combintation of 1=signals and 2=TX text, 4=TX compressed text, 6=TX in SQLite")
            ("max_time,m", po::value<std::string>(), "maximum time to run")
            ("config-file,c", po::value<std::string>()->default_value(""), "configuration file");
    // clang-format on
    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm); // can throw
        // --help option
        if (vm.count("help")) {
            std::cout << "DBT-RISE-RiscV simulator for RISC-V" << std::endl << desc << std::endl;
            return SUCCESS;
        }
        po::notify(vm); // throws on error, so do after help in case
                        // there are any problems
    } catch (po::error &e) {
        std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
        std::cerr << desc << std::endl;
        return ERROR_IN_COMMAND_LINE;
    }
    if (vm.count("verbose")) {
        auto l = logging::as_log_level(vm["verbose"].as<int>());
        LOGGER(DEFAULT)::reporting_level() = l;
        LOGGER(connection)::reporting_level() = l;
        LOGGER(SystemC)::reporting_level() = l;
        scc::Logger<>::reporting_level() = l;
    }
    if (vm.count("log-file")) {
        // configure the connection logger
        auto f = fopen(vm["log-file"].as<std::string>().c_str(), "w");
        LOG_OUTPUT(DEFAULT)::stream() = f;
        LOG_OUTPUT(connection)::stream() = f;
        LOG_OUTPUT(SystemC)::stream() = f;
    }
    ///////////////////////////////////////////////////////////////////////////
    // set up infrastructure
    ///////////////////////////////////////////////////////////////////////////
    iss::init_jit(argc, argv);
    ///////////////////////////////////////////////////////////////////////////
    // set up tracing & transaction recording
    ///////////////////////////////////////////////////////////////////////////
    auto trace_val = vm["trace"].as<unsigned>();
    scc::tracer trace("simple_system", static_cast<scc::tracer::file_type>(trace_val >> 1), trace_val != 0);
    ///////////////////////////////////////////////////////////////////////////
    // set up configuration
    ///////////////////////////////////////////////////////////////////////////
    scc::configurer cfg(vm["config-file"].as<std::string>());
    ///////////////////////////////////////////////////////////////////////////
    // instantiate top level
    ///////////////////////////////////////////////////////////////////////////
    platform i_simple_system("i_simple_system");
    // sr_report_handler::add_sc_object_to_filter(&i_simple_system.i_master,
    // sc_core::SC_WARNING, sc_core::SC_MEDIUM);
    // cfg.dump_hierarchy();
    if (vm.count("elf")) cfg.set_value("i_simple_system.i_core_complex.elf_file", vm["elf"].as<std::string>());
    if (vm.count("reset")) {
        auto str = vm["reset"].as<std::string>();
        uint64_t start_address = str.find("0x") == 0 ? std::stoull(str.substr(2), 0, 16) : std::stoull(str, 0, 10);
        cfg.set_value("i_simple_system.i_core_complex.reset_address", start_address);
    }
    if (vm.count("disass")) {
        cfg.set_value("i_simple_system.i_core_complex.enable_disass", true);
        LOGGER(disass)::reporting_level() = logging::INFO;
        auto file_name = vm["disass"].as<std::string>();
        if (file_name.length() > 0) {
            LOG_OUTPUT(disass)::stream() = fopen(file_name.c_str(), "w");
            LOGGER(disass)::print_time() = false;
            LOGGER(disass)::print_severity() = false;
        }
    }
    cfg.set_value("i_simple_system.i_core_complex.gdb_server_port", vm["gdb-port"].as<unsigned short>());
    cfg.set_value("i_simple_system.i_core_complex.dump_ir", vm.count("dump-ir") != 0);
    if (vm.count("quantum")) {
        tlm::tlm_global_quantum::instance().set(sc_core::sc_time(vm["quantum"].as<unsigned>(), sc_core::SC_NS));
    }
    ///////////////////////////////////////////////////////////////////////////
    // run simulation
    ///////////////////////////////////////////////////////////////////////////
    if(vm.count("max_time")){
    	sc_core::sc_time max_time = scc::parse_from_string(vm["max_time"].as<std::string>());
    	sc_core::sc_start(max_time);
    } else
    	sc_core::sc_start();
    if (!sc_core::sc_end_of_simulation_invoked()) sc_core::sc_stop();
    return 0;
}
