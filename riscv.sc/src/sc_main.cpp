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
//       eyck@minres.com - initial implementation
//
//
////////////////////////////////////////////////////////////////////////////////

#include <boost/program_options.hpp>
#include <iss/log_categories.h>
#include <sstream>
#include <sysc/General/system.h>
#include "scc/configurer.h"
#include "scc/report.h"
#include "scc/scv_tr_db.h"
#include "scc/tracer.h"
#include <cci_utils/broker.h>
#include <iss/jit/jit_helper.h>

using namespace sysc;
namespace po = boost::program_options;

namespace {
const size_t ERROR_IN_COMMAND_LINE = 1;
const size_t SUCCESS = 0;
const size_t ERROR_UNHANDLED_EXCEPTION = 2;
} // namespace

#include "sysc/kernel/sc_externs.h"
int
main( int argc, char* argv[]){
#ifdef _POSIX_SOURCE
    putenv(const_cast<char*>("SC_SIGNAL_WRITE_CHECK=DISABLE"));
    putenv(const_cast<char*>("SC_VCD_SCOPES=ENABLE"));
#endif
    return sc_core::sc_elab_and_sim( argc, argv );
}

int sc_main(int argc, char *argv[]) {
    //    sc_report_handler::set_handler(my_report_handler);
    scc::Logger<>::reporting_level() = logging::ERROR;
    cci::cci_register_broker(new cci_utils::broker("Global Broker"));
    ///////////////////////////////////////////////////////////////////////////
    // CLI argument parsing
    ///////////////////////////////////////////////////////////////////////////
    po::options_description desc("Options");
    // clang-format off
    desc.add_options()
            ("help,h", "Print help message")
            ("verbose,v", po::value<int>()->implicit_value(3), "Sets logging verbosity")
            ("log-file", po::value<std::string>(), "Sets default log file.")
            ("disass,d", po::value<std::string>()->implicit_value(""), "Enables disassembly")
            ("elf,l", po::value<std::string>(), "ELF file to load")
            ("gdb-port,g", po::value<unsigned short>()->default_value(0), "enable gdb server and specify port to use")
            ("dump-ir", "dump the intermediate representation")
            ("quantum", po::value<unsigned>(), "SystemC quantum time in ns")
            ("reset,r", po::value<std::string>(), "reset address")
            ("trace,t", po::value<unsigned>()->default_value(0), "enable tracing, or combintation of 1=signals and 2=TX text, 4=TX compressed text, 6=TX in SQLite")
            ("max_time,m", po::value<std::string>(), "maximum time to run")
            ("config-file,c", po::value<std::string>()->default_value(""), "read configuration from file")
			("dump-config", po::value<std::string>()->default_value(""), "dump configuration to file file");
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
    sysc::system i_system("i_system");
    // sr_report_handler::add_sc_object_to_filter(&i_simple_system.i_master,
    // sc_core::SC_WARNING, sc_core::SC_MEDIUM);
    if(vm["dump-config"].as<std::string>().size()>0){
    	std::ofstream of{vm["dump-config"].as<std::string>()};
    	if(of.is_open())
    	    cfg.dump_configuration(of);
    }
	cfg.configure();
    // overwrite with command line settings
    if (vm["gdb-port"].as<unsigned short>())
    	cfg.set_value("i_system.i_platform.i_core_complex.gdb_server_port", vm["gdb-port"].as<unsigned short>());
    if (vm.count("dump-ir"))
    	cfg.set_value("i_system.i_platform.i_core_complex.dump_ir", vm.count("dump-ir") != 0);
    if (vm.count("elf"))
    	cfg.set_value("i_system.i_platform.i_core_complex.elf_file", vm["elf"].as<std::string>());
    if (vm.count("quantum"))
        tlm::tlm_global_quantum::instance().set(sc_core::sc_time(vm["quantum"].as<unsigned>(), sc_core::SC_NS));
    if (vm.count("reset")) {
        auto str = vm["reset"].as<std::string>();
        uint64_t start_address = str.find("0x") == 0 ? std::stoull(str.substr(2), 0, 16) : std::stoull(str, 0, 10);
        cfg.set_value("i_system.i_platform.i_core_complex.reset_address", start_address);
    }
    if (vm.count("disass")) {
        cfg.set_value("i_system.i_platform.i_core_complex.enable_disass", true);
        LOGGER(disass)::reporting_level() = logging::INFO;
        auto file_name = vm["disass"].as<std::string>();
        if (file_name.length() > 0) {
            LOG_OUTPUT(disass)::stream() = fopen(file_name.c_str(), "w");
            LOGGER(disass)::print_time() = false;
            LOGGER(disass)::print_severity() = false;
        }
    }
    ///////////////////////////////////////////////////////////////////////////
    // run simulation
    ///////////////////////////////////////////////////////////////////////////
    try {
        if(vm.count("max_time")){
            sc_core::sc_time max_time = scc::parse_from_string(vm["max_time"].as<std::string>());
            sc_core::sc_start(max_time);
        } else
            sc_core::sc_start();
    } catch(sc_core::sc_report& rep){
        CLOG(FATAL, SystemC)<<"IWEF"[rep.get_severity()]<<"("<<rep.get_id()<<") "<<rep.get_msg_type()<<": "<<rep.get_msg()<<std::endl;
    }
    if (!sc_core::sc_end_of_simulation_invoked()) sc_core::sc_stop();
    return 0;
}
