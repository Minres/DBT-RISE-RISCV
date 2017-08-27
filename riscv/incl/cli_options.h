/*******************************************************************************
 * Copyright (C) 2017, MINRES Technologies GmbH
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * 
 * Contributors:
 *       eyck@minres.com - initial API and implementation
 ******************************************************************************/

#ifndef _CLI_OPTIONS_H_
#define _CLI_OPTIONS_H_
#include <boost/program_options.hpp>
#include <easylogging++.h>

namespace {
const size_t ERROR_IN_COMMAND_LINE = 1;
const size_t SUCCESS = 0;
const size_t ERROR_UNHANDLED_EXCEPTION = 2;


inline void enable_log_level(el::Configurations& conf, int level){
	switch(level){
	case 0:
		conf.set(el::Level::Fatal, el::ConfigurationType::Enabled, "false");
		/* no break */
	case 1:
		conf.set(el::Level::Error, el::ConfigurationType::Enabled, "false");
		/* no break */
	case 2:
		conf.set(el::Level::Warning, el::ConfigurationType::Enabled, "false");
		/* no break */
	case 3:
		conf.set(el::Level::Info, el::ConfigurationType::Enabled, "false");
		/* no break */
	case 4:
		conf.set(el::Level::Debug, el::ConfigurationType::Enabled, "false");
		/* no break */
	case 5:
		conf.set(el::Level::Trace, el::ConfigurationType::Enabled, "false");
		/* no break */
	}
}

inline void configure_default_logger(boost::program_options::variables_map& vm){
	el::Configurations defaultConf;
	defaultConf.setToDefault();
	defaultConf.set(el::Level::Error, el::ConfigurationType::Format,   "%datetime{%H:%m:%s.%g} %level %msg");
	defaultConf.set(el::Level::Warning, el::ConfigurationType::Format, "%datetime{%H:%m:%s.%g} %level %msg");
	defaultConf.set(el::Level::Info, el::ConfigurationType::Format,    "%datetime{%H:%m:%s.%g} %level %msg");
	defaultConf.set(el::Level::Debug, el::ConfigurationType::Format,   "%datetime{%H:%m:%s.%g} %level %msg");
	defaultConf.set(el::Level::Trace, el::ConfigurationType::Format,   "%datetime{%H:%m:%s.%g} %level %msg");
    if(vm.count("verbose"))
    	enable_log_level(defaultConf, vm["verbose"].as<int>());
	if(vm.count("log-file"))
		defaultConf.set(el::Level::Global,el::ConfigurationType::Filename, vm["log-file"].as<std::string>());
	// default logger uses default configurations
	el::Loggers::reconfigureLogger("default", defaultConf);
}

inline void configure_debugger_logger() {
	// configure the connection logger
	el::Logger* gdbServerLogger = el::Loggers::getLogger("connection");
	el::Configurations gdbServerConf;
	gdbServerConf.setToDefault();
	gdbServerConf.set(el::Level::Error, el::ConfigurationType::Format,
			"%datetime{%H:%m:%s.%g} %level [%logger] %msg");
	gdbServerConf.set(el::Level::Warning, el::ConfigurationType::Format,
			"%datetime{%H:%m:%s.%g} %level [%logger] %msg");
	gdbServerConf.set(el::Level::Info, el::ConfigurationType::Format,
			"%datetime{%H:%m:%s.%g} %level [%logger] %msg");
	gdbServerConf.set(el::Level::Debug, el::ConfigurationType::Format,
			"%datetime{%H:%m:%s.%g} %level [%logger] %msg");
	gdbServerConf.set(el::Level::Trace, el::ConfigurationType::Format,
			"%datetime{%H:%m:%s.%g} %level [%logger] %msg");
	enable_log_level(gdbServerConf, 5);
	gdbServerLogger->configure(gdbServerConf);
}

inline void configure_disass_logger(boost::program_options::variables_map& vm) {
    el::Logger* disassLogger = el::Loggers::getLogger("disass");
    el::Configurations disassConf;
    if(vm.count("disass")){
        auto file_name=vm["disass"].as<std::string>();
        disassConf.setToDefault();
        if (file_name.length() > 0) {
            disassConf.set(el::Level::Global, el::ConfigurationType::ToFile,
                    std::string("true"));
            disassConf.set(el::Level::Global,
                    el::ConfigurationType::ToStandardOutput, std::string("false"));
            disassConf.set(el::Level::Global, el::ConfigurationType::Format,
                    std::string("%msg"));
            disassConf.set(el::Level::Global, el::ConfigurationType::Filename,
                    file_name);
            std::ofstream str(file_name); // just to clear the file
        } else {
            disassConf.set(el::Level::Global, el::ConfigurationType::Format,
                    "%datetime{%H:%m:%s.%g} [%logger] %msg");
        }
    } else {
        enable_log_level(disassConf, 0);
    }
	disassLogger->configure(disassConf);
}

} // namespace

inline int parse_cli_options(boost::program_options::variables_map& vm, int argc, char *argv[]){
    namespace po = boost::program_options;
    po::options_description desc("Options");
    desc.add_options()
        ("help,h", "Print help message")
		("verbose,v", po::value<int>()->implicit_value(0), "Sets logging verbosity")
		("vmodule", po::value<std::string>(),"Defines the module(s) to be logged")
		("logging-flags", po::value<int>(),"Sets logging flag(s).")
		("log-file", po::value<std::string>(),"Sets default log file.")
		("disass,d", po::value<std::string>()->implicit_value(""),"Enables disassembly")
        ("elf,l", po::value< std::vector<std::string> >(), "ELF file(s) to load")
        ("gdb-port,g", po::value<unsigned>(), "enable gdb server and specify port to use")
        ("input,i", po::value<std::string>(), "the elf file to load (instead of hex files)")
        ("dump-ir", "dump the intermediate representation")
        ("cycles,c", po::value<int64_t>()->default_value(-1), "number of cycles to run")
        ("systemc,s", "Run as SystemC simulation")
        ("time", po::value<int>(), "SystemC siimulation time in ms")
        ("reset,r", po::value<std::string>(), "reset address")
        ("trace", po::value<uint8_t>(),  "enable tracing, or cmbintation of 1=signals and 2=TX text, 4=TX compressed text, 6=TX in SQLite")\
        ("mem,m", po::value<std::string>(), "the memory input file");
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm); // can throw
        // --help option
        if ( vm.count("help")  ){
            std::cout << "JIT-ISS simulator for AVR" << std::endl << desc << std::endl;
            return SUCCESS;
        }
        po::notify(vm); // throws on error, so do after help in case
    } catch(po::error& e){
    	// there are problems
    	std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
        std::cerr << desc << std::endl;
        return ERROR_IN_COMMAND_LINE;
    }
	return SUCCESS;
}
#endif /* _CLI_OPTIONS_H_ */
