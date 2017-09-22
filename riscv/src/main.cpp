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


#include <cli_options.h>
#include <iss/iss.h>
#include <iostream>

#include <iss/log_categories.h>
#include <iss/arch/rv32imac.h>
#include <iss/arch/rv64ia.h>
#include <iss/jit/MCJIThelper.h>
#include <boost/lexical_cast.hpp>

namespace po= boost::program_options;

int main(int argc, char *argv[]) {
    try{
        /*
         *  Define and parse the program options
         */
        po::variables_map vm;
        if(parse_cli_options(vm, argc, argv)) return ERROR_IN_COMMAND_LINE;
        if(vm.count("verbose")){
            auto l = logging::as_log_level(vm["verbose"].as<int>());
            LOGGER(DEFAULT)::reporting_level() = l;
            LOGGER(connection)::reporting_level()=l;
        }
        if(vm.count("log-file")){
            // configure the connection logger
            auto f = fopen(vm["log-file"].as<std::string>().c_str(), "w");
            LOG_OUTPUT(DEFAULT)::stream() = f;
            LOG_OUTPUT(connection)::stream() = f;
        }

        // application code comes here //
        iss::init_jit(argc, argv);
        bool  dump=vm.count("dump-ir");
        // instantiate the simulator
        std::unique_ptr<iss::vm_if> cpu = nullptr;
        if(vm.count("rv64")==1){
            if(vm.count("gdb-port")==1)
                cpu = iss::create<iss::arch::rv64ia>("rv64ia", vm["gdb-port"].as<unsigned>(), dump);
            else
                cpu = iss::create<iss::arch::rv64ia>("rv64ia", dump);
        } else {
            if(vm.count("gdb-port")==1)
                cpu = iss::create<iss::arch::rv32imac>("rv32ima", vm["gdb-port"].as<unsigned>(), dump);
            else
                cpu = iss::create<iss::arch::rv32imac>("rv32ima", dump);
        }
        if(vm.count("elf")){
            for(std::string input: vm["elf"].as<std::vector<std::string> >())
                cpu->get_arch()->load_file(input);
        } else if(vm.count("mem")){
            cpu->get_arch()->load_file(vm["mem"].as<std::string>() , iss::arch::traits<iss::arch::rv32imac>::MEM);
        }

        if(vm.count("disass")){
            cpu->setDisassEnabled(true);
            LOGGER(disass)::reporting_level()=logging::INFO;
            auto file_name=vm["disass"].as<std::string>();
            if (file_name.length() > 0) {
                LOG_OUTPUT(disass)::stream() = fopen(file_name.c_str(), "w");
                LOGGER(disass)::print_time() = false;
                LOGGER(disass)::print_severity() = false;
            }
        }
        if(vm.count("reset")){
            auto str = vm["reset"].as<std::string>();
            auto start_address = str.find("0x")==0? std::stoull(str, 0, 16):std::stoull(str, 0, 10);
            cpu->reset(start_address);
        } else {
            cpu->reset();
        }
        return cpu->start(vm["cycles"].as<int64_t>());
    } catch(std::exception& e){
        LOG(ERROR) << "Unhandled Exception reached the top of main: "
                << e.what() << ", application will now exit" << std::endl;
        return ERROR_UNHANDLED_EXCEPTION;
    }
}
