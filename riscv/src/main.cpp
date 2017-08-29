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

#include <iss/arch/rv32imac.h>
#ifndef WITHOUT_LLVM
#include <iss/jit/MCJIThelper.h>
#endif
#ifdef WITH_SYSTEMC
#include <sysc/kernel/sc_externs.h>
#endif
#include <boost/lexical_cast.hpp>

namespace po= boost::program_options;

INITIALIZE_EASYLOGGINGPP

int main(int argc, char *argv[]) {
    try{
        /** Define and parse the program options
         */
        po::variables_map vm;
        if(parse_cli_options(vm, argc, argv)) return ERROR_IN_COMMAND_LINE;
        configure_default_logger(vm);
        // configure the connection logger
    	configure_debugger_logger();

        // application code comes here //
        iss::init_jit(argc, argv);
        if(vm.count("systemc")){
//#ifdef WITH_SYSTEMC
//            return sc_core::sc_elab_and_sim(argc, argv);
//#else
            std::cerr<<"SystemC simulation is currently not supported, please rebuild with -DWITH_SYSTEMC"<<std::endl;
//#endif
        } else {
            bool  dump=vm.count("dump-ir");
            // instantiate the simulator
            std::unique_ptr<iss::vm_if> cpu = vm.count("gdb-port")?
                    iss::create<iss::arch::rv32imac>("rv32ima", vm["gdb-port"].as<unsigned>(), dump):
                    iss::create<iss::arch::rv32imac>("rv32ima", dump);
            if(vm.count("elf")){
                for(std::string input: vm["elf"].as<std::vector<std::string> >())
                    cpu->get_arch()->load_file(input);
            } else if(vm.count("mem")){
                cpu->get_arch()->load_file(vm["mem"].as<std::string>() , iss::arch::traits<iss::arch::rv32imac>::MEM);
            } //else
              //  LOG(FATAL)<<"At least one (flash-)input file (ELF or IHEX) needs to be specified";

            configure_disass_logger(vm);
            if(vm.count("disass")){
                cpu->setDisassEnabled(true);
            }
            if(vm.count("reset")){
                auto str = vm["reset"].as<std::string>();
                auto start_address = str.find("0x")==0? std::stoull(str, 0, 16):std::stoull(str, 0, 10);
                cpu->reset(start_address);
            } else {
                cpu->reset();
            }
            return cpu->start(vm["cycles"].as<int64_t>());
        }
    } catch(std::exception& e){
        LOG(ERROR) << "Unhandled Exception reached the top of main: "
                << e.what() << ", application will now exit" << std::endl;
        return ERROR_UNHANDLED_EXCEPTION;
    }
}
