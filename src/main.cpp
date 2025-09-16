/*******************************************************************************
 * Copyright (C) 2017, 2018 MINRES Technologies GmbH
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
 *******************************************************************************/

#include <array>
#include <cstdint>
#include <fmt/format.h>
#include <fstream>
#include <iostream>
#include <iss/factory.h>
#include <iss/semihosting/semihosting.h>
#include <string>
#include <unordered_map>
#include <util/ities.h>
#include <vector>

#include "util/logging.h"
#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#ifdef WITH_LLVM
#include <iss/llvm/jit_init.h>
#endif
#include "iss/plugin/cycle_estimate.h"
#include "iss/plugin/instruction_count.h"
#include <iss/log_categories.h>
#ifndef WIN32
#include <iss/plugin/loader.h>
#endif
#if defined(HAS_LUA)
#include <iss/plugin/lua.h>
#endif

namespace po = boost::program_options;
int main(int argc, char* argv[]) {
    /*
     *  Define and parse the program options
     */
    po::variables_map clim;
    po::options_description desc("Options");
    // clang-format off
    desc.add_options()
        ("help,h", "Print help message")
        ("verbose,v", po::value<int>()->default_value(4), "Sets logging verbosity")
        ("logfile,l", po::value<std::string>(), "Sets default log file.")
        ("disass,d", po::value<std::string>()->implicit_value(""), "Enables disassembly")
        ("gdb-port,g", po::value<unsigned>()->default_value(0), "enable gdb server and specify port to use")
        ("ilimit,i", po::value<uint64_t>()->default_value(std::numeric_limits<uint64_t>::max()), "max. number of instructions to simulate")
        ("flimit", po::value<uint64_t>()->default_value(std::numeric_limits<uint64_t>::max()), "max. number of fetches to simulate")
        ("reset,r", po::value<std::string>(), "reset address")
        ("dump-ir", "dump the intermediate representation")
        ("elf,f", po::value<std::vector<std::string>>(), "ELF file(s) to load")
        ("mem,m", po::value<std::string>(), "the memory input file")
        ("plugin,p", po::value<std::vector<std::string>>(), "plugin to activate")
        ("backend", po::value<std::string>()->default_value("interp"), "the ISS backend to use, options are: interp, llvm, tcc, asmjit")
        ("isa", po::value<std::string>()->default_value("rv32imac_m"), "core or isa name to use for simulation, use '?' to get list");
    // clang-format on
    auto parsed = po::command_line_parser(argc, argv).options(desc).allow_unregistered().run();
    try {
        po::store(parsed, clim); // can throw
        // --help option
        if(clim.count("help")) {
            std::cout << "DBT-RISE-TGC simulator for TGC RISC-V cores" << std::endl << desc << std::endl;
            return 0;
        }
        po::notify(clim); // throws on error, so do after help in case
    } catch(po::error& e) {
        // there are problems
        std::cerr << "ERROR: " << e.what() << std::endl << std::endl;
        std::cerr << desc << std::endl;
        return 1;
    }
    std::vector<std::string> args = collect_unrecognized(parsed.options, po::include_positional);

    LOGGER(DEFAULT)::print_time() = false;
    LOGGER(connection)::print_time() = false;
    auto l = logging::as_log_level(clim["verbose"].as<int>());
    LOGGER(DEFAULT)::reporting_level() = l;
    LOGGER(connection)::reporting_level() = l;
    if(clim.count("logfile")) {
        // configure the connection logger
        auto f = fopen(clim["logfile"].as<std::string>().c_str(), "w");
        LOG_OUTPUT(DEFAULT)::stream() = f;
        LOG_OUTPUT(connection)::stream() = f;
    }

    std::vector<iss::vm_plugin*> plugin_list;
    auto res = 0;
    try {
#ifdef WITH_LLVM
        // application code comes here //
        iss::init_jit_debug(argc, argv);
#endif
        bool dump = clim.count("dump-ir");
        auto& f = iss::core_factory::instance();
        // instantiate the simulator
        iss::vm_ptr vm{nullptr};
        iss::cpu_ptr cpu{nullptr};
        semihosting_callback<uint32_t> cb{};
        semihosting_cb_t<uint32_t> semihosting_cb = [&cb](iss::arch_if* i, uint32_t* a0, uint32_t* a1) { cb(i, a0, a1); };
        std::string isa_opt(clim["isa"].as<std::string>());
        if(isa_opt.size() == 0 || isa_opt == "?") {
            std::unordered_map<std::string, std::vector<std::string>> core_by_backend;
            for(auto& e: f.get_names()) {
                auto p = e.find(':');
                assert(p!=std::string::npos);
                core_by_backend[e.substr(p+1)].push_back(e.substr(0, p));
            }
            std::cout << "Available implementations\n";
            std::cout << "=========================\n";
            for(auto& e:core_by_backend) {
                std::sort(std::begin(e.second), std::end(e.second));
                std::cout<<"  backend "<<e.first<<":\n  - "<< util::join(e.second, "\n  - ") << std::endl;
            }
            return 0;
        } else if(isa_opt.find(':') == std::string::npos) {
            std::tie(cpu, vm) =
                f.create(isa_opt + ":" + clim["backend"].as<std::string>(), clim["gdb-port"].as<unsigned>(), &semihosting_cb);
        } else {
            auto base_isa = isa_opt.substr(0, 5);
            if(base_isa == "tgc5d" || base_isa == "tgc5e") {
                isa_opt += "_clic_pmp:" + clim["backend"].as<std::string>();
            } else {
                isa_opt += ":" + clim["backend"].as<std::string>();
            }
            std::tie(cpu, vm) = f.create(isa_opt, clim["gdb-port"].as<unsigned>(), &semihosting_cb);
        }
        if(!cpu) {
            auto list = f.get_names();
            std::sort(std::begin(list), std::end(list));
            CPPLOG(ERR) << "Could not create cpu for isa " << isa_opt << " and backend " << clim["backend"].as<std::string>() << "\n"
                        << "Available implementations (core|platform|backend):\n  - " << util::join(list, "\n  - ") << std::endl;
            return 127;
        }
        if(!vm) {
            CPPLOG(ERR) << "Could not create vm for isa " << isa_opt << " and backend " << clim["backend"].as<std::string>() << std::endl;
            return 127;
        }
        if(clim.count("plugin")) {
            for(std::string const& opt_val : clim["plugin"].as<std::vector<std::string>>()) {
                std::string plugin_name = opt_val;
                std::string arg{""};
                std::size_t found = opt_val.find('=');
                if(found != std::string::npos) {
                    plugin_name = opt_val.substr(0, found);
                    arg = opt_val.substr(found + 1, opt_val.size());
                }
#if defined(WITH_PLUGINS)
                if(plugin_name == "ic") {
                    auto* ic_plugin = new iss::plugin::instruction_count(arg);
                    vm->register_plugin(*ic_plugin);
                    plugin_list.push_back(ic_plugin);
                } else if(plugin_name == "ce") {
                    auto* ce_plugin = new iss::plugin::cycle_estimate(arg);
                    vm->register_plugin(*ce_plugin);
                    plugin_list.push_back(ce_plugin);
                } else
#endif
                {
#if !defined(WIN32)
                    std::vector<char const*> a{};
                    if(arg.length())
                        a.push_back({arg.c_str()});
                    iss::plugin::loader l(plugin_name, {{"initPlugin"}});
                    auto* plugin = l.call_function<iss::vm_plugin*>("initPlugin", a.size(), a.data());
                    if(plugin) {
                        vm->register_plugin(*plugin);
                        plugin_list.push_back(plugin);
                    } else
#endif
                    {
                        CPPLOG(ERR) << "Unknown plugin name: " << plugin_name << ", valid names are 'ce', 'ic'" << std::endl;
                        return 127;
                    }
                }
            }
        }
        if(clim.count("disass")) {
            vm->setDisassEnabled(true);
            LOGGER(disass)::reporting_level() = logging::INFO;
            LOGGER(disass)::print_time() = false;
            auto file_name = clim["disass"].as<std::string>();
            if(file_name.length() > 0) {
                LOG_OUTPUT(disass)::stream() = fopen(file_name.c_str(), "w");
                LOGGER(disass)::print_severity() = false;
            }
        }
        uint64_t start_address = 0;
        if(clim.count("mem"))
            vm->get_arch()->load_file(clim["mem"].as<std::string>());
        if(clim.count("elf"))
            for(std::string input : clim["elf"].as<std::vector<std::string>>()) {
                auto start_addr = vm->get_arch()->load_file(input);
                if(start_addr.second)
                    start_address = start_addr.first;
                else {
                    LOG(ERR) << "Error occured while loading file " << input << std::endl;
                    return 1;
                }
            }
        for(std::string input : args) {
            auto start_addr = vm->get_arch()->load_file(input); // treat remaining arguments as elf files
            if(start_addr.second)
                start_address = start_addr.first;
            else {
                LOG(ERR) << "Error occured while loading file " << input << std::endl;
                return 1;
            }
        }
        if(clim.count("reset")) {
            auto str = clim["reset"].as<std::string>();
            start_address = str.find("0x") == 0 ? std::stoull(str.substr(2), nullptr, 16) : std::stoull(str, nullptr, 10);
        }
        vm->reset(start_address);
        auto limit = clim["ilimit"].as<uint64_t>();
        auto cond = iss::finish_cond_e::JUMP_TO_SELF;
        if(clim.count("flimit")) {
            cond = cond | iss::finish_cond_e::FCOUNT_LIMIT;
            limit = clim["flimit"].as<uint64_t>();
        } else {
            cond = cond | iss::finish_cond_e::ICOUNT_LIMIT;
        }
        res = vm->start(limit, dump, cond);

        auto instr_if = vm->get_arch()->get_instrumentation_if();
        // this assumes a single input file
        std::unordered_map<std::string, uint64_t> sym_table;
        if(args.empty())
            sym_table = instr_if->get_symbol_table(clim["elf"].as<std::vector<std::string>>()[0]);
        else
            sym_table = instr_if->get_symbol_table(args[0]);
        if(sym_table.find("begin_signature") != std::end(sym_table) && sym_table.find("end_signature") != std::end(sym_table)) {
            auto start_addr = sym_table["begin_signature"];
            auto end_addr = sym_table["end_signature"];
            std::array<uint8_t, 4> data;
            std::ofstream file;
            std::string filename = fmt::format("{}.signature", isa_opt);
            std::replace(std::begin(filename), std::end(filename), '|', '_');
            // default riscof requires this filename
            filename = "DUT-tgc.signature";
            file.open(filename, std::ios::out);
            if(!file.is_open()) {
                LOG(ERR) << "Error opening file " << filename << std::endl;
                return 1;
            }
            LOGGER(DEFAULT)::reporting_level() = logging::ERR;
            for(auto addr = start_addr; addr < end_addr; addr += data.size()) {
                vm->get_arch()->read(iss::address_type::PHYSICAL, iss::access_type::DEBUG_READ, 0 /*MEM*/, addr, data.size(),
                                     data.data()); // FIXME: get space from iss::arch::traits<ARCH>::mem_type_e::MEM

                // TODO : obey Target endianess
                uint32_t to_print = (data[3] << 24) + (data[2] << 16) + (data[1] << 8) + data[0];
                file << std::hex << fmt::format("{:08x}", to_print) << std::dec << std::endl;
            }
        }
    } catch(std::exception& e) {
        CPPLOG(ERR) << "Unhandled Exception reached the top of main: " << e.what() << ", application will now exit" << std::endl;
        res = 2;
    }
    // cleanup to let plugins report if needed
    for(auto* p : plugin_list) {
        delete p;
    }
    return res;
}
