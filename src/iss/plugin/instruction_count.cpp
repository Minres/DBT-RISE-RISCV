/*******************************************************************************
 * Copyright (C) 2017 - 2023 MINRES Technologies GmbH
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

#include "instruction_count.h"
#include <iss/instrumentation_if.h>
#include <yaml-cpp/yaml.h>

#include <fstream>
#include <iss/arch_if.h>
#include <util/logging.h>

iss::plugin::instruction_count::instruction_count(std::string config_file_name) {
    if(config_file_name.length() > 0) {
        std::ifstream is(config_file_name);
        if(is.is_open()) {
            try {
                auto root = YAML::LoadAll(is);
                if(root.size() != 1) {
                    CPPLOG(ERR) << "Too many rro nodes in YAML file " << config_file_name;
                }
                for(auto p : root[0]) {
                    auto isa_subset = p.first;
                    auto instructions = p.second;
                    for(auto const& instr : instructions) {
                        instr_delay res;
                        res.instr_name = instr.first.as<std::string>();
                        res.size = instr.second["encoding"].as<std::string>().size() - 2; // not counting 0b
                        auto delay = instr.second["delay"];
                        if(delay.IsSequence()) {
                            res.not_taken_delay = delay[0].as<uint64_t>();
                            res.taken_delay = delay[1].as<uint64_t>();
                        } else {
                            res.not_taken_delay = delay.as<uint64_t>();
                            res.taken_delay = res.not_taken_delay;
                        }
                        delays.push_back(std::move(res));
                    }
                }
                rep_counts.resize(delays.size());
            } catch(YAML::ParserException& e) {
                CPPLOG(ERR) << "Could not parse input file " << config_file_name << ", reason: " << e.what();
            }
        } else {
            CPPLOG(ERR) << "Could not open input file " << config_file_name;
        }
    }
}

iss::plugin::instruction_count::~instruction_count() {
    size_t idx = 0;
    for(auto it : delays) {
        if(rep_counts[idx] > 0 && it.instr_name.find("__" != 0))
            CPPLOG(INFO) << it.instr_name << ";" << rep_counts[idx];
        idx++;
    }
}

bool iss::plugin::instruction_count::registration(const char* const version, vm_if& vm) {
    auto instr_if = vm.get_arch()->get_instrumentation_if();
    if(!instr_if)
        return false;
    return true;
}

void iss::plugin::instruction_count::callback(instr_info_t instr_info) { rep_counts[instr_info.instr_id]++; }
