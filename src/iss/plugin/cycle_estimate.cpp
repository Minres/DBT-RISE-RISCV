/*******************************************************************************
 * Copyright (C) 2017 - 2023, MINRES Technologies GmbH
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

#include "cycle_estimate.h"
#include <iss/plugin/calculator.h>
#include <yaml-cpp/yaml.h>

#include <fstream>
#include <iss/arch_if.h>
#include <util/logging.h>

using namespace std;

iss::plugin::cycle_estimate::cycle_estimate(string const& config_file_name)
: instr_if(nullptr)
, config_file_name(config_file_name) {}

iss::plugin::cycle_estimate::~cycle_estimate() = default;

bool iss::plugin::cycle_estimate::registration(const char* const version, vm_if& vm) {
    instr_if = vm.get_arch()->get_instrumentation_if();
    assert(instr_if && "No instrumentation interface available but callback executed");
    reg_base_ptr = reinterpret_cast<uint32_t*>(vm.get_arch()->get_regs_base_ptr());
    if(!instr_if)
        return false;
    const string core_name = instr_if->core_type_name();
    if(config_file_name.length() > 0) {
        std::ifstream is(config_file_name);
        if(is.is_open()) {
            try {
                auto root = YAML::LoadAll(is);
                if(root.size() != 1) {
                    CPPLOG(ERR) << "Too many root nodes in YAML file " << config_file_name;
                }
                for(auto p : root[0]) {
                    auto isa_subset = p.first;
                    auto instructions = p.second;
                    for(auto const& instr : instructions) {
                        auto idx = instr.second["index"].as<unsigned>();
                        if(delays.size() <= idx)
                            delays.resize(idx + 1);
                        auto& res = delays[idx];
                        res.is_branch = instr.second["branch"].as<bool>();
                        auto delay = instr.second["delay"];
                        if(delay.IsSequence()) {
                            res.not_taken = delay[0].as<uint64_t>();
                            res.taken = delay[1].as<uint64_t>();
                        } else {
                            try {
                                res.not_taken = delay.as<uint64_t>();
                                res.taken = res.not_taken;
                            } catch(const YAML::BadConversion& e) {
                                res.f = iss::plugin::calculator(reg_base_ptr, delay.as<std::string>());
                            }
                        }
                    }
                }
            } catch(YAML::ParserException& e) {
                CPPLOG(ERR) << "Could not parse input file " << config_file_name << ", reason: " << e.what();
                return false;
            }
        } else {
            CPPLOG(ERR) << "Could not open input file " << config_file_name;
            return false;
        }
    }
    return true;
}

void iss::plugin::cycle_estimate::callback(instr_info_t instr_info) {
    size_t instr_id = instr_info.instr_id;
    auto& entry = instr_id < delays.size() ? delays[instr_id] : illegal_desc;
    if(instr_info.phase_id == PRE_SYNC) {
        if(entry.f)
            current_delay = entry.f(instr_if->get_instr_word());
    } else {
        if(!entry.f)
            current_delay = instr_if->is_branch_taken() ? entry.taken : entry.not_taken;
        if(current_delay > 1)
            instr_if->update_last_instr_cycles(current_delay);
        current_delay = 1;
    }
}
