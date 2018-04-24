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

#include "iss/plugin/cycle_estimate.h"

#include <iss/arch_if.h>
#include <util/logging.h>
#include <fstream>

iss::plugin::cycle_estimate::cycle_estimate(std::string config_file_name)
: arch_instr(nullptr)
{
    if (config_file_name.length() > 0) {
        std::ifstream is(config_file_name);
        if (is.is_open()) {
            try {
                is >> root;
            } catch (Json::RuntimeError &e) {
                LOG(ERROR) << "Could not parse input file " << config_file_name << ", reason: " << e.what();
            }
        } else {
            LOG(ERROR) << "Could not open input file " << config_file_name;
        }
    }
}

iss::plugin::cycle_estimate::~cycle_estimate() {
}

bool iss::plugin::cycle_estimate::registration(const char* const version, vm_if& vm) {
	arch_instr = vm.get_arch()->get_instrumentation_if();
	if(!arch_instr) return false;
	const std::string  core_name = arch_instr->core_type_name();
    Json::Value &val = root[core_name];
    if(!val.isNull() && val.isArray()){
    	delays.reserve(val.size());
    	for(auto it:val){
    		auto name = it["name"];
    		auto size = it["size"];
    		auto delay = it["delay"];
    		if(!name.isString() || !size.isUInt() || !(delay.isUInt() || delay.isArray())) throw std::runtime_error("JSON parse error");
    		if(delay.isUInt()){
				delays.push_back(instr_desc{size.asUInt(), delay.asUInt(), 0});
    		} else {
				delays.push_back(instr_desc{size.asUInt(), delay[0].asUInt(), delay[1].asUInt()});
    		}
    	}
    } else {
        LOG(ERROR)<<"plugin cycle_estimate: could not find an entry for "<<core_name<<" in JSON file"<<std::endl;
    }
	return true;

}

void iss::plugin::cycle_estimate::callback(instr_info_t instr_info) {
    assert(arch_instr && "No instrumentation interface available but callback executed");
	auto entry = delays[instr_info.instr_id];
	bool taken = (arch_instr->get_next_pc()-arch_instr->get_pc()) != (entry.size/8);
    uint32_t delay = taken ? entry.taken : entry.not_taken;
    if(delay>1) arch_instr->set_curr_instr_cycles(delay);
}
