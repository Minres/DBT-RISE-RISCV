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

#include "iss/plugin/instruction_count.h"
#include "iss/instrumentation_if.h"

#include <iss/arch_if.h>
#include <util/logging.h>
#include <fstream>

iss::plugin::instruction_count::instruction_count(std::string config_file_name) {
    if (config_file_name.length() > 0) {
        std::ifstream is(config_file_name);
        if (is.is_open()) {
            try {
                is >> root;
            } catch (Json::RuntimeError &e) {
                LOG(ERR) << "Could not parse input file " << config_file_name << ", reason: " << e.what();
            }
        } else {
            LOG(ERR) << "Could not open input file " << config_file_name;
        }
    }
}

iss::plugin::instruction_count::~instruction_count() {
	size_t idx=0;
	for(auto it:delays){
		if(rep_counts[idx]>0)
			LOG(INFO)<<it.instr_name<<";"<<rep_counts[idx];
		idx++;
	}
}

bool iss::plugin::instruction_count::registration(const char* const version, vm_if& vm) {
    auto instr_if = vm.get_arch()->get_instrumentation_if();
    if(!instr_if) return false;
	const std::string  core_name = instr_if->core_type_name();
    Json::Value &val = root[core_name];
    if(!val.isNull() && val.isArray()){
    	delays.reserve(val.size());
    	for(auto it:val){
    		auto name = it["name"];
    		auto size = it["size"];
    		auto delay = it["delay"];
    		if(!name.isString() || !size.isUInt() || !(delay.isUInt() || delay.isArray())) throw std::runtime_error("JSON parse error");
    		if(delay.isUInt()){
				const instr_delay entry{name.asCString(), size.asUInt(), delay.asUInt(), 0};
				delays.push_back(entry);
    		} else {
				const instr_delay entry{name.asCString(), size.asUInt(), delay[0].asUInt(), delay[1].asUInt()};
				delays.push_back(entry);
    		}
    	}
    	rep_counts.resize(delays.size());
    } else {
        LOG(ERR)<<"plugin instruction_count: could not find an entry for "<<core_name<<" in JSON file"<<std::endl;
    }
	return true;
}

void iss::plugin::instruction_count::callback(instr_info_t instr_info, exec_info const&) {
	rep_counts[instr_info.instr_id]++;
}
