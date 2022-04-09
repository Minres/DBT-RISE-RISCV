/*******************************************************************************
 * Copyright (C) 2017, 2018, MINRES Technologies GmbH
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

#ifndef _ISS_PLUGIN_INSTRUCTION_COUNTER_H_
#define _ISS_PLUGIN_INSTRUCTION_COUNTER_H_

#include <iss/vm_plugin.h>
#include <json/json.h>
#include <string>

namespace iss {
namespace plugin {

class instruction_count : public iss::vm_plugin {
    struct instr_delay {
        std::string instr_name;
        size_t size;
        size_t not_taken_delay;
        size_t taken_delay;
    };

public:
    instruction_count() = delete;

    instruction_count(const instruction_count &) = delete;

    instruction_count(const instruction_count &&) = delete;

    instruction_count(std::string config_file_name);

    virtual ~instruction_count();

    instruction_count &operator=(const instruction_count &) = delete;

    instruction_count &operator=(const instruction_count &&) = delete;

    bool registration(const char *const version, vm_if &arch) override;

    sync_type get_sync() override { return POST_SYNC; };

    void callback(instr_info_t instr_info, exec_info const&) override;

private:
    Json::Value root;
    std::vector<instr_delay> delays;
    std::vector<uint64_t> rep_counts;
};
}
}

#endif /* _ISS_PLUGIN_INSTRUCTION_COUNTER_H_ */
