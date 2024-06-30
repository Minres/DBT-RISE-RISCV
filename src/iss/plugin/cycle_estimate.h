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

#ifndef _ISS_PLUGIN_CYCLE_ESTIMATE_H_
#define _ISS_PLUGIN_CYCLE_ESTIMATE_H_

#include "iss/instrumentation_if.h"
#include "iss/vm_plugin.h"
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace iss {

namespace plugin {

class cycle_estimate : public vm_plugin {
    struct instr_desc {
        size_t size{0};
        bool is_branch{false};
        unsigned not_taken{1};
        unsigned taken{1};
        std::function<unsigned(uint64_t)> f;
    };

public:
    cycle_estimate() = delete;

    cycle_estimate(const cycle_estimate&) = delete;

    cycle_estimate(const cycle_estimate&&) = delete;

    cycle_estimate(std::string const& config_file_name);

    virtual ~cycle_estimate();

    cycle_estimate& operator=(const cycle_estimate&) = delete;

    cycle_estimate& operator=(const cycle_estimate&&) = delete;

    bool registration(const char* const version, vm_if& arch) override;

    sync_type get_sync() override { return ALL_SYNC; };

    void callback(instr_info_t instr_info) override;

private:
    iss::instrumentation_if* instr_if{nullptr};
    uint32_t* reg_base_ptr{nullptr};
    instr_desc illegal_desc{};
    std::vector<instr_desc> delays;
    unsigned current_delay{0};
    struct pair_hash {
        size_t operator()(const std::pair<uint64_t, uint64_t>& p) const {
            std::hash<uint64_t> hash;
            return hash(p.first) + hash(p.second);
        }
    };
    std::unordered_map<std::pair<uint64_t, uint64_t>, uint64_t, pair_hash> blocks;
    std::string config_file_name;
};
} // namespace plugin
} // namespace iss

#endif /* _ISS_PLUGIN_CYCLE_ESTIMATE_H_ */
