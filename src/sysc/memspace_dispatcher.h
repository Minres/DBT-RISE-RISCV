/*******************************************************************************
 * Copyright (C) 2025 MINRES Technologies GmbH
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

#ifndef _SYSC_MEMSPACE_DISPATCHER_H_
#define _SYSC_MEMSPACE_DISPATCHER_H_
#include "scc/report.h"
#include "scc/utilities.h"
#include "tlm/scc/initiator_mixin.h"
#include "tlm/scc/target_mixin.h"
#include <cstddef>
#include <cstdint>

#include <sysc/kernel/sc_module.h>
#include <sysc/memspace_extension.h>
#include <sysc/utils/sc_report.h>
#include <tlm>
#include <tlm_core/tlm_2/tlm_generic_payload/tlm_gp.h>
#include <unordered_map>

namespace sysc {
namespace memspace {
template <typename MEMSPACE = common, unsigned BUSWIDTH = scc::LT, typename TARGET_SOCKET_TYPE = tlm::tlm_target_socket<BUSWIDTH>>
struct memspace_dispatcher : sc_core::sc_module {
    using intor_sckt = tlm::scc::initiator_mixin<tlm::tlm_initiator_socket<BUSWIDTH>>;
    using target_sckt = tlm::scc::target_mixin<TARGET_SOCKET_TYPE>;

    target_sckt target;
    sc_core::sc_vector<intor_sckt> initiator;
    memspace_dispatcher(const sc_core::sc_module_name& nm, size_t slave_cnt = 1)
    : sc_module(nm)
    , initiator("intor", slave_cnt) {
        sc_assert(slave_cnt > 0 && "memspace_dispatcher requires at least a single downstream module");
        target.register_b_transport(
            [this](tlm::tlm_generic_payload& trans, sc_core::sc_time& delay) { mapped_intor(trans)->b_transport(trans, delay); });
        target.register_get_direct_mem_ptr([this](tlm::tlm_generic_payload& trans, tlm::tlm_dmi& dmi_data) {
            return mapped_intor(trans, true)->get_direct_mem_ptr(trans, dmi_data);
        });
        target.register_transport_dbg([this](tlm::tlm_generic_payload& trans) { return mapped_intor(trans)->transport_dbg(trans); });
        for(size_t i = 0; i < initiator.size(); i++) {
            initiator[i].register_invalidate_direct_mem_ptr([this](::sc_dt::uint64 start_range, ::sc_dt::uint64 end_range) {
                target->invalidate_direct_mem_ptr(start_range, end_range);
            });
        }
    };
    template <typename TYPE> void bind_target(TYPE& socket, size_t port_idx) { initiator[port_idx].bind(socket); };
    void map_space_to_port(MEMSPACE space_id, unsigned port_idx) {
        sc_assert(port_idx < initiator.size() && "Targeted port index is higher than registered initiator sockets");
        if(space_mapping.count(space_id))
            SCCDEBUG(SCMOD) << "remapping of space_id " << static_cast<unsigned>(space_id);
        space_mapping[space_id] = port_idx;
    };
    void set_default_port(unsigned port_idx) {
        sc_assert(port_idx < initiator.size() && "Targeted port index is higher than registered initiator sockets");
        default_port = port_idx;
    };

private:
    std::unordered_map<MEMSPACE, size_t> space_mapping{};
    size_t default_port = 0;
    intor_sckt& mapped_intor(const tlm::tlm_generic_payload& trans, bool dmi = false) {
        if(auto* ext = trans.get_extension<tlm_memspace_extension<MEMSPACE>>()) {
            auto it = space_mapping.find(ext->get_space());
            if(it != space_mapping.end()) {
#ifndef NDEBUG
                auto prefix = dmi ? "DMI " : "";
                SCCTRACE(SCMOD) << "Sending " << prefix << "transaction to initiator " << it->second;
#endif
                return initiator[it->second];
            }
        }
#ifndef NDEBUG
        auto prefix = dmi ? "DMI " : "";
        SCCTRACE(SCMOD) << "Sending " << prefix << "transaction to default initiator " << default_port;
#endif
        return initiator[default_port];
    }
};
} // namespace memspace
} // namespace sysc

#endif /*_SYSC_MEMSPACE_DISPATCHER_H_*/