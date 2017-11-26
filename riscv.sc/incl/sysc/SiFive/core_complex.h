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
//       eyck@minres.com - initial implementation
//
//
////////////////////////////////////////////////////////////////////////////////

#ifndef _SYSC_SIFIVE_FE310_H_
#define _SYSC_SIFIVE_FE310_H_

#include "scc/utilities.h"
#include <tlm>
#include <tlm_utils/tlm_quantumkeeper.h>
#include <util/range_lut.h>
#include "scc/ext_attribute.h"
#include "scv4tlm/tlm_rec_initiator_socket.h"
#include "scc/initiator_mixin.h"
#include "scc/traceable.h"

class scv_tr_db;
class scv_tr_stream;
struct _scv_tr_generator_default_data;
template < class T_begin, class T_end> class scv_tr_generator;

namespace iss {
class vm_if;
namespace arch {
template <typename BASE> class riscv_hart_msu_vp;
}
namespace debugger {
class target_adapter_if;
}
}

namespace sysc {

class tlm_dmi_ext : public tlm::tlm_dmi {
public:
    bool operator==(const tlm_dmi_ext &o) const {
        return this->get_granted_access() == o.get_granted_access() &&
               this->get_start_address() == o.get_start_address() && this->get_end_address() == o.get_end_address();
    }

    bool operator!=(const tlm_dmi_ext &o) const { return !operator==(o); }
};

namespace SiFive {
class core_wrapper;

class core_complex : public sc_core::sc_module, public scc::traceable {
public:
    SC_HAS_PROCESS(core_complex);

    scc::initiator_mixin<scv4tlm::tlm_rec_initiator_socket<32>> initiator;

    sc_core::sc_in<sc_core::sc_time> clk_i;

    sc_core::sc_in<bool> rst_i;

    sc_core::sc_in<bool> global_irq_i;

    sc_core::sc_in<bool> timer_irq_i;

    sc_core::sc_in<bool> sw_irq_i;

    sc_core::sc_vector<sc_core::sc_in<bool>> local_irq_i;

    scc::ext_attribute<std::string> elf_file;

    scc::ext_attribute<bool> enable_disass;

    scc::ext_attribute<uint64_t> reset_address;

    scc::ext_attribute<unsigned short> gdb_server_port;

    scc::ext_attribute<bool> dump_ir;

    core_complex(sc_core::sc_module_name name);

    ~core_complex();

    inline void sync() {
        quantum_keeper.inc(curr_clk);
        if (quantum_keeper.need_sync()) {
            wait(quantum_keeper.get_local_time());
            quantum_keeper.reset();
        }
    }

    bool read_mem(uint64_t addr, unsigned length, uint8_t *const data, bool is_fetch);

    bool write_mem(uint64_t addr, unsigned length, const uint8_t *const data);

    bool read_mem_dbg(uint64_t addr, unsigned length, uint8_t *const data);

    bool write_mem_dbg(uint64_t addr, unsigned length, const uint8_t *const data);

    void trace(sc_core::sc_trace_file *trf) override;

    void disass_output(uint64_t pc, const std::string instr);
protected:
    void before_end_of_elaboration();
    void start_of_simulation();
    void run();
    void clk_cb();
    void sw_irq_cb();
    void timer_irq_cb();
    void global_irq_cb();
    util::range_lut<tlm_dmi_ext> read_lut, write_lut;
    tlm_utils::tlm_quantumkeeper quantum_keeper;
    std::vector<uint8_t> write_buf;
    std::unique_ptr<core_wrapper> cpu;
    std::unique_ptr<iss::vm_if> vm;
    sc_core::sc_time curr_clk;
    iss::debugger::target_adapter_if* tgt_adapter;
#ifdef WITH_SCV
    //! transaction recording database
    scv_tr_db *m_db;
    //! blocking transaction recording stream handle
    scv_tr_stream *stream_handle;
    //! transaction generator handle for blocking transactions
    scv_tr_generator<_scv_tr_generator_default_data,_scv_tr_generator_default_data> *instr_tr_handle;
    scv_tr_generator<uint64_t,_scv_tr_generator_default_data> *fetch_tr_handle;
    scv_tr_handle tr_handle;
#endif

};

} /* namespace SiFive */
} /* namespace sysc */

#endif /* _SYSC_SIFIVE_FE310_H_ */
