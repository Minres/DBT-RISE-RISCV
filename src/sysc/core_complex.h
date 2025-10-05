/*******************************************************************************
 * Copyright (C) 2017-2021 MINRES Technologies GmbH
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

#ifndef _SYSC_CORE_COMPLEX_H_
#define _SYSC_CORE_COMPLEX_H_

#include "core_complex_if.h"
#include "sc2core_if.h"
#include <iss/debugger/target_adapter_if.h>
#include <iss/debugger_if.h>
#include <iss/vm_if.h>
#include <scc/signal_opt_ports.h>
#include <scc/tick2time.h>
#include <scc/traceable.h>
#include <scc/utilities.h>
#include <tlm/scc/initiator_mixin.h>
#include <tlm/scc/scv/tlm_rec_initiator_socket.h>
#ifdef CWR_SYSTEMC
#include <scmlinc/scml_property.h>
#else
#include <cci_configuration>
#ifndef SC_SIGNAL_IF
#include <tlm/scc/signal_target_mixin.h>
#define USE_TLM_SIGNAL
#endif
#endif
#include <memory>
#include <tlm>
#include <tlm_utils/tlm_quantumkeeper.h>
#include <util/range_lut.h>

namespace iss {
class vm_plugin;
}
namespace sysc {

class tlm_dmi_ext : public tlm::tlm_dmi {
public:
    bool operator==(const tlm_dmi_ext& o) const {
        return this->get_granted_access() == o.get_granted_access() && this->get_start_address() == o.get_start_address() &&
               this->get_end_address() == o.get_end_address();
    }

    bool operator!=(const tlm_dmi_ext& o) const { return !operator==(o); }
};

namespace riscv {
#ifdef USE_TLM_SIGNAL
using irq_signal_t = tlm::scc::tlm_signal_bool_opt_in;
#else
using irq_signal_t = sc_core::sc_in<bool>;
#endif

template <unsigned int BUSWIDTH = scc::LT> class core_complex : public sc_core::sc_module, public scc::traceable, public core_complex_if {
public:
    tlm::scc::initiator_mixin<tlm::tlm_initiator_socket<BUSWIDTH>> ibus{"ibus"};

    tlm::scc::initiator_mixin<tlm::tlm_initiator_socket<BUSWIDTH>> dbus{"dbus"};

    sc_core::sc_in<bool> rst_i{"rst_i"};

    irq_signal_t ext_irq_i{"ext_irq_i"};

    irq_signal_t timer_irq_i{"timer_irq_i"};

    irq_signal_t sw_irq_i{"sw_irq_i"};

    sc_core::sc_vector<irq_signal_t> local_irq_i{"local_irq_i", 16};

#ifndef CWR_SYSTEMC
    sc_core::sc_in<sc_core::sc_time> clk_i{"clk_i"};

    cci::cci_param<std::string> elf_file{"elf_file", ""};

    cci::cci_param<bool> enable_disass{"enable_disass", false};

    cci::cci_param<bool> disable_dmi{"disable_dmi", false};

    cci::cci_param<uint64_t> reset_address{"reset_address", 0ULL};

    cci::cci_param<std::string> core_type{"core_type", "rv32imac_m"};

    cci::cci_param<std::string> backend{"backend", "interp"};

    cci::cci_param<unsigned short> gdb_server_port{"gdb_server_port", 0};

    cci::cci_param<bool> dump_ir{"dump_ir", false};

    cci::cci_param<uint32_t> mhartid{"mhartid", 0};

    cci::cci_param<uint32_t> local_irq_num{"local_irq_num", 0};

    cci::cci_param<std::string> plugins{"plugins", ""};

    core_complex(sc_core::sc_module_name const& name);

#else
    sc_core::sc_in<bool> clk_i{"clk_i"};

    scml_property<std::string> elf_file{"elf_file", ""};

    scml_property<bool> enable_disass{"enable_disass", false};

    scml_property<bool> disable_dmi{"disable_dmi", false};

    scml_property<unsigned long long> reset_address{"reset_address", 0ULL};

    scml_property<std::string> core_type{"core_type", "rv32imac"};

    scml_property<std::string> backend{"backend", "interp"};

    scml_property<unsigned> gdb_server_port{"gdb_server_port", 0};

    scml_property<bool> dump_ir{"dump_ir", false};

    scml_property<uint32_t> mhartid{"mhartid", 0};

    scml_property<std::string> plugins{"plugins", ""};

    core_complex(sc_core::sc_module_name const& name)
    : sc_module(name)
    , local_irq_i{"local_irq_i", 16}
    , elf_file{"elf_file", ""}
    , enable_disass{"enable_disass", false}
    , reset_address{"reset_address", 0ULL}
    , core_type{"core_type", "rv32imac"}
    , backend{"backend", "interp"}
    , gdb_server_port{"gdb_server_port", 0}
    , dump_ir{"dump_ir", false}
    , mhartid{"mhartid", 0}
    , plugins{"plugins", ""}
    , fetch_lut(tlm_dmi_ext())
    , read_lut(tlm_dmi_ext())
    , write_lut(tlm_dmi_ext()) {
        init();
    }

#endif

    ~core_complex();

    unsigned get_last_bus_cycles() override {
        auto mem_incr = std::max(ibus_inc, dbus_inc);
        ibus_inc = dbus_inc = 0;
        return mem_incr > 1 ? mem_incr : 1;
    }

    void sync(uint64_t cycle) override {
        auto core_inc = curr_clk * (cycle - last_sync_cycle);
        quantum_keeper.inc(core_inc);
        if(quantum_keeper.need_sync()) {
            wait(quantum_keeper.get_local_time());
            quantum_keeper.reset();
        }
        last_sync_cycle = cycle;
    }

    bool read_mem(uint64_t addr, unsigned length, uint8_t* const data, bool is_fetch) override;

    bool write_mem(uint64_t addr, unsigned length, const uint8_t* const data) override;

    bool read_mem_dbg(uint64_t addr, unsigned length, uint8_t* const data) override;

    bool write_mem_dbg(uint64_t addr, unsigned length, const uint8_t* const data) override;

    void trace(sc_core::sc_trace_file* trf) const override;

    bool disass_output(uint64_t pc, const std::string instr) override;

    void set_clock_period(sc_core::sc_time period);

    char const* hier_name() override { return name(); }

    void reset(uint64_t addr);

    inline void start(bool dump = false);

    inline std::pair<uint64_t, bool> load_file(std::string const& name);

protected:
    void create_cpu(std::string const& type, std::string const& backend, unsigned gdb_port, uint32_t hart_id);
    int cmd_sysc(int argc, char* argv[], iss::debugger::out_func, iss::debugger::data_func, iss::debugger::target_adapter_if*);

    void before_end_of_elaboration() override;
    void start_of_simulation() override;
    void forward();
    void run();
    void rst_cb();
#ifndef USE_TLM_SIGNAL
    void sw_irq_cb();
    void timer_irq_cb();
    void ext_irq_cb();
    void local_irq_cb();
#endif
    uint64_t last_sync_cycle = 0;
    util::range_lut<tlm_dmi_ext> fetch_lut, read_lut, write_lut;
    tlm_utils::tlm_quantumkeeper quantum_keeper;
    std::vector<uint8_t> write_buf;
    sc_core::sc_signal<sc_core::sc_time> curr_clk;
    uint64_t ibus_inc{0}, dbus_inc{0};
    std::unique_ptr<sc2core_if> core;
    std::unique_ptr<iss::vm_if> vm;
    iss::debugger::target_adapter_if* tgt_adapter{nullptr};

    struct {
        //! transaction recording database
        SCVNS scv_tr_db* m_db{nullptr};
        //! blocking transaction recording stream handle
        SCVNS scv_tr_stream* stream_handle{nullptr};
        //! transaction generator handle for blocking transactions
        SCVNS scv_tr_generator<SCVNS _scv_tr_generator_default_data, SCVNS _scv_tr_generator_default_data>* instr_tr_handle{nullptr};
        SCVNS scv_tr_handle tr_handle;
    } trc;
    std::unique_ptr<scc::tick2time> t2t;

private:
    void init();
    std::vector<iss::vm_plugin*> plugin_list;
};
} // namespace riscv
} /* namespace sysc */

#endif /* _SYSC_CORE_COMPLEX_H_ */
