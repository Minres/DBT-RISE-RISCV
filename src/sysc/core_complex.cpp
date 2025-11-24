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

// clang-format off
#include "core_complex.h"
#include <iss/debugger/gdb_session.h>
#include <iss/debugger/encoderdecoder.h>
#include <iss/debugger/server.h>
#include <iss/debugger/target_adapter_if.h>
#include <iss/iss.h>
#include <iss/vm_types.h>
#include "iss_factory.h"
#include "tlm/scc/quantum_keeper.h"
#include <memory>
#include <sstream>
#include <sysc/kernel/sc_simcontext.h>
#include <tlm/scc/tlm_signal_gp.h>
#ifndef WIN32
#include <iss/plugin/loader.h>
#endif
#include <scc/report.h>
#include <util/ities.h>
#include <array>
#include <iss/plugin/cycle_estimate.h>
#include <iss/plugin/instruction_count.h>
#include <util/ities.h>
// clang-format on

#ifdef HAS_SCV
#include <scv.h>
#else
#include <scv-tr.h>
using namespace scv_tr;
#endif

#ifndef CWR_SYSTEMC
#define GET_PROP_VALUE(P) P.get_value()
#else
#define GET_PROP_VALUE(P) P.getValue()
#endif

#ifdef _MSC_VER
// not #if defined(_WIN32) || defined(_WIN64) because we have strncasecmp in mingw
#define strncasecmp _strnicmp
#define strcasecmp _stricmp
#endif

namespace sysc {
namespace riscv {
using namespace std;
using namespace iss;
using namespace logging;
using namespace sc_core;

namespace {
iss::debugger::encoder_decoder encdec;
std::array<const char, 4> lvl = {{'U', 'S', 'H', 'M'}};
} // namespace

template <unsigned int BUSWIDTH, typename QK>
int core_complex<BUSWIDTH, QK>::cmd_sysc(int argc, char* argv[], debugger::out_func of, debugger::data_func df,
                                         debugger::target_adapter_if* tgt_adapter) {
    if(argc > 1) {
        if(strcasecmp(argv[1], "print_time") == 0) {
            std::string t = sc_time_stamp().to_string();
            of(t.c_str());
            std::array<char, 64> buf;
            encdec.enc_string(t.c_str(), buf.data(), 63);
            df(buf.data());
            return Ok;
        } else if(strcasecmp(argv[1], "break") == 0) {
            sc_time t;
            if(argc == 4) {
                t = scc::parse_from_string(argv[2], argv[3]);
            } else if(argc == 3) {
                t = scc::parse_from_string(argv[2]);
            } else
                return Err;
            // no check needed as it is only called if debug server is active
            tgt_adapter->add_break_condition([t]() -> unsigned {
                SCCTRACE() << "Checking condition at " << sc_time_stamp();
                return sc_time_stamp() >= t ? std::numeric_limits<unsigned>::max() : 0;
            });
            return Ok;
        }
        return Err;
    }
    return Err;
}

template <unsigned int BUSWIDTH, typename QK> void core_complex<BUSWIDTH, QK>::reset(uint64_t addr) { vm->reset(addr); }
template <unsigned int BUSWIDTH, typename QK>
inline std::pair<uint64_t, bool> core_complex<BUSWIDTH, QK>::load_file(std::string const& name) {
    iss::arch_if* cc = vm->get_arch();
    return cc->load_file(name);
};

template <unsigned int BUSWIDTH, typename QK>
void core_complex<BUSWIDTH, QK>::create_cpu(std::string const& type, std::string const& backend, unsigned gdb_port, uint32_t hart_id) {
    auto& f = sysc::iss_factory::instance();
    if(type.size() == 0 || type == "?") {
        std::unordered_map<std::string, std::vector<std::string>> core_by_backend;
        for(auto& e : f.get_names()) {
            auto p = e.find(':');
            assert(p != std::string::npos);
            core_by_backend[e.substr(p + 1)].push_back(e.substr(0, p));
        }
        std::ostringstream os;
        os << "Available implementations\n";
        os << "=========================\n";
        for(auto& e : core_by_backend) {
            std::sort(std::begin(e.second), std::end(e.second));
            if(os.str().size())
                os << "\n";
            os << "  backend " << e.first << ":\n  - " << util::join(e.second, "\n  - ");
        }
        SCCINFO(SCMOD) << "\n" << os.str();
        sc_core::sc_stop();
    } else if(type.find(':') == std::string::npos) {
        std::tie(core, vm) = f.create(type + ":" + backend, gdb_port, this);
    } else {
        auto base_isa = type.substr(0, 5);
        if(base_isa == "tgc5d" || base_isa == "tgc5e") {
            std::tie(core, vm) = f.create(type + "_clic_pmp:" + backend, gdb_port, this);
        } else {
            std::tie(core, vm) = f.create(type + ":" + backend, gdb_port, this);
        }
    }
    if(!core) {
        if(type != "?")
            SCCFATAL() << "Could not create cpu for isa " << type << " and backend " << backend;
    } else if(!vm) {
        if(type != "?")
            SCCFATAL() << "Could not create vm for isa " << type << " and backend " << backend;
    } else {
        core->set_hartid(hart_id);
        auto* srv = debugger::server<debugger::gdb_session>::get();
        if(srv)
            tgt_adapter = srv->get_target(0); // FIXME: add core_id
        if(tgt_adapter)
            tgt_adapter->add_custom_command({"sysc",
                                             [this](int argc, char* argv[], debugger::out_func of, debugger::data_func df) -> int {
                                                 return cmd_sysc(argc, argv, of, df, tgt_adapter);
                                             },
                                             "SystemC sub-commands: break <time>, print_time"});
    }
}

#ifndef CWR_SYSTEMC
template <unsigned int BUSWIDTH, typename QK>
core_complex<BUSWIDTH, QK>::core_complex(sc_module_name const& name)
: sc_module(name)
, fetch_lut(tlm_dmi_ext())
, read_lut(tlm_dmi_ext())
, write_lut(tlm_dmi_ext()) {
    init();
}
#endif

template <unsigned int BUSWIDTH, typename QK> void core_complex<BUSWIDTH, QK>::init() {
    ibus.register_invalidate_direct_mem_ptr([this](uint64_t start, uint64_t end) -> void {
        auto lut_entry = fetch_lut.getEntry(start);
        if(lut_entry.get_granted_access() != tlm::tlm_dmi::DMI_ACCESS_NONE && end <= lut_entry.get_end_address() + 1) {
            fetch_lut.removeEntry(lut_entry);
        }
    });
    dbus.register_invalidate_direct_mem_ptr([this](uint64_t start, uint64_t end) -> void {
        auto lut_entry = read_lut.getEntry(start);
        if(lut_entry.get_granted_access() != tlm::tlm_dmi::DMI_ACCESS_NONE && end <= lut_entry.get_end_address() + 1) {
            read_lut.removeEntry(lut_entry);
        }
        lut_entry = write_lut.getEntry(start);
        if(lut_entry.get_granted_access() != tlm::tlm_dmi::DMI_ACCESS_NONE && end <= lut_entry.get_end_address() + 1) {
            write_lut.removeEntry(lut_entry);
        }
    });

    SC_HAS_PROCESS(this_class); // NOLINT
    SC_THREAD(run);
    SC_METHOD(rst_cb);
    sensitive << rst_i;
#ifdef USE_TLM_SIGNAL
    sw_irq_i.register_nb_transport([this](tlm::scc::tlm_signal_gp<bool>& gp, tlm::tlm_phase& p, sc_core::sc_time& t) {
        cpu->local_irq(3, gp.get_value());
        return tlm::TLM_COMPLETED;
    });
    timer_irq_i.register_nb_transport([this](tlm::scc::tlm_signal_gp<bool>& gp, tlm::tlm_phase& p, sc_core::sc_time& t) {
        cpu->local_irq(7, gp.get_value());
        return tlm::TLM_COMPLETED;
    });
    ext_irq_i.register_nb_transport([this](tlm::scc::tlm_signal_gp<bool>& gp, tlm::tlm_phase& p, sc_core::sc_time& t) {
        cpu->local_irq(11, gp.get_value());
        return tlm::TLM_COMPLETED;
    });
    for(auto i = 0U; i < local_irq_i.size(); ++i)
        local_irq_i[i].register_nb_transport([this, i](tlm::scc::tlm_signal_gp<bool>& gp, tlm::tlm_phase& p, sc_core::sc_time& t) {
            cpu->local_irq(16 + i, gp.get_value());
            return tlm::TLM_COMPLETED;
        });
#else
    SC_METHOD(sw_irq_cb);
    sensitive << sw_irq_i;
    SC_METHOD(timer_irq_cb);
    sensitive << timer_irq_i;
    SC_METHOD(ext_irq_cb);
    sensitive << ext_irq_i;
    SC_METHOD(local_irq_cb);
    for(auto pin : local_irq_i)
        sensitive << pin;
#endif

    SC_METHOD(forward);
#ifndef CWR_SYSTEMC
    sensitive << clk_i;
#else
    sensitive << curr_clk;
    t2t.reset(new scc::tick2time{"t2t"});
    t2t->clk_i(clk_i);
    t2t->clk_o(curr_clk);
#endif
}

template <unsigned int BUSWIDTH, typename QK> core_complex<BUSWIDTH, QK>::~core_complex() {
    for(auto* p : plugin_list)
        delete p;
    if(post_run_stats.get_value()) {
        auto instr_if = vm->get_arch()->get_instrumentation_if();
        auto instrs = instr_if->get_instr_count();
        auto cycles = instr_if->get_total_cycles();
        SCCINFO(SCMOD) << "Ran " << instrs << " instructions in " << cycles << " cycles";
    }
}

template <unsigned int BUSWIDTH, typename QK> void core_complex<BUSWIDTH, QK>::trace(sc_trace_file* trf) const {}

template <unsigned int BUSWIDTH, typename QK> void core_complex<BUSWIDTH, QK>::before_end_of_elaboration() {
    auto& type = GET_PROP_VALUE(core_type);
    SCCDEBUG(SCMOD) << "instantiating core " << type << " with " << GET_PROP_VALUE(backend) << " backend";
    // cpu = scc::make_unique<core_wrapper>(this);
    create_cpu(type, GET_PROP_VALUE(backend), GET_PROP_VALUE(gdb_server_port), GET_PROP_VALUE(mhartid));
    if(type == "?")
        return;
#ifndef CWR_SYSTEMC
    if(!local_irq_num.is_default_value()) {
        core->set_irq_count(16 + local_irq_num);
    }
#endif
    sc_assert(vm);
    auto instr_trace = GET_PROP_VALUE(enable_instr_trace) ? trc.init(this->name()) : false;
    auto disass = GET_PROP_VALUE(enable_disass);
    if(disass)
        core->enable_disass(true);
    vm->setDisassEnabled(disass || instr_trace);
    if(GET_PROP_VALUE(plugins).length()) {
        auto p = util::split(GET_PROP_VALUE(plugins), ';');
        for(std::string const& opt_val : p) {
            std::string plugin_name = opt_val;
            std::string filename{"cycles.txt"};
            std::size_t found = opt_val.find('=');
            if(found != std::string::npos) {
                plugin_name = opt_val.substr(0, found);
                filename = opt_val.substr(found + 1, opt_val.size());
            }
            if(plugin_name == "ic") {
                auto* plugin = new iss::plugin::instruction_count(filename);
                vm->register_plugin(*plugin);
                plugin_list.push_back(plugin);
            } else if(plugin_name == "ce") {
                auto* plugin = new iss::plugin::cycle_estimate(filename);
                vm->register_plugin(*plugin);
                plugin_list.push_back(plugin);
            } else {
#ifndef WIN32
                std::array<char const*, 1> a{{filename.c_str()}};
                iss::plugin::loader l(plugin_name, {{"initPlugin"}});
                auto* plugin = l.call_function<iss::vm_plugin*>("initPlugin", a.size(), a.data());
                if(plugin) {
                    vm->register_plugin(*plugin);
                    plugin_list.push_back(plugin);
                } else
#endif
                    SCCERR(SCMOD) << "Unknown plugin '" << plugin_name << "' or plugin not found";
            }
        }
    }
}

template <unsigned int BUSWIDTH, typename QK> void core_complex<BUSWIDTH, QK>::start_of_simulation() {
    if(GET_PROP_VALUE(elf_file).size() > 0) {
        auto file_names = util::split(GET_PROP_VALUE(elf_file), ',');
        for(auto& s : file_names) {
            std::pair<uint64_t, bool> load_result = load_file(s);
            if(!std::get<1>(load_result)) {
                SCCWARN(SCMOD) << "Could not load FW file " << s;
            } else {
#ifndef CWR_SYSTEMC
                if(reset_address.is_default_value())
                    reset_address.set_value(load_result.first);
#else
                if(start_addr.second == true)
                    reset_address = start_addr.first;
#endif
            }
        }
    }
}

template <unsigned int BUSWIDTH, typename QK> void core_complex<BUSWIDTH, QK>::disass_output(uint64_t pc, std::string const& instr_str) {
    trc.disass_output(pc, instr_str, lvl[core->get_mode()], core->get_state());
}

template <unsigned int BUSWIDTH, typename QK> void core_complex<BUSWIDTH, QK>::forward() {
#ifndef CWR_SYSTEMC
    set_clock_period(clk_i.read());
#else
    set_clock_period(curr_clk.read());

#endif
}

template <unsigned int BUSWIDTH, typename QK> void core_complex<BUSWIDTH, QK>::set_clock_period(sc_core::sc_time period) {
    curr_clk = period;
    if(period == SC_ZERO_TIME)
        core->set_interrupt_execution(true);
}

template <unsigned int BUSWIDTH, typename QK> void core_complex<BUSWIDTH, QK>::rst_cb() {
    if(rst_i.read())
        core->set_interrupt_execution(true);
}

#ifndef USE_TLM_SIGNAL
template <unsigned int BUSWIDTH, typename QK> void core_complex<BUSWIDTH, QK>::sw_irq_cb() { core->local_irq(3, sw_irq_i.read()); }

template <unsigned int BUSWIDTH, typename QK> void core_complex<BUSWIDTH, QK>::timer_irq_cb() { core->local_irq(7, timer_irq_i.read()); }

template <unsigned int BUSWIDTH, typename QK> void core_complex<BUSWIDTH, QK>::ext_irq_cb() { core->local_irq(11, ext_irq_i.read()); }

template <unsigned int BUSWIDTH, typename QK> void core_complex<BUSWIDTH, QK>::local_irq_cb() {
    for(auto i = 0U; i < local_irq_i.size(); ++i) {
        if(local_irq_i[i].event()) {
            core->local_irq(16 + i, local_irq_i[i].read());
        }
    }
}
#endif

template <unsigned int BUSWIDTH, typename QK> void core_complex<BUSWIDTH, QK>::run() {
    wait(SC_ZERO_TIME); // separate from elaboration phase
    do {
        wait(SC_ZERO_TIME);
        if(rst_i.read()) {
            reset(GET_PROP_VALUE(reset_address));
            wait(rst_i.negedge_event());
        }
        while(curr_clk.read() == SC_ZERO_TIME) {
            wait(curr_clk.value_changed_event());
        }
        quantum_keeper.reset();
        core->set_interrupt_execution(false);
        run_iss();
    } while(!core->get_interrupt_execution());
    if(finish_evt_inuse)
        finish_evt.notify();
    else {
        if(sc_core::sc_get_simulator_status() != sc_core::SC_SIM_USER_STOP) // stop simulation
            sc_stop();
    }
}

template <unsigned int BUSWIDTH, typename QK> bool core_complex<BUSWIDTH, QK>::read_mem(addr_t addr, unsigned length, uint8_t* const data) {
    bool is_fetch = addr.space == std::numeric_limits<decltype(addr.space)>::max() ? true : false;
    auto& dmi_lut = is_fetch ? fetch_lut : read_lut;
    auto lut_entry = dmi_lut.getEntry(addr.val);
    if(lut_entry.get_granted_access() != tlm::tlm_dmi::DMI_ACCESS_NONE && (addr.val + length) <= (lut_entry.get_end_address() + 1)) {
        auto offset = addr.val - lut_entry.get_start_address();
        std::copy(lut_entry.get_dmi_ptr() + offset, lut_entry.get_dmi_ptr() + offset + length, data);
        if(is_fetch)
            ibus_inc += lut_entry.get_read_latency() / curr_clk;
        else
            dbus_inc += lut_entry.get_read_latency() / curr_clk;
#ifndef NDEBUG
        SCCTRACE(this->name()) << "[local time: " << quantum_keeper.get_local_time() << "]: finish dmi_read_mem(0x" << std::hex << addr.val
                               << ") : 0x"
                               << (length == 4   ? *(uint32_t*)data
                                   : length == 2 ? *(uint16_t*)data
                                                 : (unsigned)*data);
#endif
        return true;
    } else {
        tlm::tlm_generic_payload gp;
        gp.set_command(tlm::TLM_READ_COMMAND);
        gp.set_address(addr.val);
        gp.set_data_ptr(data);
        gp.set_data_length(length);
        gp.set_streaming_width(length);
        sc_time delay = quantum_keeper.get_local_time();
        auto pre_delay = delay;
        exec_b_transport(gp, delay, is_fetch);
        if(pre_delay > delay) {
            quantum_keeper.reset();
        } else {
            auto incr = (delay - quantum_keeper.get_local_time()) / curr_clk;
            if(is_fetch)
                ibus_inc += incr;
            else
                dbus_inc += incr;
        }
        SCCTRACE(this->name()) << "[local time: " << delay << "]: finish read_mem(0x" << std::hex << addr.val << ") : 0x"
                               << (length == 4   ? *(uint32_t*)data
                                   : length == 2 ? *(uint16_t*)data
                                                 : (unsigned)*data);
        if(gp.get_response_status() != tlm::TLM_OK_RESPONSE) {
            return false;
        }
        if(gp.is_dmi_allowed() && !GET_PROP_VALUE(disable_dmi)) {
            gp.set_command(tlm::TLM_READ_COMMAND);
            gp.set_address(addr.val);
            tlm_dmi_ext dmi_data;
            if(exec_get_direct_mem_ptr(gp, dmi_data)) {
                if(dmi_data.is_read_allowed() && (addr.val + length - 1) <= dmi_data.get_end_address())
                    dmi_lut.addEntry(dmi_data, dmi_data.get_start_address(), dmi_data.get_end_address() - dmi_data.get_start_address() + 1);
            }
        }
        return true;
    }
}

template <unsigned int BUSWIDTH, typename QK>
bool core_complex<BUSWIDTH, QK>::write_mem(addr_t addr, unsigned length, const uint8_t* const data) {
    auto lut_entry = write_lut.getEntry(addr.val);
    if(lut_entry.get_granted_access() != tlm::tlm_dmi::DMI_ACCESS_NONE && (addr.val + length) <= (lut_entry.get_end_address() + 1)) {
        auto offset = addr.val - lut_entry.get_start_address();
        std::copy(data, data + length, lut_entry.get_dmi_ptr() + offset);
        dbus_inc += lut_entry.get_write_latency() / curr_clk;
#ifndef NDEBUG
        SCCTRACE(this->name()) << "[local time: " << quantum_keeper.get_local_time() << "]: finish dmi_write_mem(0x" << std::hex << addr.val
                               << ") : 0x"
                               << (length == 4   ? *(uint32_t*)data
                                   : length == 2 ? *(uint16_t*)data
                                                 : (unsigned)*data);
#endif
        return true;
    } else {
        write_buf.resize(length);
        std::copy(data, data + length, write_buf.begin()); // need to copy as TLM does not guarantee data integrity
        tlm::tlm_generic_payload gp;
        gp.set_command(tlm::TLM_WRITE_COMMAND);
        gp.set_address(addr.val);
        gp.set_data_ptr(write_buf.data());
        gp.set_data_length(length);
        gp.set_streaming_width(length);
        sc_time delay = quantum_keeper.get_local_time();
        auto pre_delay = delay;
        exec_b_transport(gp, delay);
        if(pre_delay > delay)
            quantum_keeper.reset();
        else
            dbus_inc += (delay - quantum_keeper.get_local_time()) / curr_clk;
        SCCTRACE(this->name()) << "[local time: " << delay << "]: finish write_mem(0x" << std::hex << addr.val << ") : 0x"
                               << (length == 4   ? *(uint32_t*)data
                                   : length == 2 ? *(uint16_t*)data
                                                 : (unsigned)*data);
        if(gp.get_response_status() != tlm::TLM_OK_RESPONSE) {
            return false;
        }
        if(gp.is_dmi_allowed() && !GET_PROP_VALUE(disable_dmi)) {
            gp.set_command(tlm::TLM_READ_COMMAND);
            gp.set_address(addr.val);
            tlm_dmi_ext dmi_data;
            if(exec_get_direct_mem_ptr(gp, dmi_data)) {
                if(dmi_data.is_write_allowed() && (addr.val + length - 1) <= dmi_data.get_end_address())
                    write_lut.addEntry(dmi_data, dmi_data.get_start_address(),
                                       dmi_data.get_end_address() - dmi_data.get_start_address() + 1);
            }
        }
        return true;
    }
}

template <unsigned int BUSWIDTH, typename QK>
bool core_complex<BUSWIDTH, QK>::read_mem_dbg(addr_t addr, unsigned length, uint8_t* const data) {
    tlm::tlm_generic_payload gp;
    gp.set_command(tlm::TLM_READ_COMMAND);
    gp.set_address(addr.val);
    gp.set_data_ptr(data);
    gp.set_data_length(length);
    gp.set_streaming_width(length);
    return dbus->transport_dbg(gp) == length;
}

template <unsigned int BUSWIDTH, typename QK>
bool core_complex<BUSWIDTH, QK>::write_mem_dbg(addr_t addr, unsigned length, const uint8_t* const data) {
    write_buf.resize(length);
    std::copy(data, data + length, write_buf.begin()); // need to copy as TLM does not guarantee data integrity
    tlm::tlm_generic_payload gp;
    gp.set_command(tlm::TLM_WRITE_COMMAND);
    gp.set_address(addr.val);
    gp.set_data_ptr(write_buf.data());
    gp.set_data_length(length);
    gp.set_streaming_width(length);
    return dbus->transport_dbg(gp) == length;
}

template class core_complex<scc::LT>;
template class core_complex<32>;
template class core_complex<64>;
template class core_complex<scc::LT, tlm::scc::quantumkeeper_mt>;
template class core_complex<32, tlm::scc::quantumkeeper_mt>;
template class core_complex<64, tlm::scc::quantumkeeper_mt>;

} // namespace riscv
} /* namespace sysc */
