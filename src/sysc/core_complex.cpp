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

#include "sysc/core_complex.h"
#include "iss/arch/riscv_hart_msu_vp.h"
#include "iss/arch/rv32imac.h"
#include "iss/debugger/encoderdecoder.h"
#include "iss/debugger/gdb_session.h"
#include "iss/debugger/server.h"
#include "iss/debugger/target_adapter_if.h"
#include "iss/iss.h"
#include "iss/vm_types.h"
#include "scc/report.h"
#include <sstream>
#include <iostream>

#ifdef WITH_SCV
#include <array>
#include <scv.h>
#endif

namespace sysc {
namespace SiFive {
using namespace std;
using namespace iss;
using namespace logging;
using namespace sc_core;

namespace {
iss::debugger::encoder_decoder encdec;
}

//using core_type = iss::arch::rv32imac;
using core_type = iss::arch::rv32imac;

namespace {

std::array<const char, 4> lvl = {{'U', 'S', 'H', 'M'}};

std::array<const char*, 16> trap_str = { {
		"Instruction address misaligned",
		"Instruction access fault",
		"Illegal instruction",
		"Breakpoint",
		"Load address misaligned",
		"Load access fault",
		"Store/AMO address misaligned",
		"Store/AMO access fault",
		"Environment call from U-mode",
		"Environment call from S-mode",
		"Reserved",
		"Environment call from M-mode",
		"Instruction page fault",
		"Load page fault",
		"Reserved",
		"Store/AMO page fault"
} };
std::array<const char*, 12> irq_str = { {
		"User software interrupt", "Supervisor software interrupt", "Reserved", "Machine software interrupt",
		"User timer interrupt",    "Supervisor timer interrupt",    "Reserved", "Machine timer interrupt",
		"User external interrupt", "Supervisor external interrupt", "Reserved", "Machine external interrupt" } };
}

class core_wrapper : public iss::arch::riscv_hart_msu_vp<core_type> {
public:
    using base_type = arch::riscv_hart_msu_vp<core_type>;
    using phys_addr_t = typename arch::traits<core_type>::phys_addr_t;
    core_wrapper(core_complex *owner)
    : owner(owner) 
    {
    }

    uint32_t get_mode() { return this->reg.machine_state; }

    inline void set_interrupt_execution(bool v) { this->interrupt_sim = v?1:0; }

    inline bool get_interrupt_execution() { return this->interrupt_sim; }

    base_type::hart_state<base_type::reg_t> &get_state() { return this->state; }

    void notify_phase(exec_phase p) override {
        if (p == ISTART) owner->sync(this->reg.icount + cycle_offset);
    }

    sync_type needed_sync() const override { return PRE_SYNC; }

    void disass_output(uint64_t pc, const std::string instr) override {
        if (INFO <= Log<Output2FILE<disass>>::reporting_level() && Output2FILE<disass>::stream()) {
            std::stringstream s;
            s << "[p:" << lvl[this->reg.machine_state] << ";s:0x" << std::hex << std::setfill('0')
              << std::setw(sizeof(reg_t) * 2) << (reg_t)state.mstatus << std::dec << ";c:" << this->reg.icount << "]";
            Log<Output2FILE<disass>>().get(INFO, "disass")
                << "0x" << std::setw(16) << std::right << std::setfill('0') << std::hex << pc << "\t\t" << std::setw(40)
                << std::setfill(' ') << std::left << instr << s.str();
        }
        owner->disass_output(pc, instr);
    };

    status read_mem(phys_addr_t addr, unsigned length, uint8_t *const data) override {
        if (addr.access && access_type::DEBUG)
            return owner->read_mem_dbg(addr.val, length, data) ? Ok : Err;
        else {
            return owner->read_mem(addr.val, length, data, addr.access && access_type::FETCH) ? Ok : Err;
        }
    }

    status write_mem(phys_addr_t addr, unsigned length, const uint8_t *const data) override {
        if (addr.access && access_type::DEBUG)
            return owner->write_mem_dbg(addr.val, length, data) ? Ok : Err;
        else {
            auto res = owner->write_mem(addr.val, length, data) ? Ok : Err;
            // clear MTIP on mtimecmp write
            if (addr.val == 0x2004000) {
                reg_t val;
                this->read_csr(arch::mip, val);
                if (val & (1ULL << 7)) this->write_csr(arch::mip, val & ~(1ULL << 7));
            }
            return res;
        }
    }

    status read_csr(unsigned addr, reg_t &val) override {
        if((addr==arch::time || addr==arch::timeh) && owner->mtime_o.get_interface(0)){
            uint64_t time_val;
            bool ret = owner->mtime_o->nb_peek(time_val);
            if (addr == iss::arch::time) {
                val = static_cast<reg_t>(time_val);
            } else if (addr == iss::arch::timeh) {
                if (sizeof(reg_t) != 4) return iss::Err;
                val = static_cast<reg_t>(time_val >> 32);
            }
            return ret?Ok:Err;
        } else {
            return base_type::read_csr(addr, val);
        }
    }

    void wait_until(uint64_t flags) override {
        SCCDEBUG(owner->name()) << "Sleeping until interrupt";
        do {
            sc_core::wait(wfi_evt);
        } while (this->reg.pending_trap == 0);
        base_type::wait_until(flags);
    }

    void local_irq(short id, bool value) {
        base_type::reg_t mask = 0;
        switch (id) {
        case 16: // SW
            mask = 1 << 3;
            break;
        case 17: // timer
            mask = 1 << 7;
            break;
        case 18: // external
            mask = 1 << 11;
            break;
        default:
            /* do nothing*/
            break;
        }
        if (value) {
            this->csr[arch::mip] |= mask;
            wfi_evt.notify();
        } else
            this->csr[arch::mip] &= ~mask;
        this->check_interrupt();
    }

private:
    core_complex *const owner;
    sc_event wfi_evt;
};

int cmd_sysc(int argc, char *argv[], debugger::out_func of, debugger::data_func df,
             debugger::target_adapter_if *tgt_adapter) {
    if (argc > 1) {
        if (strcasecmp(argv[1], "print_time") == 0) {
            std::string t = sc_time_stamp().to_string();
            of(t.c_str());
            std::array<char, 64> buf;
            encdec.enc_string(t.c_str(), buf.data(), 63);
            df(buf.data());
            return Ok;
        } else if (strcasecmp(argv[1], "break") == 0) {
            sc_time t;
            if (argc == 4) {
                t = scc::parse_from_string(argv[2], argv[3]);
            } else if (argc == 3) {
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

core_complex::core_complex(sc_module_name name)
: sc_module(name)
, read_lut(tlm_dmi_ext())
, write_lut(tlm_dmi_ext())
, tgt_adapter(nullptr)
#ifdef WITH_SCV
, m_db(scv_tr_db::get_default_db())
, stream_handle(nullptr)
, instr_tr_handle(nullptr)
, fetch_tr_handle(nullptr)
#endif
{
    SC_HAS_PROCESS(core_complex);// NOLINT
    initiator.register_invalidate_direct_mem_ptr([=](uint64_t start, uint64_t end) -> void {
        auto lut_entry = read_lut.getEntry(start);
        if (lut_entry.get_granted_access() != tlm::tlm_dmi::DMI_ACCESS_NONE && end <= lut_entry.get_end_address() + 1) {
            read_lut.removeEntry(lut_entry);
        }
        lut_entry = write_lut.getEntry(start);
        if (lut_entry.get_granted_access() != tlm::tlm_dmi::DMI_ACCESS_NONE && end <= lut_entry.get_end_address() + 1) {
            write_lut.removeEntry(lut_entry);
        }
    });

    SC_THREAD(run);
    SC_METHOD(clk_cb);
    sensitive << clk_i;
    SC_METHOD(rst_cb);
    sensitive << rst_i;
    SC_METHOD(sw_irq_cb);
    sensitive << sw_irq_i;
    SC_METHOD(timer_irq_cb);
    sensitive << timer_irq_i;
    SC_METHOD(global_irq_cb);
    sensitive << global_irq_i;
}

core_complex::~core_complex() = default;

void core_complex::trace(sc_trace_file *trf) const {}

using vm_ptr= std::unique_ptr<iss::vm_if>;
vm_ptr create_cpu(core_wrapper* cpu, std::string const& backend, unsigned gdb_port){
    if(backend == "interp")
        return vm_ptr{iss::interp::create<core_type>(cpu, gdb_port)};
#ifdef WITH_LLVM
    if(backend == "llvm")
        return vm_ptr{iss::llvm::create(lcpu, gdb_port)};
#endif
    if(backend == "tcc")
        return vm_ptr{iss::tcc::create<core_type>(cpu, gdb_port)};
    return {nullptr};
}

void core_complex::before_end_of_elaboration() {
    SCCDEBUG(SCMOD)<<"instantiating iss::arch::mnrv32 with "<<backend.get_value()<<" backend";
    cpu = scc::make_unique<core_wrapper>(this);
    vm = create_cpu(cpu.get(), backend.get_value(), gdb_server_port.get_value());
#ifdef WITH_SCV
    vm->setDisassEnabled(enable_disass.get_value() || m_db != nullptr);
#else
    vm->setDisassEnabled(enable_disass.get_value());
#endif
    auto *srv = debugger::server<debugger::gdb_session>::get();
    if (srv) tgt_adapter = srv->get_target();
    if (tgt_adapter)
        tgt_adapter->add_custom_command(
            {"sysc", [this](int argc, char *argv[], debugger::out_func of,
                            debugger::data_func df) -> int { return cmd_sysc(argc, argv, of, df, tgt_adapter); },
             "SystemC sub-commands: break <time>, print_time"});
}

void core_complex::start_of_simulation() {
    quantum_keeper.reset();
    if (elf_file.get_value().size() > 0) {
        istringstream is(elf_file.get_value());
        string s;
        while (getline(is, s, ',')) {
            std::pair<uint64_t, bool> start_addr = cpu->load_file(s);
            if (reset_address.is_default_value() && start_addr.second == true)
                reset_address.set_value(start_addr.first);
        }
    }
    if (m_db != nullptr && stream_handle == nullptr) {
        string basename(this->name());
        stream_handle = new SCVNS scv_tr_stream((basename + ".instr").c_str(), "TRANSACTOR", m_db);
        instr_tr_handle = new SCVNS scv_tr_generator<>("execute", *stream_handle);
        fetch_tr_handle = new SCVNS scv_tr_generator<uint64_t>("fetch", *stream_handle);
    }
}

void core_complex::disass_output(uint64_t pc, const std::string instr_str) {
    if (m_db == nullptr) return;
    if (tr_handle.is_active()) tr_handle.end_transaction();
    tr_handle = instr_tr_handle->begin_transaction();
    tr_handle.record_attribute("PC", pc);
    tr_handle.record_attribute("INSTR", instr_str);
    tr_handle.record_attribute("MODE", lvl[cpu->get_mode()]);
    tr_handle.record_attribute("MSTATUS", cpu->get_state().mstatus.backing.val);
    tr_handle.record_attribute("LTIME_START", quantum_keeper.get_current_time().value() / 1000);
}

void core_complex::clk_cb() {
    curr_clk = clk_i.read();
    if (curr_clk == SC_ZERO_TIME) cpu->set_interrupt_execution(true);
}

void core_complex::rst_cb() {
    if (rst_i.read()) cpu->set_interrupt_execution(true);
}

void core_complex::sw_irq_cb() { cpu->local_irq(16, sw_irq_i.read()); }

void core_complex::timer_irq_cb() { cpu->local_irq(17, timer_irq_i.read()); }

void core_complex::global_irq_cb() { cpu->local_irq(18, global_irq_i.read()); }

void core_complex::run() {
    wait(SC_ZERO_TIME); // separate from elaboration phase
    do {
        if (rst_i.read()) {
            cpu->reset(reset_address.get_value());
            wait(rst_i.negedge_event());
        }
        while (clk_i.read() == SC_ZERO_TIME) {
            wait(clk_i.value_changed_event());
        }
        cpu->set_interrupt_execution(false);
        vm->start();
    } while (cpu->get_interrupt_execution());
    sc_stop();
}

bool core_complex::read_mem(uint64_t addr, unsigned length, uint8_t *const data, bool is_fetch) {
    auto lut_entry = read_lut.getEntry(addr);
    if (lut_entry.get_granted_access() != tlm::tlm_dmi::DMI_ACCESS_NONE &&
        addr + length <= lut_entry.get_end_address() + 1) {
        auto offset = addr - lut_entry.get_start_address();
        std::copy(lut_entry.get_dmi_ptr() + offset, lut_entry.get_dmi_ptr() + offset + length, data);
        quantum_keeper.inc(lut_entry.get_read_latency());
        return true;
    } else {
        tlm::tlm_generic_payload gp;
        gp.set_command(tlm::TLM_READ_COMMAND);
        gp.set_address(addr);
        gp.set_data_ptr(data);
        gp.set_data_length(length);
        gp.set_streaming_width(length);
        sc_time delay{quantum_keeper.get_local_time()};
#ifdef WITH_SCV
        if (m_db != nullptr && tr_handle.is_valid()) {
            if (is_fetch && tr_handle.is_active()) {
                tr_handle.end_transaction();
            }
            auto preExt = new scv4tlm::tlm_recording_extension(tr_handle, this);
            gp.set_extension(preExt);
        }
#endif
        initiator->b_transport(gp, delay);
        SCCTRACE(this->name()) << "read_mem(0x" << std::hex << addr << ") : " << data;
        if (gp.get_response_status() != tlm::TLM_OK_RESPONSE) {
            return false;
        }
        if (gp.is_dmi_allowed()) {
            gp.set_command(tlm::TLM_READ_COMMAND);
            gp.set_address(addr);
            tlm_dmi_ext dmi_data;
            if (initiator->get_direct_mem_ptr(gp, dmi_data)) {
                if (dmi_data.is_read_allowed())
                    read_lut.addEntry(dmi_data, dmi_data.get_start_address(),
                                      dmi_data.get_end_address() - dmi_data.get_start_address() + 1);
                if (dmi_data.is_write_allowed())
                    write_lut.addEntry(dmi_data, dmi_data.get_start_address(),
                                       dmi_data.get_end_address() - dmi_data.get_start_address() + 1);
            }
        }
        return true;
    }
}

bool core_complex::write_mem(uint64_t addr, unsigned length, const uint8_t *const data) {
    auto lut_entry = write_lut.getEntry(addr);
    if (lut_entry.get_granted_access() != tlm::tlm_dmi::DMI_ACCESS_NONE &&
        addr + length <= lut_entry.get_end_address() + 1) {
        auto offset = addr - lut_entry.get_start_address();
        std::copy(data, data + length, lut_entry.get_dmi_ptr() + offset);
        quantum_keeper.inc(lut_entry.get_read_latency());
        return true;
    } else {
        write_buf.resize(length);
        std::copy(data, data + length, write_buf.begin()); // need to copy as TLM does not guarantee data integrity
        tlm::tlm_generic_payload gp;
        gp.set_command(tlm::TLM_WRITE_COMMAND);
        gp.set_address(addr);
        gp.set_data_ptr(write_buf.data());
        gp.set_data_length(length);
        gp.set_streaming_width(length);
        sc_time delay{quantum_keeper.get_local_time()};
#ifdef WITH_SCV
        if (m_db != nullptr && tr_handle.is_valid()) {
            auto preExt = new scv4tlm::tlm_recording_extension(tr_handle, this);
            gp.set_extension(preExt);
        }
#endif
        initiator->b_transport(gp, delay);
        quantum_keeper.set(delay);
        SCCTRACE() << "write_mem(0x" << std::hex << addr << ") : " << data;
        if (gp.get_response_status() != tlm::TLM_OK_RESPONSE) {
            return false;
        }
        if (gp.is_dmi_allowed()) {
            gp.set_command(tlm::TLM_READ_COMMAND);
            gp.set_address(addr);
            tlm_dmi_ext dmi_data;
            if (initiator->get_direct_mem_ptr(gp, dmi_data)) {
                if (dmi_data.is_read_allowed())
                    read_lut.addEntry(dmi_data, dmi_data.get_start_address(),
                                      dmi_data.get_end_address() - dmi_data.get_start_address() + 1);
                if (dmi_data.is_write_allowed())
                    write_lut.addEntry(dmi_data, dmi_data.get_start_address(),
                                       dmi_data.get_end_address() - dmi_data.get_start_address() + 1);
            }
        }
        return true;
    }
}

bool core_complex::read_mem_dbg(uint64_t addr, unsigned length, uint8_t *const data) {
    auto lut_entry = read_lut.getEntry(addr);
    if (lut_entry.get_granted_access() != tlm::tlm_dmi::DMI_ACCESS_NONE &&
        addr + length <= lut_entry.get_end_address() + 1) {
        auto offset = addr - lut_entry.get_start_address();
        std::copy(lut_entry.get_dmi_ptr() + offset, lut_entry.get_dmi_ptr() + offset + length, data);
        quantum_keeper.inc(lut_entry.get_read_latency());
        return true;
    } else {
        tlm::tlm_generic_payload gp;
        gp.set_command(tlm::TLM_READ_COMMAND);
        gp.set_address(addr);
        gp.set_data_ptr(data);
        gp.set_data_length(length);
        gp.set_streaming_width(length);
        return initiator->transport_dbg(gp) == length;
    }
}

bool core_complex::write_mem_dbg(uint64_t addr, unsigned length, const uint8_t *const data) {
    auto lut_entry = write_lut.getEntry(addr);
    if (lut_entry.get_granted_access() != tlm::tlm_dmi::DMI_ACCESS_NONE &&
        addr + length <= lut_entry.get_end_address() + 1) {
        auto offset = addr - lut_entry.get_start_address();
        std::copy(data, data + length, lut_entry.get_dmi_ptr() + offset);
        quantum_keeper.inc(lut_entry.get_read_latency());
        return true;
    } else {
        write_buf.resize(length);
        std::copy(data, data + length, write_buf.begin()); // need to copy as TLM does not guarantee data integrity
        tlm::tlm_generic_payload gp;
        gp.set_command(tlm::TLM_WRITE_COMMAND);
        gp.set_address(addr);
        gp.set_data_ptr(write_buf.data());
        gp.set_data_length(length);
        gp.set_streaming_width(length);
        return initiator->transport_dbg(gp) == length;
    }
}

} /* namespace SiFive */
} /* namespace sysc */
