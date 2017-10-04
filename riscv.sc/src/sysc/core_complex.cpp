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
//       eyck@minres.com - initial API and implementation
//
//
////////////////////////////////////////////////////////////////////////////////

#include <iss/arch/riscv_hart_msu_vp.h>
#include <iss/arch/rv32imac.h>
#include <iss/iss.h>
#include <iss/vm_types.h>
#include <sysc/SiFive/core_complex.h>

#include "scc/report.h"

namespace sysc {
namespace SiFive {

class core_wrapper : public iss::arch::riscv_hart_msu_vp<iss::arch::rv32imac> {
public:
    using core_type = iss::arch::rv32imac;
    using base_type = iss::arch::riscv_hart_msu_vp<iss::arch::rv32imac>;
    using phys_addr_t = typename iss::arch::traits<iss::arch::rv32imac>::phys_addr_t;
    core_wrapper(core_complex *owner)
    : owner(owner) {}

    void notify_phase(iss::arch_if::exec_phase phase);

    iss::status read_mem(phys_addr_t addr, unsigned length, uint8_t *const data) {
        if (addr.type & iss::DEBUG)
            return owner->read_mem_dbg(addr.val, length, data) ? iss::Ok : iss::Err;
        else
            return owner->read_mem(addr.val, length, data) ? iss::Ok : iss::Err;
    }

    iss::status write_mem(phys_addr_t addr, unsigned length, const uint8_t *const data) {
        if (addr.type & iss::DEBUG)
            return owner->write_mem_dbg(addr.val, length, data) ? iss::Ok : iss::Err;
        else
            return owner->write_mem(addr.val, length, data) ? iss::Ok : iss::Err;
    }

private:
    core_complex *const owner;
};

void core_wrapper::notify_phase(exec_phase phase) {
    core_type::notify_phase(phase);
    if (phase == ISTART) owner->sync();
}

core_complex::core_complex(sc_core::sc_module_name name)
: sc_core::sc_module(name)
, NAMED(initiator)
, NAMED(clk_i)
, NAMED(rst_i)
, NAMED(elf_file, this)
, NAMED(enable_disass, true, this)
, NAMED(reset_address, 0ULL, this)
, NAMED(gdb_server_port, 0, this)
, NAMED(dump_ir, false, this)
, read_lut(tlm_dmi_ext())
, write_lut(tlm_dmi_ext()) {

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
}

core_complex::~core_complex() = default;

void core_complex::trace(sc_core::sc_trace_file *trf) {}

void core_complex::before_end_of_elaboration() {
    cpu = std::make_unique<core_wrapper>(this);
    vm = iss::create<iss::arch::rv32imac>(cpu.get(), gdb_server_port.value, dump_ir.value);
    vm->setDisassEnabled(enable_disass.value);
}

void core_complex::start_of_simulation() {
    quantum_keeper.reset();
    if (elf_file.value.size() > 0) cpu->load_file(elf_file.value);
}

void core_complex::clk_cb() { curr_clk = clk_i.read(); }

void core_complex::run() {
    wait(sc_core::SC_ZERO_TIME);
    cpu->reset(reset_address.value);
    try {
        vm->start(-1);
    } catch (iss::simulation_stopped &e) {
    }
    sc_core::sc_stop();
}

bool core_complex::read_mem(uint64_t addr, unsigned length, uint8_t *const data) {
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
        gp.set_streaming_width(4);
        auto delay{quantum_keeper.get_local_time()};
        initiator->b_transport(gp, delay);
        LOG(TRACE) << "read_mem(0x" << std::hex << addr << ") : " << data;
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
        gp.set_streaming_width(4);
        auto delay{quantum_keeper.get_local_time()};
        initiator->b_transport(gp, delay);
        quantum_keeper.set(delay);
        LOG(TRACE) << "write_mem(0x" << std::hex << addr << ") : " << data;
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
