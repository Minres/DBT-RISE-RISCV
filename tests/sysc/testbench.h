#pragma once

#include <cstdint>
#include <scc/memory.h>
#include <scc/router.h>
#include <sysc/core_complex.h>
#include <systemc>
#include <tlm/scc/signal_initiator_mixin.h>
#include <tlm>
#include <vector>

// One rv32 core whose instruction and data buses reach a single RAM through a
// router, since a TLM target socket accepts only one initiator.
struct testbench : sc_core::sc_module {
    static constexpr uint64_t ram_size = 1ULL << 20;

    sysc::riscv::core_complex<> cpu;
    scc::router<> bus{"bus", 1, 2};
    scc::memory<ram_size> ram{"ram"};
    sc_core::sc_signal<sc_core::sc_time> clk{"clk", sc_core::sc_time(10, sc_core::SC_NS)};
    sc_core::sc_signal<bool> rst{"rst", false};
    sc_core::sc_vector<tlm::scc::tlm_signal_bool_out> clint_irq{"clint_irq"};

    explicit testbench(sc_core::sc_module_name const& nm, size_t clint_irq_count = 32)
    : sc_core::sc_module(nm)
    , cpu("cpu", clint_irq_count) {
        clint_irq.init(clint_irq_count);
        for(auto i = 0U; i < clint_irq_count; ++i)
            clint_irq[i].bind(cpu.clint_irq_i[i]);
        cpu.ibus(bus.target[0]);
        cpu.dbus(bus.target[1]);
        bus.bind_target(ram.target, 0, 0, ram_size);
        cpu.clk_i(clk);
        cpu.rst_i(rst);
    }

    void load(uint64_t addr, std::vector<uint32_t> const& words) {
        for(auto w : words) {
            write_word(addr, w);
            addr += 4;
        }
    }

    void write_word(uint64_t addr, uint32_t value) { debug_access(tlm::TLM_WRITE_COMMAND, addr, value); }

    uint32_t read_word(uint64_t addr) {
        uint32_t value = 0;
        debug_access(tlm::TLM_READ_COMMAND, addr, value);
        return value;
    }

private:
    void debug_access(tlm::tlm_command cmd, uint64_t addr, uint32_t& value) {
        tlm::tlm_generic_payload gp;
        gp.set_command(cmd);
        gp.set_address(addr);
        gp.set_data_ptr(reinterpret_cast<unsigned char*>(&value));
        gp.set_data_length(sizeof(value));
        gp.set_streaming_width(sizeof(value));
        ram.target.get_base_interface().transport_dbg(gp);
    }
};
