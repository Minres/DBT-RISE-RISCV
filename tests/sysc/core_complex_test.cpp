#include "rv32_program.h"
#include "test_main.h"
#include "testbench.h"

#include <gtest/gtest.h>
#include <memory>
#include <tlm>

namespace {

using namespace sc_core;

constexpr uint64_t reset_address = 0x0;
constexpr int32_t start_count_addr = 0x400;
constexpr int32_t loop_count_addr = 0x404;

// Counts its own starts in one RAM word, then spins incrementing another.
// RAM survives a core reset, so the start count tells how often the core
// came out of reset.
std::vector<uint32_t> counting_program() {
    using namespace rv32;
    return {
        lw(5, start_count_addr, 0),
        addi(5, 5, 1),
        sw(5, start_count_addr, 0),
        lw(6, loop_count_addr, 0), // loop:
        addi(6, 6, 1),
        sw(6, loop_count_addr, 0),
        jal(0, -12), // j loop
    };
}

class CoreComplexTest : public SystemCTestFixture {
protected:
    void SetUp() override {
        tlm::tlm_global_quantum::instance().set(sc_time(1, SC_US));
        tb = std::make_unique<testbench>("tb");
        tb->load(reset_address, counting_program());
    }

    void TearDown() override {
        tb.reset();
        SystemCTestFixture::TearDown();
    }

    uint32_t start_count() { return tb->read_word(start_count_addr); }
    uint32_t loop_count() { return tb->read_word(loop_count_addr); }

    std::unique_ptr<testbench> tb;
};

TEST_F(CoreComplexTest, RunsAProgramFromTheResetAddress) {
    sc_start(sc_time(100, SC_US));

    EXPECT_EQ(start_count(), 1u);
    EXPECT_GT(loop_count(), 0u);
}

TEST_F(CoreComplexTest, RunsAProgramOnAnRv32imacCoreWithPmpAndAia) {
    tb->cpu.core_type.set_value("rv32imac_mp_aia");
    sc_start(sc_time(100, SC_US));

    EXPECT_EQ(start_count(), 1u);
    EXPECT_GT(loop_count(), 0u);
}

TEST_F(CoreComplexTest, AdvancesSimulationTimeWhileRunning) {
    sc_start(sc_time(100, SC_US));
    auto loops_at_100us = loop_count();
    sc_start(sc_time(100, SC_US));

    EXPECT_EQ(sc_time_stamp(), sc_time(200, SC_US));
    EXPECT_GT(loop_count(), loops_at_100us);
}

// A peripheral may reset the core from its own process while the core thread
// is suspended in its quantum keeper.
TEST_F(CoreComplexTest, KeepsExecutingAfterAResetWhileRunning) {
    sc_spawn([this] {
        wait(sc_time(100, SC_US));
        tb->cpu.reset(reset_address);
    });
    sc_start(sc_time(100, SC_US) + sc_time(1, SC_NS));
    auto loops_at_reset = loop_count();
    sc_start(sc_time(100, SC_US));

    EXPECT_GT(loop_count(), loops_at_reset);
}

TEST_F(CoreComplexTest, RestartsAtTheResetAddressAfterAResetWhileRunning) {
    sc_spawn([this] {
        wait(sc_time(100, SC_US));
        tb->cpu.reset(reset_address);
    });
    sc_start(sc_time(200, SC_US));

    EXPECT_EQ(start_count(), 2u);
}

constexpr uint64_t trap_handler_addr = 0x100;
constexpr int32_t mcause_addr = 0x408;
constexpr unsigned upper_local_irq = 40;

// Enables only upper_local_irq through mieh, then spins. The trap handler
// stores mcause and spins.
std::vector<uint32_t> upper_local_irq_program() {
    using namespace rv32;
    constexpr unsigned mstatus = 0x300, mtvec = 0x305, mieh = 0x314, mcause = 0x342;
    constexpr unsigned mstatus_mie = 0x8;
    std::vector<uint32_t> program{
        addi(5, 0, trap_handler_addr),           // t0 = handler
        csrrw(0, mtvec, 5),                      // mtvec = t0
        addi(5, 0, 1 << (upper_local_irq - 32)), // t0 = enable bit
        csrrw(0, mieh, 5),                       // mieh = t0
        csrrsi(0, mstatus, mstatus_mie),         // mstatus.MIE = 1
        jal(0, 0),                               // spin
    };
    program.resize(trap_handler_addr / 4, 0);
    std::vector<uint32_t> trap_handler{
        csrrs(6, mcause, 0),   // t1 = mcause
        sw(6, mcause_addr, 0), // store t1
        jal(0, 0),             // spin
    };
    program.insert(program.end(), trap_handler.begin(), trap_handler.end());
    return program;
}

class AiaCoreComplexTest : public SystemCTestFixture {
protected:
    void SetUp() override {
        tlm::tlm_global_quantum::instance().set(sc_time(1, SC_US));
        tb = std::make_unique<testbench>("tb", 64);
        tb->cpu.core_type.set_value("rv32imac_mp_aia");
        tb->load(reset_address, upper_local_irq_program());
    }

    void TearDown() override {
        tb.reset();
        SystemCTestFixture::TearDown();
    }

    std::unique_ptr<testbench> tb;
};

TEST_F(AiaCoreComplexTest, TakesALocalInterruptAbove31OnRv32) {
    sc_spawn([this] { tb->clint_irq[upper_local_irq].write_now(true); });
    sc_start(sc_time(100, SC_US));

    constexpr uint32_t interrupt_bit = 1U << 31;
    EXPECT_EQ(tb->read_word(mcause_addr), interrupt_bit | upper_local_irq);
}

} // namespace
