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
// came out of reset. It starts with a nop because a reset taken while the core
// waits in its quantum keeper resumes at the instruction after the reset
// address.
std::vector<uint32_t> counting_program() {
    using namespace rv32;
    return {
        addi(0, 0, 0), lw(5, start_count_addr, 0), addi(5, 5, 1), sw(5, start_count_addr, 0), lw(6, loop_count_addr, 0), // loop:
        addi(6, 6, 1), sw(6, loop_count_addr, 0),  jal(0, -12),                                                          // j loop
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

TEST_F(CoreComplexTest, AdvancesSimulationTimeWhileRunning) {
    sc_start(sc_time(100, SC_US));
    auto loops_at_100us = loop_count();
    sc_start(sc_time(100, SC_US));

    EXPECT_EQ(sc_time_stamp(), sc_time(200, SC_US));
    EXPECT_GT(loop_count(), loops_at_100us);
}

} // namespace
