#include "test_main.h"

#include <algorithm>
#include <cstdlib>
#include <gtest/gtest.h>
#include <scc/report.h>
#include <systemc>

int sc_main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    // Also registers the global CCI broker, which must exist exactly once and
    // outlive every simulation context the fixture creates.
    auto* verbose = std::getenv("SCC_TEST_VERBOSE");
    auto level = verbose ? static_cast<scc::log>(std::min(std::strtoul(verbose, nullptr, 10) + 4, 7UL)) : scc::log::WARNING;
    scc::init_logging(scc::LogConfig().logLevel(level).logAsync(false));
    return RUN_ALL_TESTS();
}

SystemCTestFixture::SystemCTestFixture() {
    sc_core::sc_report_handler::set_actions(sc_core::SC_ID_NO_SC_START_ACTIVITY_, sc_core::SC_DO_NOTHING);
    ResetSystemCContext();
}

void SystemCTestFixture::ResetSystemCContext() {
    sc_core::sc_curr_simcontext = new sc_core::sc_simcontext();
    sc_core::sc_default_global_context = sc_core::sc_curr_simcontext;
}

void SystemCTestFixture::TearDown() { ResetSystemCContext(); }
