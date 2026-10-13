#pragma once

#include <gtest/gtest.h>
#include <systemc>

// Gives every test its own SystemC simulation context, so one process can
// elaborate and simulate a fresh design per test.
class SystemCTestFixture : public ::testing::Test {
protected:
    SystemCTestFixture();

    // SystemC cannot destroy a simulation context, so each reset leaks the
    // previous one on purpose.
    void ResetSystemCContext();

    // Derived fixtures must destroy the modules they own before calling this:
    // a module has to die while the context that registered it is current.
    void TearDown() override;
};
