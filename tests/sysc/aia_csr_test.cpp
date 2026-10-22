#include <gtest/gtest.h>
#include <iss/arch/riscv_hart_mu_p.h>
#include <iss/arch/rv32imac.h>

namespace {

using namespace iss::arch;

template <typename HART> class AiaCsrTest : public ::testing::Test {
protected:
    using reg_t = typename HART::reg_t;

    void SetUp() override {
        hart.reset(0);
        hart.set_clint_irq_num(64);
    }

    reg_t read_csr(unsigned csr) {
        reg_t val = 0;
        EXPECT_EQ(hart.read(csr_addr(iss::access_type::READ, csr), sizeof(val), reinterpret_cast<uint8_t*>(&val)), iss::Ok);
        return val;
    }

    void write_csr(unsigned csr, reg_t val) {
        EXPECT_EQ(hart.write(csr_addr(iss::access_type::WRITE, csr), sizeof(val), reinterpret_cast<uint8_t const*>(&val)), iss::Ok);
    }

    HART hart;

private:
    static iss::addr_t csr_addr(iss::access_type access, unsigned csr) {
        return {iss::address_type::LOGICAL, access, traits<typename HART::core>::CSR, csr};
    }
};

using MuAiaCsrTest = AiaCsrTest<riscv_hart_mu_p<rv32imac, FEAT_AIA>>;

TEST_F(MuAiaCsrTest, WritingMiphSetsTheUpperHalfOfMipAndLeavesTheLowerHalfAlone) {
    constexpr reg_t irq40 = 1U << (40 - 32);
    write_csr(miph, irq40);

    EXPECT_EQ(read_csr(miph), irq40);
    EXPECT_EQ(read_csr(mip), 0u);
}

} // namespace
