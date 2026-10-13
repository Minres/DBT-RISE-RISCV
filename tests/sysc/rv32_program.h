#pragma once

#include <cstdint>

// Minimal RV32I encoder, enough to hand-assemble test programs without a
// RISC-V toolchain.
namespace rv32 {

constexpr uint32_t i_type(int32_t imm, unsigned rs1, unsigned funct3, unsigned rd, unsigned opcode) {
    return (static_cast<uint32_t>(imm & 0xfff) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode;
}

constexpr uint32_t lw(unsigned rd, int32_t offset, unsigned rs1) { return i_type(offset, rs1, 2, rd, 0x03); }

constexpr uint32_t addi(unsigned rd, unsigned rs1, int32_t imm) { return i_type(imm, rs1, 0, rd, 0x13); }

constexpr uint32_t sw(unsigned rs2, int32_t offset, unsigned rs1) {
    auto imm = static_cast<uint32_t>(offset & 0xfff);
    return ((imm >> 5) << 25) | (rs2 << 20) | (rs1 << 15) | (2 << 12) | ((imm & 0x1f) << 7) | 0x23;
}

constexpr uint32_t jal(unsigned rd, int32_t offset) {
    auto imm = static_cast<uint32_t>(offset);
    return (((imm >> 20) & 0x1) << 31) | (((imm >> 1) & 0x3ff) << 21) | (((imm >> 11) & 0x1) << 20) | (((imm >> 12) & 0xff) << 12) |
           (rd << 7) | 0x6f;
}

} // namespace rv32
