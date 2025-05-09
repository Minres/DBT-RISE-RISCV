/*******************************************************************************
 * Copyright (C) 2025 MINRES Technologies GmbH
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
 * Contributors:
 *       eyck@minres.com - initial implementation
 ******************************************************************************/

#include "memory_if.h"
#include "iss/arch/riscv_hart_common.h"
#include "iss/vm_types.h"
#include <util/logging.h>

namespace iss {
namespace mem {
struct clic_config {
    uint64_t clic_base{0xc0000000};
    unsigned clic_int_ctl_bits{4};
    unsigned clic_num_irq{16};
    unsigned clic_num_trigger{0};
    bool nmode{false};
};

inline void read_reg_with_offset(uint32_t reg, uint8_t offs, uint8_t* const data, unsigned length) {
    auto reg_ptr = reinterpret_cast<uint8_t*>(&reg);
    switch(offs) {
    default:
        for(auto i = 0U; i < length; ++i)
            *(data + i) = *(reg_ptr + i);
        break;
    case 1:
        for(auto i = 0U; i < length; ++i)
            *(data + i) = *(reg_ptr + 1 + i);
        break;
    case 2:
        for(auto i = 0U; i < length; ++i)
            *(data + i) = *(reg_ptr + 2 + i);
        break;
    case 3:
        *data = *(reg_ptr + 3);
        break;
    }
}

inline void write_reg_with_offset(uint32_t& reg, uint8_t offs, const uint8_t* const data, unsigned length) {
    auto reg_ptr = reinterpret_cast<uint8_t*>(&reg);
    switch(offs) {
    default:
        for(auto i = 0U; i < length; ++i)
            *(reg_ptr + i) = *(data + i);
        break;
    case 1:
        for(auto i = 0U; i < length; ++i)
            *(reg_ptr + 1 + i) = *(data + i);
        break;
    case 2:
        for(auto i = 0U; i < length; ++i)
            *(reg_ptr + 2 + i) = *(data + i);
        break;
    case 3:
        *(reg_ptr + 3) = *data;
        break;
    }
}

template <typename WORD_TYPE> struct clic : public memory_elem {
    using this_class = clic<WORD_TYPE>;
    using reg_t = WORD_TYPE;
    constexpr static unsigned WORD_LEN = sizeof(WORD_TYPE) * 8;

    clic(arch::priv_if<WORD_TYPE> hart_if, clic_config cfg)
    : hart_if(hart_if)
    , cfg(cfg) {
        clic_int_reg.resize(cfg.clic_num_irq, clic_int_reg_t{.raw = 0});
        clic_cfg_reg = 0x30;
        clic_mact_lvl = clic_mprev_lvl = (1 << (cfg.clic_int_ctl_bits)) - 1;
        clic_uact_lvl = clic_uprev_lvl = (1 << (cfg.clic_int_ctl_bits)) - 1;
        hart_if.csr_rd_cb[arch::mtvt] = MK_CSR_RD_CB(read_plain);
        hart_if.csr_wr_cb[arch::mtvt] = MK_CSR_WR_CB(write_xtvt);
        //        hart_if.csr_rd_cb[mxnti] = MK_CSR_RD_CB(read_plain(a,r);};
        //        hart_if.csr_wr_cb[mxnti] = MK_CSR_WR_CB(write_plain(a,r);};
        hart_if.csr_rd_cb[arch::mintstatus] = MK_CSR_RD_CB(read_intstatus);
        hart_if.csr_wr_cb[arch::mintstatus] = MK_CSR_WR_CB(write_null);
        //        hart_if.csr_rd_cb[mscratchcsw] = MK_CSR_RD_CB(read_plain(a,r);};
        //        hart_if.csr_wr_cb[mscratchcsw] = MK_CSR_WR_CB(write_plain(a,r);};
        //        hart_if.csr_rd_cb[mscratchcswl] = MK_CSR_RD_CB(read_plain(a,r);};
        //        hart_if.csr_wr_cb[mscratchcswl] = MK_CSR_WR_CB(write_plain(a,r);};
        hart_if.csr_rd_cb[arch::mintthresh] = MK_CSR_RD_CB(read_plain);
        hart_if.csr_wr_cb[arch::mintthresh] = MK_CSR_WR_CB(write_intthresh);
        if(cfg.nmode) {
            hart_if.csr_rd_cb[arch::utvt] = MK_CSR_RD_CB(read_plain);
            hart_if.csr_wr_cb[arch::utvt] = MK_CSR_WR_CB(write_xtvt);
            hart_if.csr_rd_cb[arch::uintstatus] = MK_CSR_RD_CB(read_intstatus);
            hart_if.csr_wr_cb[arch::uintstatus] = MK_CSR_WR_CB(write_null);
            hart_if.csr_rd_cb[arch::uintthresh] = MK_CSR_RD_CB(read_plain);
            hart_if.csr_wr_cb[arch::uintthresh] = MK_CSR_WR_CB(write_intthresh);
        }
        hart_if.csr[arch::mintthresh] = (1 << (cfg.clic_int_ctl_bits)) - 1;
        hart_if.csr[arch::uintthresh] = (1 << (cfg.clic_int_ctl_bits)) - 1;
    }

    ~clic() = default;

    memory_if get_mem_if() override {
        return memory_if{.rd_mem{util::delegate<rd_mem_func_sig>::from<this_class, &this_class::read_mem>(this)},
                         .wr_mem{util::delegate<wr_mem_func_sig>::from<this_class, &this_class::write_mem>(this)}};
    }

    void set_next(memory_if mem) override { down_stream_mem = mem; }

    std::tuple<uint64_t, uint64_t> get_range() override { return {cfg.clic_base, cfg.clic_base + 0x7fff}; }

private:
    iss::status read_mem(iss::access_type access, uint64_t addr, unsigned length, uint8_t* data) {
        if(addr >= cfg.clic_base && (addr + length) < (cfg.clic_base + 0x8000))
            return read_clic(addr, length, data);
        return down_stream_mem.rd_mem(access, addr, length, data);
    }

    iss::status write_mem(iss::access_type access, uint64_t addr, unsigned length, uint8_t const* data) {
        if(addr >= cfg.clic_base && (addr + length) < (cfg.clic_base + 0x8000))
            return write_clic(addr, length, data);
        return down_stream_mem.wr_mem(access, addr, length, data);
    }

    iss::status read_clic(uint64_t addr, unsigned length, uint8_t* data);

    iss::status write_clic(uint64_t addr, unsigned length, uint8_t const* data);

    iss::status write_null(unsigned addr, reg_t val) { return iss::status::Ok; }

    iss::status read_plain(unsigned addr, reg_t& val) {
        val = hart_if.csr[addr];
        return iss::Ok;
    }

    iss::status write_xtvt(unsigned addr, reg_t val) {
        hart_if.csr[addr] = val & ~0x3fULL;
        return iss::Ok;
    }

    iss::status read_cause(unsigned addr, reg_t& val);
    iss::status write_cause(unsigned addr, reg_t val);

    iss::status read_intstatus(unsigned addr, reg_t& val);
    iss::status write_intthresh(unsigned addr, reg_t val);

protected:
    arch::priv_if<WORD_TYPE> hart_if;
    memory_if down_stream_mem;
    clic_config cfg;
    uint8_t clic_cfg_reg{0};
    std::array<uint32_t, 32> clic_inttrig_reg;
    union clic_int_reg_t {
        struct {
            uint8_t ip;
            uint8_t ie;
            uint8_t attr;
            uint8_t ctl;
        };
        uint32_t raw;
    };
    std::vector<clic_int_reg_t> clic_int_reg;
    uint8_t clic_mprev_lvl{0}, clic_uprev_lvl{0};
    uint8_t clic_mact_lvl{0}, clic_uact_lvl{0};
};

template <typename WORD_TYPE> iss::status clic<WORD_TYPE>::read_clic(uint64_t addr, unsigned length, uint8_t* const data) {
    if(addr == cfg.clic_base) { // cliccfg
        *data = clic_cfg_reg;
        for(auto i = 1; i < length; ++i)
            *(data + i) = 0;
    } else if(addr >= (cfg.clic_base + 0x40) && (addr + length) <= (cfg.clic_base + 0x40 + cfg.clic_num_trigger * 4)) { // clicinttrig
        auto offset = ((addr & 0x7fff) - 0x40) / 4;
        read_reg_with_offset(clic_inttrig_reg[offset], addr & 0x3, data, length);
    } else if(addr >= (cfg.clic_base + 0x1000) &&
              (addr + length) <= (cfg.clic_base + 0x1000 + cfg.clic_num_irq * 4)) { // clicintip/clicintie/clicintattr/clicintctl
        auto offset = ((addr & 0x7fff) - 0x1000) / 4;
        read_reg_with_offset(clic_int_reg[offset].raw, addr & 0x3, data, length);
    } else {
        for(auto i = 0U; i < length; ++i)
            *(data + i) = 0;
    }
    return iss::Ok;
}

template <typename WORD_TYPE> iss::status clic<WORD_TYPE>::write_clic(uint64_t addr, unsigned length, const uint8_t* const data) {
    if(addr == cfg.clic_base) { // cliccfg
        clic_cfg_reg = (clic_cfg_reg & ~0x1e) | (*data & 0x1e);
    } else if(addr >= (cfg.clic_base + 0x40) && (addr + length) <= (cfg.clic_base + 0x40 + cfg.clic_num_trigger * 4)) { // clicinttrig
        auto offset = ((addr & 0x7fff) - 0x40) / 4;
        write_reg_with_offset(clic_inttrig_reg[offset], addr & 0x3, data, length);
    } else if(addr >= (cfg.clic_base + 0x1000) &&
              (addr + length) <= (cfg.clic_base + 0x1000 + cfg.clic_num_irq * 4)) { // clicintip/clicintie/clicintattr/clicintctl
        auto offset = ((addr & 0x7fff) - 0x1000) / 4;
        write_reg_with_offset(clic_int_reg[offset].raw, addr & 0x3, data, length);
        clic_int_reg[offset].raw &= 0xf0c70101; // clicIntCtlBits->0xf0, clicintattr->0xc7, clicintie->0x1, clicintip->0x1
    }
    return iss::Ok;
}

template <typename WORD_TYPE> iss::status clic<WORD_TYPE>::read_cause(unsigned addr, reg_t& val) {
    if((hart_if.csr[arch::mtvec] & 0x3) == 3) {
        val = hart_if.csr[addr] & (1UL << (sizeof(reg_t) * 8) | (hart_if.mcause_max_irq - 1) | (0xfUL << 16));
        auto mode = (addr >> 8) & 0x3;
        switch(mode) {
        case 0:
            val |= clic_uprev_lvl << 16;
            val |= hart_if.state.mstatus.UPIE << 27;
            break;
        default:
            val |= clic_mprev_lvl << 16;
            val |= hart_if.state.mstatus.MPIE << 27;
            val |= hart_if.state.mstatus.MPP << 28;
            break;
        }
    } else
        val = hart_if.csr[addr] & ((1UL << (sizeof(WORD_TYPE) * 8 - 1)) | (hart_if.mcause_max_irq - 1));
    return iss::Ok;
}

template <typename WORD_TYPE> iss::status clic<WORD_TYPE>::write_cause(unsigned addr, reg_t val) {
    if((hart_if.csr[arch::mtvec] & 0x3) == 3) {
        auto mask = ((1UL << (sizeof(WORD_TYPE) * 8 - 1)) | (hart_if.mcause_max_irq - 1) | (0xfUL << 16));
        hart_if.csr[addr] = (val & mask) | (hart_if.csr[addr] & ~mask);
        auto mode = (addr >> 8) & 0x3;
        switch(mode) {
        case 0:
            clic_uprev_lvl = ((val >> 16) & 0xff) | (1 << (8 - cfg.clic_int_ctl_bits)) - 1;
            hart_if.state.mstatus.UPIE = (val >> 27) & 0x1;
            break;
        default:
            clic_mprev_lvl = ((val >> 16) & 0xff) | (1 << (8 - cfg.clic_int_ctl_bits)) - 1;
            hart_if.state.mstatus.MPIE = (val >> 27) & 0x1;
            hart_if.state.mstatus.MPP = (val >> 28) & 0x3;
            break;
        }
    } else {
        auto mask = ((1UL << (sizeof(WORD_TYPE) * 8 - 1)) | (hart_if.mcause_max_irq - 1));
        hart_if.csr[addr] = (val & mask) | (hart_if.csr[addr] & ~mask);
    }
    return iss::Ok;
}

template <typename WORD_TYPE> iss::status clic<WORD_TYPE>::read_intstatus(unsigned addr, reg_t& val) {
    auto mode = (addr >> 8) & 0x3;
    val = clic_uact_lvl & 0xff;
    if(mode == 0x3)
        val += (clic_mact_lvl & 0xff) << 24;
    return iss::Ok;
}

template <typename WORD_TYPE> iss::status clic<WORD_TYPE>::write_intthresh(unsigned addr, reg_t val) {
    hart_if.csr[addr] = (val & 0xff) | (1 << (cfg.clic_int_ctl_bits)) - 1;
    return iss::Ok;
}

} // namespace mem
} // namespace iss
