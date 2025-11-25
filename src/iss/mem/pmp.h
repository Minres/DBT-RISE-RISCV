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

#include "iss/arch/riscv_hart_common.h"
#include "iss/arch/traits.h"
#include "iss/vm_types.h"
#include "memory_if.h"
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

template <typename PLAT> struct pmp : public memory_elem {
    using this_class = pmp<PLAT>;
    using reg_t = typename PLAT::reg_t;

    pmp(arch::priv_if<reg_t> hart_if)
    : hart_if(hart_if) {
        for(size_t i = arch::pmpaddr0; i <= arch::pmpaddr15; ++i) {
            hart_if.csr_rd_cb[i] = MK_CSR_RD_CB(read_plain);
            hart_if.csr_wr_cb[i] = MK_CSR_WR_CB(write_plain);
        }
        for(size_t i = arch::pmpcfg0; i < arch::pmpcfg0 + 16 / sizeof(reg_t); ++i) {
            hart_if.csr_rd_cb[i] = MK_CSR_RD_CB(read_plain);
            hart_if.csr_wr_cb[i] = MK_CSR_WR_CB(write_pmpcfg);
        }
    }

    virtual ~pmp() = default;

    memory_if get_mem_if() override {
        return memory_if{.rd_mem{util::delegate<rd_mem_func_sig>::from<this_class, &this_class::read_mem>(this)},
                         .wr_mem{util::delegate<wr_mem_func_sig>::from<this_class, &this_class::write_mem>(this)}};
    }

    void set_next(memory_if mem) override { down_stream_mem = mem; }

private:
    iss::status read_mem(const addr_t& addr, unsigned length, uint8_t* data) {
        assert((addr.type == iss::address_type::PHYSICAL || is_debug(addr.access)) && "Only physical addresses are expected in pmp");
        if(likely(addr.space == arch::traits<PLAT>::MEM) && !pmp_check(addr.access, addr.val, length) && !is_debug(addr.access)) {
            hart_if.fault_data = addr;
            if(is_debug(addr.access))
                throw trap_access(0, addr.val);
            hart_if.reg.trap_state = (1UL << 31) | ((addr.access == access_type::FETCH ? 1 : 5) << 16); // issue trap 1
            return iss::Err;
        }
        return down_stream_mem.rd_mem(addr, length, data);
    }

    iss::status write_mem(const addr_t& addr, unsigned length, uint8_t const* data) {
        assert((addr.type == iss::address_type::PHYSICAL || is_debug(addr.access)) && "Only physical addresses are expected in pmp");
        if(likely(addr.space == arch::traits<PLAT>::MEM) && !pmp_check(addr.access, addr.val, length) && !is_debug(addr.access)) {
            hart_if.fault_data = addr;
            if(is_debug(addr.access))
                throw trap_access(0, addr.val);
            hart_if.reg.trap_state = (1UL << 31) | (7 << 16); // issue trap 1
            return iss::Err;
        }
        return down_stream_mem.wr_mem(addr, length, data);
    }

    iss::status read_plain(unsigned addr, reg_t& val) {
        val = hart_if.csr[addr];
        return iss::Ok;
    }

    iss::status write_plain(unsigned addr, reg_t const& val) {
        hart_if.csr[addr] = val;
        return iss::Ok;
    }

    iss::status write_pmpcfg(unsigned addr, reg_t val) {
        hart_if.csr[addr] = val & 0x9f9f9f9f;
        return iss::Ok;
    }

    bool pmp_check(const access_type type, const uint64_t addr, const unsigned len);

protected:
    arch::priv_if<reg_t> hart_if;
    memory_if down_stream_mem;
};

template <typename PLAT> bool pmp<PLAT>::pmp_check(const access_type type, const uint64_t addr, const unsigned len) {
    constexpr auto PMP_SHIFT = 2U;
    constexpr auto PMP_R = 0x1U;
    constexpr auto PMP_W = 0x2U;
    constexpr auto PMP_X = 0x4U;
    constexpr auto PMP_A = 0x18U;
    constexpr auto PMP_L = 0x80U;
    constexpr auto PMP_TOR = 0x1U;
    constexpr auto PMP_NA4 = 0x2U;
    constexpr auto PMP_NAPOT = 0x3U;
    reg_t base = 0;
    auto any_active = false;
    auto const cfg_reg_size = sizeof(reg_t);
    for(size_t i = 0; i < 16; i++) {
        reg_t tor = hart_if.csr[arch::pmpaddr0 + i] << PMP_SHIFT;
        uint8_t cfg = hart_if.csr[arch::pmpcfg0 + (i / cfg_reg_size)] >> (i % cfg_reg_size);
        if(cfg & PMP_A) {
            any_active = true;
            auto pmp_a = (cfg & PMP_A) >> 3;
            auto is_tor = pmp_a == PMP_TOR;
            auto is_na4 = pmp_a == PMP_NA4;

            reg_t mask = (hart_if.csr[arch::pmpaddr0 + i] << 1) | (!is_na4);
            mask = ~(mask & ~(mask + 1)) << PMP_SHIFT;

            // Check each 4-byte sector of the access
            auto any_match = false;
            auto all_match = true;
            for(reg_t offset = 0; offset < len; offset += 1 << PMP_SHIFT) {
                reg_t cur_addr = addr + offset;
                auto napot_match = ((cur_addr ^ tor) & mask) == 0;
                auto tor_match = base <= (cur_addr + len - 1) && cur_addr < tor;
                auto match = is_tor ? tor_match : napot_match;
                any_match |= match;
                all_match &= match;
            }
            if(any_match) {
                // If the PMP matches only a strict subset of the access, fail it
                if(!all_match)
                    return false;
                return (hart_if.reg.PRIV == arch::PRIV_M && !(cfg & PMP_L)) || (type == access_type::READ && (cfg & PMP_R)) ||
                       (type == access_type::WRITE && (cfg & PMP_W)) || (type == access_type::FETCH && (cfg & PMP_X));
            }
        }
        base = tor;
    }
    //    constexpr auto pmp_num_regs = 16;
    //    reg_t tor_base = 0;
    //    auto any_active = false;
    //    auto lower_addr = addr >>2;
    //    auto upper_addr = (addr+len-1)>>2;
    //    for (size_t i = 0; i < pmp_num_regs; i++) {
    //        uint8_t cfg = csr[pmpcfg0+(i/4)]>>(i%4);
    //        uint8_t cfg_next = i==(pmp_num_regs-1)? 0 : csr[pmpcfg0+((i+1)/4)]>>((i+1)%4);
    //        auto pmpaddr = csr[pmpaddr0+i];
    //        if (cfg & PMP_A) {
    //            any_active=true;
    //            auto is_tor = bit_sub<3, 2>(cfg) == PMP_TOR;
    //            auto is_napot = bit_sub<4, 1>(cfg) && bit_sub<3, 2>(cfg_next)!= PMP_TOR;
    //            if(is_napot) {
    //                reg_t mask = bit_sub<3, 1>(cfg)?~( pmpaddr & ~(pmpaddr + 1)): 0x3fffffff;
    //                auto mpmpaddr = pmpaddr & mask;
    //                if((lower_addr&mask) == mpmpaddr && (upper_addr&mask)==mpmpaddr)
    //                    return  (hart_if.reg.PRIV == PRIV_M && !(cfg & PMP_L)) ||
    //                            (type == access_type::READ && (cfg & PMP_R)) ||
    //                            (type == access_type::WRITE && (cfg & PMP_W)) ||
    //                            (type == access_type::FETCH && (cfg & PMP_X));
    //            } else if(is_tor) {
    //                if(lower_addr>=tor_base && upper_addr<=pmpaddr)
    //                    return  (hart_if.reg.PRIV == PRIV_M && !(cfg & PMP_L)) ||
    //                            (type == access_type::READ && (cfg & PMP_R)) ||
    //                            (type == access_type::WRITE && (cfg & PMP_W)) ||
    //                            (type == access_type::FETCH && (cfg & PMP_X));
    //            }
    //        }
    //        tor_base = pmpaddr;
    //    }
    return !any_active || hart_if.reg.PRIV == arch::PRIV_M;
}

} // namespace mem
} // namespace iss
