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

#ifndef _MEMORY_WITH_HTIF_
#define _MEMORY_WITH_HTIF_

#include "iss/arch/riscv_hart_common.h"
#include "iss/arch/traits.h"
#include "iss/vm_types.h"
#include "memory_if.h"
#include <array>
#include <cstdlib>
#include <util/logging.h>
#include <util/sparse_array.h>

namespace iss {
namespace mem {
template <typename PLAT> struct neumann_memory_with_htif : public memory_elem {
    using this_class = neumann_memory_with_htif<PLAT>;
    using reg_t = typename PLAT::reg_t;

    neumann_memory_with_htif(arch::priv_if<reg_t> hart_if)
    : hart_if(hart_if) {}

    ~neumann_memory_with_htif() = default;

    memory_if get_mem_if() override {
        return memory_if{.rd_mem{util::delegate<rd_mem_func_sig>::from<this_class, &this_class::read_mem>(this)},
                         .wr_mem{util::delegate<wr_mem_func_sig>::from<this_class, &this_class::write_mem>(this)}};
    }

    void set_next(memory_if) override {
        // intentionally left empty, leaf element
    }

private:
    iss::status read_mem(const iss::addr_t& addr, unsigned length, uint8_t* data) {
        assert((addr.type == iss::address_type::PHYSICAL || is_debug(addr.access)) &&
               "Only physical addresses are expected in memory_with_htif");
        mem_type& mem = addr.space == iss::arch::traits<PLAT>::IMEM ? memories[iss::arch::traits<PLAT>::MEM] : memories[addr.space];
        if(mem.is_allocated(addr.val)) {
            const auto& p = mem(addr.val / mem.page_size);
            auto offs = addr.val & mem.page_addr_mask;
            if((offs + length) > mem.page_size) {
                auto first_part = mem.page_size - offs;
                std::copy(p.data() + offs, p.data() + offs + first_part, data);
                const auto& p2 = mem((addr.val / mem.page_size) + 1);
                std::copy(p2.data(), p2.data() + length - first_part, data + first_part);
            } else {
                std::copy(p.data() + offs, p.data() + offs + length, data);
            }
        } else {
            // no allocated page so return randomized data
            for(size_t i = 0; i < length; i++)
                data[i] = std::rand() % 256;
        }
        return iss::Ok;
    }

    iss::status write_mem(const iss::addr_t& addr, unsigned length, uint8_t const* data) {
        assert((addr.type == iss::address_type::PHYSICAL || is_debug(addr.access)) &&
               "Only physical addresses are expected in memory_with_htif");
        mem_type& mem = addr.space == iss::arch::traits<PLAT>::IMEM ? memories[iss::arch::traits<PLAT>::MEM] : memories[addr.space];
        auto& p = mem(addr.val / mem.page_size);
        auto offs = addr.val & mem.page_addr_mask;
        if((offs + length) > mem.page_size) {
            auto first_part = mem.page_size - offs;
            std::copy(data, data + first_part, p.data() + offs);
            auto& p2 = mem((addr.val / mem.page_size) + 1);
            std::copy(data + first_part, data + length, p2.data());
        } else {
            std::copy(data, data + length, p.data() + offs);
        }
        if(addr.val == hart_if.tohost) {
            return hart_if.exec_htif(data, length);
        }
        return iss::Ok;
    }

protected:
    // Currently no type erasure for the sparse_array is available, so all memories
    // have the largest possible size. Memory footprint should still be small as it
    // a sparse array
    using mem_type = util::sparse_array < uint8_t,
          arch::traits<PLAT>::max_mem_size<1ull << 36 ? arch::traits<PLAT>::max_mem_size : (1ull << 36)>;
    std::array<mem_type, arch::traits<PLAT>::mem_sizes.size()> memories{};
    arch::priv_if<reg_t> hart_if;
};
} // namespace mem
} // namespace iss
#endif // _MEMORY_WITH_HTIF_
