/*******************************************************************************
 * Copyright (C) 2023 MINRES Technologies GmbH
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

#ifndef _RISCV_HART_M_P_WT_CACHE_H
#define _RISCV_HART_M_P_WT_CACHE_H

#include <iss/vm_types.h>
#include <map>
#include <memory>
#include <util/ities.h>
#include <vector>

namespace iss {
namespace arch {
namespace cache {

enum class state { INVALID, VALID };
struct line {
    uint64_t tag_addr{0};
    state st{state::INVALID};
    std::vector<uint8_t> data;
    line(unsigned line_sz)
    : data(line_sz) {}
};
struct set {
    std::vector<line> ways;
    set(unsigned ways_count, line const& l)
    : ways(ways_count, l) {}
};
struct cache {
    std::vector<set> sets;

    cache(unsigned size, unsigned line_sz, unsigned ways) {
        line const ref_line{line_sz};
        set const ref_set{ways, ref_line};
        sets.resize(size / (ways * line_sz), ref_set);
    }
};

struct wt_policy {
    bool is_cacheline_hit(cache& c);
};
} // namespace cache

// write thru, allocate on read, direct mapped or set-associative with round-robin replacement policy
template <typename BASE> class wt_cache : public BASE {
public:
    using base_class = BASE;
    using this_class = wt_cache<BASE>;
    using reg_t = typename BASE::reg_t;
    using mem_read_f = typename BASE::mem_read_f;
    using mem_write_f = typename BASE::mem_write_f;
    using phys_addr_t = typename BASE::phys_addr_t;

    wt_cache(feature_config cfg = feature_config{});
    virtual ~wt_cache() = default;

    unsigned size{4096};
    unsigned line_sz{64};
    unsigned ways{1};
    uint64_t io_address{0xf0000000};
    uint64_t io_addr_mask{0xf0000000};

protected:
    iss::status read_cache(phys_addr_t addr, unsigned, uint8_t* const);
    iss::status write_cache(phys_addr_t addr, unsigned, uint8_t const* const);
    std::function<mem_read_f> cache_mem_rd_delegate;
    std::function<mem_write_f> cache_mem_wr_delegate;
    std::unique_ptr<cache::cache> dcache_ptr;
    std::unique_ptr<cache::cache> icache_ptr;
    size_t get_way_select() { return 0; }
};

template <typename BASE>
inline wt_cache<BASE>::wt_cache(feature_config cfg)
: BASE(cfg)
, io_address{cfg.io_address}
, io_addr_mask{cfg.io_addr_mask} {
    auto cb = base_class::replace_mem_access(
        [this](phys_addr_t a, unsigned l, uint8_t* const d) -> iss::status { return read_cache(a, l, d); },
        [this](phys_addr_t a, unsigned l, uint8_t const* const d) -> iss::status { return write_cache(a, l, d); });
    cache_mem_rd_delegate = cb.first;
    cache_mem_wr_delegate = cb.second;
}

template <typename BASE> iss::status iss::arch::wt_cache<BASE>::read_cache(phys_addr_t a, unsigned l, uint8_t* const d) {
    if(!icache_ptr) {
        icache_ptr.reset(new cache::cache(size, line_sz, ways));
        dcache_ptr.reset(new cache::cache(size, line_sz, ways));
    }
    if((a.access & iss::access_type::FETCH) == iss::access_type::FETCH || (a.val & io_addr_mask) != io_address) {
        auto set_addr = (a.val & (size - 1)) >> util::ilog2(line_sz * ways);
        auto tag_addr = a.val >> util::ilog2(line_sz);
        auto& set = (is_fetch(a.access) ? icache_ptr : dcache_ptr)->sets[set_addr];
        for(auto& cl : set.ways) {
            if(cl.st == cache::state::VALID && cl.tag_addr == tag_addr) {
                auto start_addr = a.val & (line_sz - 1);
                for(auto i = 0U; i < l; ++i)
                    d[i] = cl.data[start_addr + i];
                return iss::Ok;
            }
        }
        auto& cl = set.ways[get_way_select()];
        phys_addr_t cl_addr{a};
        cl_addr.val = tag_addr << util::ilog2(line_sz);
        cache_mem_rd_delegate(cl_addr, line_sz, cl.data.data());
        cl.tag_addr = tag_addr;
        cl.st = cache::state::VALID;
        auto start_addr = a.val & (line_sz - 1);
        for(auto i = 0U; i < l; ++i)
            d[i] = cl.data[start_addr + i];
        return iss::Ok;
    } else
        return cache_mem_rd_delegate(a, l, d);
}

template <typename BASE> iss::status iss::arch::wt_cache<BASE>::write_cache(phys_addr_t a, unsigned l, const uint8_t* const d) {
    if(!dcache_ptr)
        dcache_ptr.reset(new cache::cache(size, line_sz, ways));
    auto res = cache_mem_wr_delegate(a, l, d);
    if(res == iss::Ok && ((a.val & io_addr_mask) != io_address)) {
        auto set_addr = (a.val & (size - 1)) >> util::ilog2(line_sz * ways);
        auto tag_addr = a.val >> util::ilog2(line_sz);
        auto& set = dcache_ptr->sets[set_addr];
        for(auto& cl : set.ways) {
            if(cl.st == cache::state::VALID && cl.tag_addr == tag_addr) {
                auto start_addr = a.val & (line_sz - 1);
                for(auto i = 0U; i < l; ++i)
                    cl.data[start_addr + i] = d[i];
                break;
            }
        }
    }
    return res;
}

} // namespace arch
} // namespace iss

#endif /* _RISCV_HART_M_P_H */
