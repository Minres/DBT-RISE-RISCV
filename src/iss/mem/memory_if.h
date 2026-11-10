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

#ifndef _MEMORY_MEMORY_IF_
#define _MEMORY_MEMORY_IF_

#include "iss/checkpoint_if.h"
#include "iss/vm_types.h"
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <util/delegate.h>
#include <utility>
#include <vector>

namespace iss {
namespace mem {

using rd_mem_func_sig = iss::status(const addr_t& addr, unsigned length, uint8_t* data);
using wr_mem_func_sig = iss::status(const addr_t& addr, unsigned length, uint8_t const* data);

struct memory_if {
    util::delegate<rd_mem_func_sig> rd_mem;
    util::delegate<wr_mem_func_sig> wr_mem;
};

// The referenced state must provide value semantics, including deep copies of owned data.
template <typename... State> struct memory_checkpoint : checkpoint_if {
    explicit memory_checkpoint(State&... state)
    : state(state...) {}

    checkpoint_hndl create() override {
        const auto handle = next_handle;
        snapshots.try_emplace(handle, state);
        ++next_handle;
        return handle;
    }

    bool restore(checkpoint_hndl handle) override {
        auto it = snapshots.find(handle);
        if(it == snapshots.end())
            return false;
        state = it->second;
        return true;
    }

    bool drop(checkpoint_hndl handle) override { return snapshots.erase(handle) != 0; }

private:
    std::tuple<State&...> state;
    checkpoint_hndl next_handle{1};
    std::unordered_map<checkpoint_hndl, std::tuple<State...>> snapshots;
};

struct memory_elem {
    virtual ~memory_elem() = default;
    virtual memory_if get_mem_if() = 0;
    virtual void set_next(memory_if) = 0;
    virtual std::tuple<uint64_t, uint64_t> get_range() { return {0, std::numeric_limits<uint64_t>::max()}; }
    // Stateless elements may retain the default implementation.
    virtual checkpoint_if* get_checkpoint_if() { return nullptr; }
};

struct memory_hierarchy : checkpoint_if {
    void root(memory_elem&);
    void prepend(memory_elem&);
    void append(memory_elem&);
    void insert_before_last(memory_elem&);
    void insert_after_first(memory_elem&);
    void replace_last(memory_elem&);
    void prepend(std::unique_ptr<memory_elem>&&);
    void append(std::unique_ptr<memory_elem>&&);
    void insert_before_last(std::unique_ptr<memory_elem>&&);
    void insert_after_first(std::unique_ptr<memory_elem>&&);
    void replace_last(std::unique_ptr<memory_elem>&&);

    // Checkpoints cover all elements except the root, which owns the hart checkpoint.
    // Keep the hierarchy and its elements alive and unchanged until checkpoints are dropped.
    checkpoint_hndl create() override;
    bool restore(checkpoint_hndl) override;
    bool drop(checkpoint_hndl) override;

protected:
    void check_mutable() const;
    void update_chain();
    std::deque<memory_elem*> hierarchy;
    std::vector<std::unique_ptr<memory_elem>> owned_elems;
    bool root_set{false};
    using element_checkpoint = std::pair<checkpoint_if*, checkpoint_hndl>;
    std::unordered_map<checkpoint_hndl, std::vector<element_checkpoint>> snapshots;
    checkpoint_hndl next_handle{1};
};

} // namespace mem
} // namespace iss
#endif
