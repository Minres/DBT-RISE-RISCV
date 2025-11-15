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

#include "iss/vm_types.h"
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <util/delegate.h>
#include <vector>

namespace iss {
namespace mem {

using rd_mem_func_sig = iss::status(iss::access_type, uint32_t space, uint64_t addr, unsigned length, uint8_t* data);
using wr_mem_func_sig = iss::status(iss::access_type, uint32_t space, uint64_t addr, unsigned length, uint8_t const* data);

struct memory_if {
    util::delegate<rd_mem_func_sig> rd_mem;
    util::delegate<wr_mem_func_sig> wr_mem;
};

struct memory_elem {
    virtual ~memory_elem() = default;
    virtual memory_if get_mem_if() = 0;
    virtual void set_next(memory_if) = 0;
    virtual std::tuple<uint64_t, uint64_t> get_range() { return {0, std::numeric_limits<uint64_t>::max()}; }
};

struct memory_hierarchy {
    void root(memory_elem&);
    void prepend(memory_elem&);
    void append(memory_elem&);
    void insert_before(memory_elem&);
    void insert_after(memory_elem&);
    void replace_last(memory_elem&);
    void prepend(std::unique_ptr<memory_elem>&&);
    void append(std::unique_ptr<memory_elem>&&);
    void insert_before(std::unique_ptr<memory_elem>&&);
    void insert_after(std::unique_ptr<memory_elem>&&);
    void replace_last(std::unique_ptr<memory_elem>&&);

protected:
    void update_chain();
    std::deque<memory_elem*> hierarchy;
    std::vector<std::unique_ptr<memory_elem>> owned_elems;
    bool root_set{false};
};

} // namespace mem
} // namespace iss
#endif
