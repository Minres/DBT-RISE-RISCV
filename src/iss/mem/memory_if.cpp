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
#include <algorithm>

namespace iss {
namespace mem {
void memory_hierarchy::root(memory_elem& e) {
    hierarchy.push_front(&e);
    root_set = true;
    update_chain();
}
void memory_hierarchy::prepend(memory_elem& e) {
    if(root_set)
        hierarchy.insert(hierarchy.begin() + 1, &e);
    else
        hierarchy.push_front(&e);
    update_chain();
}
void memory_hierarchy::append(memory_elem& e) {
    hierarchy.push_back(&e);
    update_chain();
}
void memory_hierarchy::insert_before(memory_elem&) {}
void memory_hierarchy::insert_after(memory_elem&) {}
void memory_hierarchy::replace_last(memory_elem& e) {
    auto old = hierarchy.back();
    auto it = std::find_if(std::begin(owned_elems), std::end(owned_elems),
                           [old](std::unique_ptr<memory_elem> const& p) { return p.get() == old; });
    hierarchy.pop_back();
    if(it != std::end(owned_elems))
        owned_elems.erase(it);
    hierarchy.push_back(&e);
    update_chain();
}
void memory_hierarchy::update_chain() {
    bool tail = false;
    for(size_t i = 1; i < hierarchy.size(); ++i) {
        hierarchy[i - 1]->set_next(hierarchy[i]->get_mem_if());
    }
}

void memory_hierarchy::prepend(std::unique_ptr<memory_elem>&& p) {
    prepend(*p);
    owned_elems.push_back(std::move(p));
}

void memory_hierarchy::append(std::unique_ptr<memory_elem>&& p) {
    append(*p);
    owned_elems.push_back(std::move(p));
}

void memory_hierarchy::insert_before(std::unique_ptr<memory_elem>&& p) {
    insert_before(*p);
    owned_elems.push_back(std::move(p));
}

void memory_hierarchy::insert_after(std::unique_ptr<memory_elem>&& p) {
    insert_after(*p);
    owned_elems.push_back(std::move(p));
}

void memory_hierarchy::replace_last(std::unique_ptr<memory_elem>&& p) {
    replace_last(*p);
    owned_elems.push_back(std::move(p));
}

} // namespace mem
} // namespace iss
