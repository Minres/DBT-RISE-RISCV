/*******************************************************************************
 * Copyright (C) 2021 MINRES Technologies GmbH
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
 *******************************************************************************/

#ifndef _ISS_FACTORY_H_
#define _ISS_FACTORY_H_

#include <algorithm>
#include <functional>
#include <iss/iss.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace iss {

using cpu_ptr = std::unique_ptr<iss::arch_if>;
using vm_ptr = std::unique_ptr<iss::vm_if>;

template <typename PLAT> std::tuple<cpu_ptr, vm_ptr> create_cpu(std::string const& backend, unsigned gdb_port) {
    using core_type = typename PLAT::core;
    core_type* lcpu = new PLAT();
    if(backend == "interp")
        return {cpu_ptr{lcpu}, vm_ptr{iss::interp::create(lcpu, gdb_port)}};
#ifdef WITH_LLVM
    if(backend == "llvm")
        return {cpu_ptr{lcpu}, vm_ptr{iss::llvm::create(lcpu, gdb_port)}};
#endif
#ifdef WITH_TCC
    if(backend == "tcc")
        return {cpu_ptr{lcpu}, vm_ptr{iss::tcc::create(lcpu, gdb_port)}};
#endif
    return {nullptr, nullptr};
}

class core_factory {
    using cpu_ptr = std::unique_ptr<iss::arch_if>;
    using vm_ptr = std::unique_ptr<iss::vm_if>;
    using base_t = std::tuple<cpu_ptr, vm_ptr>;
    using create_fn = std::function<base_t(unsigned, void*)>;
    using registry_t = std::unordered_map<std::string, create_fn>;

    registry_t registry;

    core_factory() = default;
    core_factory(const core_factory&) = delete;
    core_factory& operator=(const core_factory&) = delete;

public:
    static core_factory& instance() {
        static core_factory bf;
        return bf;
    }

    bool register_creator(const std::string& className, create_fn const& fn) {
        registry[className] = fn;
        return true;
    }

    base_t create(std::string const& className, unsigned gdb_port = 0, void* init_data = nullptr) const {
        registry_t::const_iterator regEntry = registry.find(className);
        if(regEntry != registry.end())
            return regEntry->second(gdb_port, init_data);
        return {nullptr, nullptr};
    }

    std::vector<std::string> get_names() {
        std::vector<std::string> keys{registry.size()};
        std::transform(std::begin(registry), std::end(registry), std::begin(keys),
                       [](std::pair<std::string, create_fn> const& p) { return p.first; });
        return keys;
    }
};

} // namespace iss

#endif /* _ISS_FACTORY_H_ */
