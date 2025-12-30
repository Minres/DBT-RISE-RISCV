/*******************************************************************************
 * Copyright (C) 2017-2021 MINRES Technologies GmbH
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

#ifndef _SYSC_CORE_COMPLEX__IF_H_
#define _SYSC_CORE_COMPLEX__IF_H_

#include <iss/vm_types.h>
#include <scc/signal_opt_ports.h>
#include <util/delegate.h>

namespace sysc {
namespace riscv {
struct core_complex_if {

    virtual ~core_complex_if() = default;

    virtual bool read_mem(const iss::addr_t& addr, unsigned length, uint8_t* const data) = 0;

    virtual bool write_mem(const iss::addr_t& addr, unsigned length, const uint8_t* const data) = 0;

    virtual bool read_mem_dbg(const iss::addr_t& addr, unsigned length, uint8_t* const data) = 0;

    virtual bool write_mem_dbg(const iss::addr_t& addr, unsigned length, const uint8_t* const data) = 0;

    virtual void disass_output(uint64_t pc, std::string const& instr) = 0;

    virtual unsigned get_last_bus_cycles() = 0;

    //! Allow quantum keeper handling
    virtual void sync(uint64_t) = 0;

    util::delegate<void(std::function<void(void)>&)> exec_on_sysc;

    virtual char const* hier_name() = 0;

    scc::sc_in_opt<uint64_t> mtime_i{"mtime_i"};
};
} // namespace riscv
} /* namespace sysc */

#endif /* _SYSC_CORE_COMPLEX__IF_H_ */
