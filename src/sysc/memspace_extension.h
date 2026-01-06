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
 *******************************************************************************/

#ifndef _SYSC_MEMSPACE_EXTENSION_H_
#define _SYSC_MEMSPACE_EXTENSION_H_
#include <cstdint>
#include <iss/vm_types.h>
#include <limits>
#include <tlm>
namespace sysc {
namespace memspace {
// the same enum as the mem_type_e in CORENAME.h
enum class common : uint32_t { MEM, FENCE, RES, CSR, IMEM = std::numeric_limits<decltype(iss::addr_t::space)>::max() };

template <typename MEMSPACE = common> class tlm_memspace_extension : public tlm::tlm_extension<tlm_memspace_extension<MEMSPACE>> {
public:
    tlm_memspace_extension(MEMSPACE space)
    : space(space) {}

    tlm::tlm_extension_base* clone() const { return new tlm_memspace_extension(this->space); }

    void copy_from(tlm::tlm_extension_base const& other) { *this = static_cast<const tlm_memspace_extension&>(other); }
    MEMSPACE get_space() const { return space; }

private:
    MEMSPACE space;
};
} // namespace memspace
} // namespace sysc

#endif /*_SYSC_MEMSPACE_EXTENSION_H_*/