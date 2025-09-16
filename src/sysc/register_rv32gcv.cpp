/*******************************************************************************
 * Copyright (C) 2024 MINRES Technologies GmbH
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
// clang-format off
#include <sysc/iss_factory.h>
#include <iss/arch/rv32gcv.h>
#include <iss/arch/riscv_hart_m_p.h>
#include <iss/arch/riscv_hart_mu_p.h>
#include <iss/arch/riscv_hart_msu_vp.h>
#include <sysc/sc_core_adapter.h>
#include <sysc/core_complex.h>
#include <array>

namespace iss {
namespace interp {
using namespace sysc;

volatile std::array<bool, 2 + 1> rv32gcv_init = {
        iss_factory::instance().register_creator("rv32gcv_msu:interp", [](unsigned gdb_port, void* data) -> iss_factory::base_t {
            auto* cc = reinterpret_cast<sysc::riscv::core_complex_if*>(data);
            auto* cpu = new sc_core_adapter<arch::riscv_hart_msu_vp<arch::rv32gcv>>(cc);
            return {sysc::core_ptr{cpu}, vm_ptr{create(static_cast<arch::rv32gcv*>(cpu), gdb_port)}};
        }),
        iss_factory::instance().register_creator("rv32gcv_m:interp", [](unsigned gdb_port, void* data) -> iss_factory::base_t {
            auto* cc = reinterpret_cast<sysc::riscv::core_complex_if*>(data);
            auto* cpu = new sc_core_adapter<arch::riscv_hart_m_p<arch::rv32gcv>>(cc);
            return {sysc::core_ptr{cpu}, vm_ptr{create(static_cast<arch::rv32gcv*>(cpu), gdb_port)}};
        }),
        iss_factory::instance().register_creator("rv32gcv_mu:interp", [](unsigned gdb_port, void* data) -> iss_factory::base_t {
            auto* cc = reinterpret_cast<sysc::riscv::core_complex_if*>(data);
            auto* cpu = new sc_core_adapter<arch::riscv_hart_mu_p<arch::rv32gcv>>(cc);
            return {sysc::core_ptr{cpu}, vm_ptr{create(static_cast<arch::rv32gcv*>(cpu), gdb_port)}};
        })
};
}
#if defined(WITH_LLVM)
namespace llvm {
using namespace sysc;
volatile std::array<bool, 2> rv32gcv_init = {
        iss_factory::instance().register_creator("rv32gcv_m:llvm", [](unsigned gdb_port, void* data) -> iss_factory::base_t {
            auto* cc = reinterpret_cast<sysc::riscv::core_complex_if*>(data);
            auto* cpu = new sc_core_adapter<arch::riscv_hart_m_p<arch::rv32gcv>>(cc);
            return {sysc::core_ptr{cpu}, vm_ptr{create(static_cast<arch::rv32gcv*>(cpu), gdb_port)}};
        }),
        iss_factory::instance().register_creator("rv32gcv_mu:llvm", [](unsigned gdb_port, void* data) -> iss_factory::base_t {
            auto* cc = reinterpret_cast<sysc::riscv::core_complex_if*>(data);
            auto* cpu = new sc_core_adapter<arch::riscv_hart_mu_p<arch::rv32gcv>>(cc);
            return {sysc::core_ptr{cpu}, vm_ptr{create(static_cast<arch::rv32gcv*>(cpu), gdb_port)}};
        })
};
}
#endif
#if defined(WITH_TCC)
namespace tcc {
using namespace sysc;
volatile std::array<bool, 2> rv32gcv_init = {
        iss_factory::instance().register_creator("rv32gcv_m:tcc", [](unsigned gdb_port, void* data) -> iss_factory::base_t {
            auto* cc = reinterpret_cast<sysc::riscv::core_complex_if*>(data);
            auto* cpu = new sc_core_adapter<arch::riscv_hart_m_p<arch::rv32gcv>>(cc);
            return {sysc::core_ptr{cpu}, vm_ptr{create(static_cast<arch::rv32gcv*>(cpu), gdb_port)}};
        }),
        iss_factory::instance().register_creator("rv32gcv_mu:tcc", [](unsigned gdb_port, void* data) -> iss_factory::base_t {
            auto* cc = reinterpret_cast<sysc::riscv::core_complex_if*>(data);
            auto* cpu = new sc_core_adapter<arch::riscv_hart_mu_p<arch::rv32gcv>>(cc);
            return {sysc::core_ptr{cpu}, vm_ptr{create(static_cast<arch::rv32gcv*>(cpu), gdb_port)}};
        })
};
}
#endif
}
// clang-format on
