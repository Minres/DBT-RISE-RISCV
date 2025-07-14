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
 *******************************************************************************/

// clang-format off
#include <iss/arch/rv32i.h>
#include <iss/arch/rv32imac.h>
#include <iss/arch/rv32gc.h>
#include <iss/arch/rv64i.h>
#include <iss/arch/rv64gc.h>
#include <iss/arch/tgc5c.h>
#include <iss/arch/riscv_hart_m_p.h>
#include <iss/arch/riscv_hart_mu_p.h>
#include <iss/arch/riscv_hart_msu_vp.h>
#include "iss_factory.h"
#include "sc_core_adapter.h"
#include <array>
// clang-format on

namespace iss {
namespace interp {
using namespace sysc;
volatile std::array<bool, 12> riscv_init = {
    iss_factory::instance().register_creator("rv32i|m_p|interp",
                                             [](unsigned gdb_port, void* data) -> iss_factory::base_t {
                                                 auto cc = reinterpret_cast<sysc::riscv_vp::core_complex_if*>(data);
                                                 auto* cpu = new sc_core_adapter<arch::riscv_hart_m_p<arch::rv32i>>(cc);
                                                 return {sysc::sc_cpu_ptr{cpu}, vm_ptr{create(static_cast<arch::rv32i*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("rv32i|mu_p|interp",
                                             [](unsigned gdb_port, void* data) -> iss_factory::base_t {
                                                 auto cc = reinterpret_cast<sysc::riscv_vp::core_complex_if*>(data);
                                                 auto* cpu = new sc_core_adapter<arch::riscv_hart_m_p<arch::rv32i>>(cc);
                                                 return {sysc::sc_cpu_ptr{cpu}, vm_ptr{create(static_cast<arch::rv32i*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("rv32imac|m_p|interp",
                                             [](unsigned gdb_port, void* data) -> iss_factory::base_t {
                                                 auto cc = reinterpret_cast<sysc::riscv_vp::core_complex_if*>(data);
                                                 auto* cpu = new sc_core_adapter<arch::riscv_hart_m_p<arch::rv32imac>>(cc);
                                                 return {sysc::sc_cpu_ptr{cpu},
                                                         vm_ptr{create(static_cast<arch::rv32imac*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("rv32imac|mu_p|interp",
                                             [](unsigned gdb_port, void* data) -> iss_factory::base_t {
                                                 auto cc = reinterpret_cast<sysc::riscv_vp::core_complex_if*>(data);
                                                 auto* cpu = new sc_core_adapter<arch::riscv_hart_mu_p<arch::rv32imac>>(cc);
                                                 return {sysc::sc_cpu_ptr{cpu},
                                                         vm_ptr{create(static_cast<arch::rv32imac*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("rv32gc|m_p|interp",
                                             [](unsigned gdb_port, void* data) -> iss_factory::base_t {
                                                 auto cc = reinterpret_cast<sysc::riscv_vp::core_complex_if*>(data);
                                                 auto* cpu = new sc_core_adapter<arch::riscv_hart_m_p<arch::rv32gc>>(cc);
                                                 return {sysc::sc_cpu_ptr{cpu}, vm_ptr{create(static_cast<arch::rv32gc*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("rv32gc|mu_p|interp",
                                             [](unsigned gdb_port, void* data) -> iss_factory::base_t {
                                                 auto cc = reinterpret_cast<sysc::riscv_vp::core_complex_if*>(data);
                                                 auto* cpu = new sc_core_adapter<arch::riscv_hart_mu_p<arch::rv32gc>>(cc);
                                                 return {sysc::sc_cpu_ptr{cpu}, vm_ptr{create(static_cast<arch::rv32gc*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("rv32gc|msu_vp|interp",
                                             [](unsigned gdb_port, void* data) -> iss_factory::base_t {
                                                 auto cc = reinterpret_cast<sysc::riscv_vp::core_complex_if*>(data);
                                                 auto* cpu = new sc_core_adapter<arch::riscv_hart_mu_p<arch::rv32gc>>(cc);
                                                 return {sysc::sc_cpu_ptr{cpu}, vm_ptr{create(static_cast<arch::rv32gc*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("rv64i|m_p|interp",
                                             [](unsigned gdb_port, void* data) -> iss_factory::base_t {
                                                 auto cc = reinterpret_cast<sysc::riscv_vp::core_complex_if*>(data);
                                                 auto* cpu = new sc_core_adapter<arch::riscv_hart_m_p<arch::rv64i>>(cc);
                                                 return {sysc::sc_cpu_ptr{cpu}, vm_ptr{create(static_cast<arch::rv64i*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("rv64i|mu_p|interp",
                                             [](unsigned gdb_port, void* data) -> iss_factory::base_t {
                                                 auto cc = reinterpret_cast<sysc::riscv_vp::core_complex_if*>(data);
                                                 auto* cpu = new sc_core_adapter<arch::riscv_hart_m_p<arch::rv64i>>(cc);
                                                 return {sysc::sc_cpu_ptr{cpu}, vm_ptr{create(static_cast<arch::rv64i*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("rv64gc|m_p|interp",
                                             [](unsigned gdb_port, void* data) -> iss_factory::base_t {
                                                 auto cc = reinterpret_cast<sysc::riscv_vp::core_complex_if*>(data);
                                                 auto* cpu = new sc_core_adapter<arch::riscv_hart_m_p<arch::rv64gc>>(cc);
                                                 return {sysc::sc_cpu_ptr{cpu}, vm_ptr{create(static_cast<arch::rv64gc*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("tgc5c|m_p|interp",
                                             [](unsigned gdb_port, void* data) -> iss_factory::base_t {
                                                 auto cc = reinterpret_cast<sysc::riscv_vp::core_complex_if*>(data);
                                                 auto* cpu = new sc_core_adapter<arch::riscv_hart_m_p<arch::tgc5c>>(cc);
                                                 return {sysc::sc_cpu_ptr{cpu}, vm_ptr{create(static_cast<arch::tgc5c*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("tgc5c|mu_p|interp", [](unsigned gdb_port, void* data) -> iss_factory::base_t {
        auto cc = reinterpret_cast<sysc::riscv_vp::core_complex_if*>(data);
        auto* cpu = new sc_core_adapter<arch::riscv_hart_mu_p<arch::tgc5c>>(cc);
        return {sysc::sc_cpu_ptr{cpu}, vm_ptr{create(static_cast<arch::tgc5c*>(cpu), gdb_port)}};
    })}; // namespace interp
} // namespace interp
#if defined(WITH_LLVM)
namespace llvm {
using namespace sysc;
volatile std::array<bool, 4> riscv_init = {
    iss_factory::instance().register_creator("rv32imc|m_p|llvm",
                                             [](unsigned gdb_port, void* data) -> iss_factory::base_t {
                                                 auto cc = reinterpret_cast<sysc::riscv_vp::core_complex_if*>(data);
                                                 auto* cpu = new sc_core_adapter<arch::riscv_hart_m_p<arch::rv32imc>>(cc);
                                                 return {sysc::sc_cpu_ptr{cpu}, vm_ptr{create(static_cast<arch::rv32imc*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("rv32imc|mu_p|llvm",
                                             [](unsigned gdb_port, void* data) -> iss_factory::base_t {
                                                 auto cc = reinterpret_cast<sysc::riscv_vp::core_complex_if*>(data);
                                                 auto* cpu = new sc_core_adapter<arch::riscv_hart_mu_p<arch::rv32imc>>(cc);
                                                 return {sysc::sc_cpu_ptr{cpu}, vm_ptr{create(static_cast<arch::rv32imc*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("tgc5c|m_p|llvm",
                                             [](unsigned gdb_port, void* data) -> iss_factory::base_t {
                                                 auto cc = reinterpret_cast<sysc::riscv_vp::core_complex_if*>(data);
                                                 auto* cpu = new sc_core_adapter<arch::riscv_hart_m_p<arch::tgc5c>>(cc);
                                                 return {sysc::sc_cpu_ptr{cpu}, vm_ptr{create(static_cast<arch::tgc5c*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("tgc5c|mu_p|llvm", [](unsigned gdb_port, void* data) -> iss_factory::base_t {
        auto cc = reinterpret_cast<sysc::riscv_vp::core_complex_if*>(data);
        auto* cpu = new sc_core_adapter<arch::riscv_hart_mu_p<arch::tgc5c>>(cc);
        return {sysc::sc_cpu_ptr{cpu}, vm_ptr{create(static_cast<arch::tgc5c*>(cpu), gdb_port)}};
    })};
} // namespace llvm
#endif
#if defined(WITH_ASMJIT)
namespace asmjit {
using namespace sysc;
volatile std::array<bool, 11> riscv_init = {
    iss_factory::instance().register_creator("rv32i|m_p|asmjit",
                                             [](unsigned gdb_port, void* data) -> iss_factory::base_t {
                                                 auto cc = reinterpret_cast<sysc::riscv_vp::core_complex_if*>(data);
                                                 auto* cpu = new sc_core_adapter<arch::riscv_hart_m_p<arch::rv32i>>(cc);
                                                 return {sysc::sc_cpu_ptr{cpu}, vm_ptr{create(static_cast<arch::rv32i*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("rv32i|mu_p|asmjit",
                                             [](unsigned gdb_port, void* data) -> iss_factory::base_t {
                                                 auto cc = reinterpret_cast<sysc::riscv_vp::core_complex_if*>(data);
                                                 auto* cpu = new sc_core_adapter<arch::riscv_hart_mu_p<arch::rv32i>>(cc);
                                                 return {sysc::sc_cpu_ptr{cpu}, vm_ptr{create(static_cast<arch::rv32i*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("rv32imac|m_p|asmjit",
                                             [](unsigned gdb_port, void* data) -> iss_factory::base_t {
                                                 auto cc = reinterpret_cast<sysc::riscv_vp::core_complex_if*>(data);
                                                 auto* cpu = new sc_core_adapter<arch::riscv_hart_m_p<arch::rv32imac>>(cc);
                                                 return {sysc::sc_cpu_ptr{cpu},
                                                         vm_ptr{create(static_cast<arch::rv32imac*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("rv32imac|mu_p|asmjit",
                                             [](unsigned gdb_port, void* data) -> iss_factory::base_t {
                                                 auto cc = reinterpret_cast<sysc::riscv_vp::core_complex_if*>(data);
                                                 auto* cpu = new sc_core_adapter<arch::riscv_hart_mu_p<arch::rv32imac>>(cc);
                                                 return {sysc::sc_cpu_ptr{cpu},
                                                         vm_ptr{create(static_cast<arch::rv32imac*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("rv32gc|m_p|asmjit",
                                             [](unsigned gdb_port, void* data) -> iss_factory::base_t {
                                                 auto cc = reinterpret_cast<sysc::riscv_vp::core_complex_if*>(data);
                                                 auto* cpu = new sc_core_adapter<arch::riscv_hart_m_p<arch::rv32gc>>(cc);
                                                 return {sysc::sc_cpu_ptr{cpu}, vm_ptr{create(static_cast<arch::rv32gc*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("rv32gc|mu_p|asmjit",
                                             [](unsigned gdb_port, void* data) -> iss_factory::base_t {
                                                 auto cc = reinterpret_cast<sysc::riscv_vp::core_complex_if*>(data);
                                                 auto* cpu = new sc_core_adapter<arch::riscv_hart_mu_p<arch::rv32gc>>(cc);
                                                 return {sysc::sc_cpu_ptr{cpu}, vm_ptr{create(static_cast<arch::rv32gc*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("rv32gc|msu_vp|asmjit",
                                             [](unsigned gdb_port, void* data) -> iss_factory::base_t {
                                                 auto cc = reinterpret_cast<sysc::riscv_vp::core_complex_if*>(data);
                                                 auto* cpu = new sc_core_adapter<arch::riscv_hart_msu_vp<arch::rv32gc>>(cc);
                                                 return {sysc::sc_cpu_ptr{cpu}, vm_ptr{create(static_cast<arch::rv32gc*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("rv64i|m_p|asmjit",
                                             [](unsigned gdb_port, void* data) -> iss_factory::base_t {
                                                 auto cc = reinterpret_cast<sysc::riscv_vp::core_complex_if*>(data);
                                                 auto* cpu = new sc_core_adapter<arch::riscv_hart_m_p<arch::rv64i>>(cc);
                                                 return {sysc::sc_cpu_ptr{cpu}, vm_ptr{create(static_cast<arch::rv64i*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("rv64i|mu_p|asmjit",
                                             [](unsigned gdb_port, void* data) -> iss_factory::base_t {
                                                 auto cc = reinterpret_cast<sysc::riscv_vp::core_complex_if*>(data);
                                                 auto* cpu = new sc_core_adapter<arch::riscv_hart_mu_p<arch::rv64i>>(cc);
                                                 return {sysc::sc_cpu_ptr{cpu}, vm_ptr{create(static_cast<arch::rv64i*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("tgc5c|m_p|asmjit",
                                             [](unsigned gdb_port, void* data) -> iss_factory::base_t {
                                                 auto cc = reinterpret_cast<sysc::riscv_vp::core_complex_if*>(data);
                                                 auto* cpu = new sc_core_adapter<arch::riscv_hart_m_p<arch::tgc5c>>(cc);
                                                 return {sysc::sc_cpu_ptr{cpu}, vm_ptr{create(static_cast<arch::tgc5c*>(cpu), gdb_port)}};
                                             }),
    iss_factory::instance().register_creator("tgc5c|mu_p|asmjit", [](unsigned gdb_port, void* data) -> iss_factory::base_t {
        auto cc = reinterpret_cast<sysc::riscv_vp::core_complex_if*>(data);
        auto* cpu = new sc_core_adapter<arch::riscv_hart_mu_p<arch::tgc5c>>(cc);
        return {sysc::sc_cpu_ptr{cpu}, vm_ptr{create(static_cast<arch::tgc5c*>(cpu), gdb_port)}};
    })};
} // namespace asmjit
#endif
} // namespace iss
