/*******************************************************************************
 * Copyright (C) 2017, 2018 MINRES Technologies GmbH
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

#ifndef _ISS_ARCH_DEBUGGER_RISCV_TARGET_ADAPTER_H_
#define _ISS_ARCH_DEBUGGER_RISCV_TARGET_ADAPTER_H_

#include "iss/arch_if.h"
#include <iss/arch/traits.h>
#include <iss/debugger/target_adapter_base.h>
#include <iss/iss.h>

#include <array>
#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif
#include <fmt/format.h>
#include <util/logging.h>

namespace iss {
namespace debugger {

char const* const get_csr_name(unsigned);
constexpr auto csr_offset = 100U;

using namespace iss::arch;
using namespace iss::debugger;

template <typename ARCH> class riscv_target_adapter : public target_adapter_base {
public:
    riscv_target_adapter(server_if* srv, iss::arch_if* core)
    : target_adapter_base(srv)
    , core(core) {}

    /*============== Thread Control ===============================*/

    /* Set generic thread */
    status set_gen_thread(rp_thread_ref& thread) override;

    /* Set control thread */
    status set_ctrl_thread(rp_thread_ref& thread) override;

    /* Get thread status */
    status is_thread_alive(rp_thread_ref& thread, bool& alive) override;

    /*============= Register Access ================================*/

    /* Read all registers. buf is 4-byte aligned and it is in
     target byte order. If  register is not available
     corresponding bytes in avail_buf are 0, otherwise
     avail buf is 1 */
    status read_registers(std::vector<uint8_t>& data, std::vector<uint8_t>& avail) override;

    /* Write all registers. buf is 4-byte aligned and it is in target
     byte order */
    status write_registers(const std::vector<uint8_t>& data) override;

    /* Read one register. buf is 4-byte aligned and it is in
     target byte order. If  register is not available
     corresponding bytes in avail_buf are 0, otherwise
     avail buf is 1 */
    status read_single_register(unsigned int reg_no, std::vector<uint8_t>& buf, std::vector<uint8_t>& avail_buf) override;

    /* Write one register. buf is 4-byte aligned and it is in target byte
     order */
    status write_single_register(unsigned int reg_no, const std::vector<uint8_t>& buf) override;

    /*=================== Memory Access =====================*/

    /* Read memory, buf is 4-bytes aligned and it is in target
     byte order */
    status read_mem(uint64_t addr, std::vector<uint8_t>& buf) override;

    /* Write memory, buf is 4-bytes aligned and it is in target
     byte order */
    status write_mem(uint64_t addr, const std::vector<uint8_t>& buf) override;

    status process_query(unsigned int& mask, const rp_thread_ref& arg, rp_thread_info& info) override;

    status thread_list_query(int first, const rp_thread_ref& arg, std::vector<rp_thread_ref>& result, size_t max_num, size_t& num,
                             bool& done) override;

    status current_thread_query(rp_thread_ref& thread) override;

    status offsets_query(uint64_t& text, uint64_t& data, uint64_t& bss) override;

    status crc_query(uint64_t addr, size_t len, uint32_t& val) override;

    status raw_query(std::string in_buf, std::string& out_buf) override;

    status threadinfo_query(int first, std::string& out_buf) override;

    status threadextrainfo_query(const rp_thread_ref& thread, std::string& out_buf) override;

    status packetsize_query(std::string& out_buf) override;

    status add_break(break_type type, uint64_t addr, unsigned int length) override;

    status remove_break(break_type type, uint64_t addr, unsigned int length) override;

    status resume_from_addr(bool step, int sig, uint64_t addr, rp_thread_ref thread, std::function<void(unsigned)> stop_callback) override;

    status target_xml_query(std::string& out_buf) override;

protected:
    static inline constexpr addr_t map_addr(const addr_t& i) { return i; }
    std::string csr_xml;
    iss::arch_if* core;
    rp_thread_ref thread_idx;
};

template <typename ARCH> typename std::enable_if<iss::arch::traits<ARCH>::FLEN != 0, unsigned>::type get_f0_offset() {
    return iss::arch::traits<ARCH>::F0;
}

template <typename ARCH> typename std::enable_if<iss::arch::traits<ARCH>::FLEN == 0, unsigned>::type get_f0_offset() { return 0; }

template <typename ARCH> status riscv_target_adapter<ARCH>::set_gen_thread(rp_thread_ref& thread) {
    thread_idx = thread;
    return Ok;
}

template <typename ARCH> status riscv_target_adapter<ARCH>::set_ctrl_thread(rp_thread_ref& thread) {
    thread_idx = thread;
    return Ok;
}

template <typename ARCH> status riscv_target_adapter<ARCH>::is_thread_alive(rp_thread_ref& thread, bool& alive) {
    alive = 1;
    return Ok;
}

/* List threads. If first is non-zero then start from the first thread,
 * otherwise start from arg, result points to array of threads to be
 * filled out, result size is number of elements in the result,
 * num points to the actual number of threads found, done is
 * set if all threads are processed.
 */
template <typename ARCH>
status riscv_target_adapter<ARCH>::thread_list_query(int first, const rp_thread_ref& arg, std::vector<rp_thread_ref>& result,
                                                     size_t max_num, size_t& num, bool& done) {
    if(first == 0) {
        result.clear();
        result.push_back(thread_idx);
        num = 1;
        done = true;
        return Ok;
    } else
        return NotSupported;
}

template <typename ARCH> status riscv_target_adapter<ARCH>::current_thread_query(rp_thread_ref& thread) {
    thread = thread_idx;
    return Ok;
}

template <typename ARCH> status riscv_target_adapter<ARCH>::read_registers(std::vector<uint8_t>& data, std::vector<uint8_t>& avail) {
    CPPLOG(TRACE) << "reading target registers";
    data.clear();
    avail.clear();
    const uint8_t* reg_base = core->get_regs_base_ptr();
    auto start_reg = arch::traits<ARCH>::X0;
    for(size_t i = 0; i < 33; ++i) {
        if(i < arch::traits<ARCH>::RFS || i == arch::traits<ARCH>::PC) {
            auto reg_no = i < 32 ? start_reg + i : arch::traits<ARCH>::PC;
            unsigned offset = traits<ARCH>::reg_byte_offsets[reg_no];
            for(size_t j = 0; j < arch::traits<ARCH>::XLEN / 8; ++j) {
                data.push_back(*(reg_base + offset + j));
                avail.push_back(0xff);
            }
        } else {
            for(size_t j = 0; j < arch::traits<ARCH>::XLEN / 8; ++j) {
                data.push_back(0);
                avail.push_back(0);
            }
        }
    }
    if(iss::arch::traits<ARCH>::FLEN > 0) {
        auto fstart_reg = get_f0_offset<ARCH>();
        for(size_t i = 0; i < 32; ++i) {
            auto reg_no = fstart_reg + i;
            auto reg_width = arch::traits<ARCH>::reg_bit_widths[reg_no] / 8;
            unsigned offset = traits<ARCH>::reg_byte_offsets[reg_no];
            for(size_t j = 0; j < reg_width; ++j) {
                data.push_back(*(reg_base + offset + j));
                avail.push_back(0xff);
            }
        }
    }
    return Ok;
}

template <typename ARCH> status riscv_target_adapter<ARCH>::write_registers(const std::vector<uint8_t>& data) {
    auto start_reg = arch::traits<ARCH>::X0;
    auto* reg_base = core->get_regs_base_ptr();
    auto iter = data.data();
    auto iter_end = data.data() + data.size();
    for(size_t i = 0; i < 33 && iter < iter_end; ++i) {
        auto reg_width = arch::traits<ARCH>::XLEN / 8;
        if(i < arch::traits<ARCH>::RFS) {
            auto offset = traits<ARCH>::reg_byte_offsets[start_reg + i];
            std::copy(iter, iter + reg_width, reg_base + offset);
        } else if(i == 32) {
            auto offset = traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::PC];
            std::copy(iter, iter + reg_width, reg_base + offset);
        }
        iter += reg_width;
    }
    if(iss::arch::traits<ARCH>::FLEN > 0) {
        auto fstart_reg = get_f0_offset<ARCH>();
        auto reg_width = arch::traits<ARCH>::FLEN / 8;
        for(size_t i = 0; i < 32 && iter < iter_end; ++i) {
            unsigned offset = traits<ARCH>::reg_byte_offsets[fstart_reg + i];
            std::copy(iter, iter + reg_width, reg_base + offset);
            iter += reg_width;
        }
    }
    return Ok;
}

template <typename ARCH>
status riscv_target_adapter<ARCH>::read_single_register(unsigned int reg_no, std::vector<uint8_t>& data, std::vector<uint8_t>& avail) {
    if(reg_no < csr_offset) {
        // auto reg_size = arch::traits<ARCH>::reg_bit_width(static_cast<typename
        // arch::traits<ARCH>::reg_e>(reg_no))/8;
        auto* reg_base = core->get_regs_base_ptr();
        auto reg_width = arch::traits<ARCH>::reg_bit_widths[reg_no] / 8;
        data.resize(reg_width);
        avail.resize(reg_width);
        auto offset = traits<ARCH>::reg_byte_offsets[reg_no];
        std::copy(reg_base + offset, reg_base + offset + reg_width, data.begin());
        std::fill(avail.begin(), avail.end(), 0xff);
    } else {
        typed_addr_t<iss::address_type::PHYSICAL> a(iss::access_type::DEBUG_READ, traits<ARCH>::CSR, reg_no - csr_offset);
        data.resize(sizeof(typename traits<ARCH>::reg_t));
        avail.resize(sizeof(typename traits<ARCH>::reg_t));
        std::fill(avail.begin(), avail.end(), 0xff);
        core->read(a, data.size(), data.data());
        std::fill(avail.begin(), avail.end(), 0xff);
    }
    return data.size() > 0 ? Ok : Err;
}

template <typename ARCH> status riscv_target_adapter<ARCH>::write_single_register(unsigned int reg_no, const std::vector<uint8_t>& data) {
    if(reg_no < csr_offset) {
        auto* reg_base = core->get_regs_base_ptr();
        auto reg_width = arch::traits<ARCH>::reg_bit_widths[static_cast<typename arch::traits<ARCH>::reg_e>(reg_no)] / 8;
        auto offset = traits<ARCH>::reg_byte_offsets[reg_no];
        std::copy(data.begin(), data.begin() + reg_width, reg_base + offset);
    } else {
        typed_addr_t<iss::address_type::PHYSICAL> a(iss::access_type::DEBUG_WRITE, traits<ARCH>::CSR, reg_no - csr_offset);
        core->write(a, data.size(), data.data());
    }
    return Ok;
}

template <typename ARCH> status riscv_target_adapter<ARCH>::read_mem(uint64_t addr, std::vector<uint8_t>& data) {
    auto a = map_addr({iss::access_type::DEBUG_READ, iss::address_type::VIRTUAL, 0, addr});
    auto f = [&]() -> status { return core->read(a, data.size(), data.data()); };
    return srv->execute_syncronized(f);
}

template <typename ARCH> status riscv_target_adapter<ARCH>::write_mem(uint64_t addr, const std::vector<uint8_t>& data) {
    auto a = map_addr({iss::access_type::DEBUG_WRITE, iss::address_type::VIRTUAL, 0, addr});
    auto f = [&]() -> status { return core->write(a, data.size(), data.data()); };
    return srv->execute_syncronized(f);
}

template <typename ARCH>
status riscv_target_adapter<ARCH>::process_query(unsigned int& mask, const rp_thread_ref& arg, rp_thread_info& info) {
    return NotSupported;
}

template <typename ARCH> status riscv_target_adapter<ARCH>::offsets_query(uint64_t& text, uint64_t& data, uint64_t& bss) {
    text = 0;
    data = 0;
    bss = 0;
    return Ok;
}

template <typename ARCH> status riscv_target_adapter<ARCH>::crc_query(uint64_t addr, size_t len, uint32_t& val) { return NotSupported; }

template <typename ARCH> status riscv_target_adapter<ARCH>::raw_query(std::string in_buf, std::string& out_buf) { return NotSupported; }

template <typename ARCH> status riscv_target_adapter<ARCH>::threadinfo_query(int first, std::string& out_buf) {
    if(first) {
        out_buf = fmt::format("m{:x}", thread_idx.val);
    } else {
        out_buf = "l";
    }
    return Ok;
}

template <typename ARCH> status riscv_target_adapter<ARCH>::threadextrainfo_query(const rp_thread_ref& thread, std::string& out_buf) {
    std::array<char, 20> buf;
    memset(buf.data(), 0, 20);
    sprintf(buf.data(), "%02x%02x%02x%02x%02x%02x%02x%02x%02x", 'R', 'u', 'n', 'n', 'a', 'b', 'l', 'e', 0);
    out_buf = buf.data();
    return Ok;
}

template <typename ARCH> status riscv_target_adapter<ARCH>::packetsize_query(std::string& out_buf) {
    out_buf = "PacketSize=1000";
    return Ok;
}

template <typename ARCH> status riscv_target_adapter<ARCH>::add_break(break_type type, uint64_t addr, unsigned int length) {
    switch(type) {
    default:
        return Err;
    case SW_EXEC:
    case HW_EXEC: {
        auto saddr = map_addr({iss::access_type::FETCH, iss::address_type::PHYSICAL, 0, addr});
        auto eaddr = map_addr({iss::access_type::FETCH, iss::address_type::PHYSICAL, 0, addr + length});
        target_adapter_base::bp_lut.addEntry(++target_adapter_base::bp_count, saddr.val, eaddr.val - saddr.val);
        CPPLOG(TRACE) << "Adding breakpoint with handle " << target_adapter_base::bp_count << " for addr 0x" << std::hex << saddr.val
                      << std::dec;
        CPPLOG(TRACE) << "Now having " << target_adapter_base::bp_lut.size() << " breakpoints";
        return Ok;
    }
    }
}

template <typename ARCH> status riscv_target_adapter<ARCH>::remove_break(break_type type, uint64_t addr, unsigned int length) {
    switch(type) {
    default:
        return Err;
    case SW_EXEC:
    case HW_EXEC: {
        auto saddr = map_addr({iss::access_type::FETCH, iss::address_type::PHYSICAL, 0, addr});
        unsigned handle = target_adapter_base::bp_lut.getEntry(saddr.val);
        if(handle) {
            CPPLOG(TRACE) << "Removing breakpoint with handle " << handle << " for addr 0x" << std::hex << saddr.val << std::dec;
            // TODO: check length of addr range
            target_adapter_base::bp_lut.removeEntry(handle);
            CPPLOG(TRACE) << "Now having " << target_adapter_base::bp_lut.size() << " breakpoints";
            return Ok;
        }
        CPPLOG(TRACE) << "Now having " << target_adapter_base::bp_lut.size() << " breakpoints";
        return Err;
    }
    }
}

template <typename ARCH>
status riscv_target_adapter<ARCH>::resume_from_addr(bool step, int sig, uint64_t addr, rp_thread_ref thread,
                                                    std::function<void(unsigned)> stop_callback) {
    auto* reg_base = core->get_regs_base_ptr();
    auto reg_width = arch::traits<ARCH>::reg_bit_widths[arch::traits<ARCH>::PC] / 8;
    auto offset = traits<ARCH>::reg_byte_offsets[arch::traits<ARCH>::PC];
    const uint8_t* iter = reinterpret_cast<const uint8_t*>(&addr);
    std::copy(iter, iter + reg_width, reg_base);
    return resume_from_current(step, sig, thread, stop_callback);
}

template <typename ARCH> status riscv_target_adapter<ARCH>::target_xml_query(std::string& out_buf) {
    if(!csr_xml.size()) {
        std::ostringstream oss;
        oss << "<?xml version=\"1.0\"?><!DOCTYPE feature SYSTEM \"gdb-target.dtd\"><target version=\"1.0\">\n";
        if(iss::arch::traits<ARCH>::XLEN == 32)
            oss << "<architecture>riscv:rv32</architecture>\n";
        else if(iss::arch::traits<ARCH>::XLEN == 64)
            oss << "  <architectureriscv:rv64</architecture>\n";
        oss << "  <feature name=\"org.gnu.gdb.riscv.cpu\">\n";
        auto reg_base_num = iss::arch::traits<ARCH>::X0;
        for(auto i = 0U; i < iss::arch::traits<ARCH>::RFS; ++i) {
            oss << "    <reg name=\"x" << i << "\" bitsize=\"" << iss::arch::traits<ARCH>::reg_bit_widths[reg_base_num + i]
                << "\" type=\"int\" regnum=\"" << i << "\"/>\n";
        }
        oss << "    <reg name=\"pc\" bitsize=\"" << iss::arch::traits<ARCH>::reg_bit_widths[iss::arch::traits<ARCH>::PC]
            << "\" type=\"code_ptr\" regnum=\"" << 32U << "\"/>\n";
        oss << "  </feature>\n";
        if(iss::arch::traits<ARCH>::FLEN > 0) {
            oss << "  <feature name=\"org.gnu.gdb.riscv.fpu\">\n";
            auto reg_base_num = get_f0_offset<ARCH>();
            auto type = iss::arch::traits<ARCH>::FLEN == 32 ? "ieee_single" : "riscv_double";
            for(auto i = 0U; i < 32; ++i) {
                oss << "    <reg name=\"f" << i << "\" bitsize=\"" << iss::arch::traits<ARCH>::reg_bit_widths[reg_base_num + i]
                    << "\" type=\"" << type << "\" regnum=\"" << i + 33 << "\"/>\n";
            }
            oss << "    <reg name=\"fcsr\" bitsize=\"" << iss::arch::traits<ARCH>::XLEN << "\" regnum=\"103\" type int/>\n";
            oss << "    <reg name=\"fflags\" bitsize=\"" << iss::arch::traits<ARCH>::XLEN << "\" regnum=\"101\" type int/>\n";
            oss << "    <reg name=\"frm\" bitsize=\"" << iss::arch::traits<ARCH>::XLEN << "\" regnum=\"102\" type int/>\n";
            oss << "  </feature>\n";
        }
        oss << "  <feature name=\"org.gnu.gdb.riscv.csr\">\n";
        std::vector<uint8_t> data;
        std::vector<uint8_t> avail;
        data.resize(sizeof(typename traits<ARCH>::reg_t));
        avail.resize(sizeof(typename traits<ARCH>::reg_t));
        for(auto i = 0U; i < 4096; ++i) {
            typed_addr_t<iss::address_type::PHYSICAL> a(iss::access_type::DEBUG_READ, traits<ARCH>::CSR, i);
            std::fill(avail.begin(), avail.end(), 0xff);
            auto res = core->read(a, data.size(), data.data());
            if(res == iss::Ok) {
                oss << "    <reg name=\"" << get_csr_name(i) << "\" bitsize=\"" << iss::arch::traits<ARCH>::XLEN
                    << "\"  type=\"int\" regnum=\"" << (i + csr_offset) << "\"/>\n";
            }
        }
        oss << "  </feature>\n";
        oss << "</target>\n";
        csr_xml = oss.str();
    }
    out_buf = csr_xml;
    return Ok;
}
} // namespace debugger
} // namespace iss

#endif /* _ISS_DEBUGGER_RISCV_TARGET_ADAPTER_H_ */
