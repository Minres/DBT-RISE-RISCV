/*
 * riscv_target_adapter.h
 *
 *  Created on: 26.09.2017
 *      Author: eyck
 */

#ifndef _ISS_DEBUGGER_RISCV_TARGET_ADAPTER_H_
#define _ISS_DEBUGGER_RISCV_TARGET_ADAPTER_H_

#include "iss/arch_if.h"
#include <iss/arch/traits.h>
#include <iss/debugger/target_adapter_base.h>
#include <iss/iss.h>

#include <memory>
#include <util/logging.h>

namespace iss {
namespace debugger {
using namespace iss::arch;
using namespace iss::debugger;

template <typename ARCH> class riscv_target_adapter : public target_adapter_base {
public:
    riscv_target_adapter(server_if *srv, iss::arch_if *core)
    : target_adapter_base(srv)
    , core(core) {}

    /*============== Thread Control ===============================*/

    /* Set generic thread */
    status set_gen_thread(rp_thread_ref &thread) override;

    /* Set control thread */
    status set_ctrl_thread(rp_thread_ref &thread) override;

    /* Get thread status */
    status is_thread_alive(rp_thread_ref &thread, bool &alive) override;

    /*============= Register Access ================================*/

    /* Read all registers. buf is 4-byte aligned and it is in
     target byte order. If  register is not available
     corresponding bytes in avail_buf are 0, otherwise
     avail buf is 1 */
    status read_registers(std::vector<uint8_t> &data, std::vector<uint8_t> &avail) override;

    /* Write all registers. buf is 4-byte aligned and it is in target
     byte order */
    status write_registers(const std::vector<uint8_t> &data) override;

    /* Read one register. buf is 4-byte aligned and it is in
     target byte order. If  register is not available
     corresponding bytes in avail_buf are 0, otherwise
     avail buf is 1 */
    status read_single_register(unsigned int reg_no, std::vector<uint8_t> &buf,
                                std::vector<uint8_t> &avail_buf) override;

    /* Write one register. buf is 4-byte aligned and it is in target byte
     order */
    status write_single_register(unsigned int reg_no, const std::vector<uint8_t> &buf) override;

    /*=================== Memory Access =====================*/

    /* Read memory, buf is 4-bytes aligned and it is in target
     byte order */
    status read_mem(uint64_t addr, std::vector<uint8_t> &buf) override;

    /* Write memory, buf is 4-bytes aligned and it is in target
     byte order */
    status write_mem(uint64_t addr, const std::vector<uint8_t> &buf) override;

    status process_query(unsigned int &mask, const rp_thread_ref &arg, rp_thread_info &info) override;

    status thread_list_query(int first, const rp_thread_ref &arg, std::vector<rp_thread_ref> &result, size_t max_num,
                             size_t &num, bool &done) override;

    status current_thread_query(rp_thread_ref &thread) override;

    status offsets_query(uint64_t &text, uint64_t &data, uint64_t &bss) override;

    status crc_query(uint64_t addr, size_t len, uint32_t &val) override;

    status raw_query(std::string in_buf, std::string &out_buf) override;

    status threadinfo_query(int first, std::string &out_buf) override;

    status threadextrainfo_query(const rp_thread_ref &thread, std::string &out_buf) override;

    status packetsize_query(std::string &out_buf) override;

    status add_break(int type, uint64_t addr, unsigned int length) override;

    status remove_break(int type, uint64_t addr, unsigned int length) override;

    status resume_from_addr(bool step, int sig, uint64_t addr, rp_thread_ref thread, std::function<void(unsigned)> stop_callback) override;

    status target_xml_query(std::string& out_buf) override;

protected:
    static inline constexpr addr_t map_addr(const addr_t &i) { return i; }

    iss::arch_if *core;
    rp_thread_ref thread_idx;
};

template <typename ARCH> status riscv_target_adapter<ARCH>::set_gen_thread(rp_thread_ref &thread) {
    thread_idx = thread;
    return Ok;
}

template <typename ARCH> status riscv_target_adapter<ARCH>::set_ctrl_thread(rp_thread_ref &thread) {
    thread_idx = thread;
    return Ok;
}

template <typename ARCH> status riscv_target_adapter<ARCH>::is_thread_alive(rp_thread_ref &thread, bool &alive) {
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
status riscv_target_adapter<ARCH>::thread_list_query(int first, const rp_thread_ref &arg,
                                                     std::vector<rp_thread_ref> &result, size_t max_num, size_t &num,
                                                     bool &done) {
    if (first == 0) {
        result.clear();
        result.push_back(thread_idx);
        num = 1;
        done = true;
        return Ok;
    } else
        return NotSupported;
}

template <typename ARCH> status riscv_target_adapter<ARCH>::current_thread_query(rp_thread_ref &thread) {
    thread = thread_idx;
    return Ok;
}

template <typename ARCH>
status riscv_target_adapter<ARCH>::read_registers(std::vector<uint8_t> &data, std::vector<uint8_t> &avail) {
    LOG(TRACE) << "reading target registers";
    // return idx<0?:;
    data.clear();
    avail.clear();
    const uint8_t *reg_base = core->get_regs_base_ptr();
    for (size_t reg_no = 0; reg_no < arch::traits<ARCH>::NUM_REGS; ++reg_no) {
        auto reg_width = arch::traits<ARCH>::reg_bit_width(static_cast<typename arch::traits<ARCH>::reg_e>(reg_no)) / 8;
        unsigned offset = traits<ARCH>::reg_byte_offset(reg_no);
        for (size_t j = 0; j < reg_width; ++j) {
            data.push_back(*(reg_base + offset + j));
            avail.push_back(0xff);
        }
//        if(arch::traits<ARCH>::XLEN < 64)
//            for(unsigned j=0; j<4; ++j){
//                data.push_back(0);
//                avail.push_back(0xff);
//            }
    }
    // work around fill with F type registers
    if (arch::traits<ARCH>::NUM_REGS < 65) {
        auto reg_width = sizeof(typename arch::traits<ARCH>::reg_t);
        for (size_t reg_no = 0; reg_no < 33; ++reg_no) {
            for (size_t j = 0; j < reg_width; ++j) {
                data.push_back(0x0);
                avail.push_back(0x00);
            }
//            if(arch::traits<ARCH>::XLEN < 64)
//                for(unsigned j=0; j<4; ++j){
//                    data.push_back(0x0);
//                    avail.push_back(0x00);
//                }
        }
    }
    return Ok;
}

template <typename ARCH> status riscv_target_adapter<ARCH>::write_registers(const std::vector<uint8_t> &data) {
    auto reg_count = arch::traits<ARCH>::NUM_REGS;
    auto *reg_base = core->get_regs_base_ptr();
    auto iter = data.data();
    for (size_t reg_no = 0; reg_no < reg_count; ++reg_no) {
        auto reg_width = arch::traits<ARCH>::reg_bit_width(static_cast<typename arch::traits<ARCH>::reg_e>(reg_no)) / 8;
        auto offset = traits<ARCH>::reg_byte_offset(reg_no);
        std::copy(iter, iter + reg_width, reg_base);
        iter += 4;
        reg_base += offset;
    }
    return Ok;
}

template <typename ARCH>
status riscv_target_adapter<ARCH>::read_single_register(unsigned int reg_no, std::vector<uint8_t> &data,
                                                        std::vector<uint8_t> &avail) {
    if (reg_no < 65) {
        // auto reg_size = arch::traits<ARCH>::reg_bit_width(static_cast<typename
        // arch::traits<ARCH>::reg_e>(reg_no))/8;
        auto *reg_base = core->get_regs_base_ptr();
        auto reg_width = arch::traits<ARCH>::reg_bit_width(static_cast<typename arch::traits<ARCH>::reg_e>(reg_no)) / 8;
        data.resize(reg_width);
        avail.resize(reg_width);
        auto offset = traits<ARCH>::reg_byte_offset(reg_no);
        std::copy(reg_base + offset, reg_base + offset + reg_width, data.begin());
        std::fill(avail.begin(), avail.end(), 0xff);
    } else {
        typed_addr_t<iss::PHYSICAL> a(iss::DEBUG_READ, traits<ARCH>::CSR, reg_no - 65);
        data.resize(sizeof(typename traits<ARCH>::reg_t));
        avail.resize(sizeof(typename traits<ARCH>::reg_t));
        std::fill(avail.begin(), avail.end(), 0xff);
        core->read(a, data.size(), data.data());
    }
    return data.size() > 0 ? Ok : Err;
}

template <typename ARCH>
status riscv_target_adapter<ARCH>::write_single_register(unsigned int reg_no, const std::vector<uint8_t> &data) {
    if (reg_no < 65) {
        auto *reg_base = core->get_regs_base_ptr();
        auto reg_width = arch::traits<ARCH>::reg_bit_width(static_cast<typename arch::traits<ARCH>::reg_e>(reg_no)) / 8;
        auto offset = traits<ARCH>::reg_byte_offset(reg_no);
        std::copy(data.begin(), data.begin() + reg_width, reg_base + offset);
    } else {
        typed_addr_t<iss::PHYSICAL> a(iss::DEBUG_WRITE, traits<ARCH>::CSR, reg_no - 65);
        core->write(a, data.size(), data.data());
    }
    return Ok;
}

template <typename ARCH> status riscv_target_adapter<ARCH>::read_mem(uint64_t addr, std::vector<uint8_t> &data) {
    auto a = map_addr({iss::DEBUG_READ, iss::VIRTUAL, 0, addr});
    auto f = [&]() -> status { return core->read(a, data.size(), data.data()); };
    return srv->execute_syncronized(f);
}

template <typename ARCH> status riscv_target_adapter<ARCH>::write_mem(uint64_t addr, const std::vector<uint8_t> &data) {
    auto a = map_addr({iss::DEBUG_READ, iss::VIRTUAL, 0, addr});
    return srv->execute_syncronized(&arch_if::write, core, a, data.size(), data.data());
}

template <typename ARCH>
status riscv_target_adapter<ARCH>::process_query(unsigned int &mask, const rp_thread_ref &arg, rp_thread_info &info) {
    return NotSupported;
}

template <typename ARCH>
status riscv_target_adapter<ARCH>::offsets_query(uint64_t &text, uint64_t &data, uint64_t &bss) {
    text = 0;
    data = 0;
    bss = 0;
    return Ok;
}

template <typename ARCH> status riscv_target_adapter<ARCH>::crc_query(uint64_t addr, size_t len, uint32_t &val) {
    return NotSupported;
}

template <typename ARCH> status riscv_target_adapter<ARCH>::raw_query(std::string in_buf, std::string &out_buf) {
    return NotSupported;
}

template <typename ARCH> status riscv_target_adapter<ARCH>::threadinfo_query(int first, std::string &out_buf) {
    if (first) {
        std::stringstream ss;
        ss << "m" << std::hex << thread_idx.val;
        out_buf = ss.str();
    } else {
        out_buf = "l";
    }
    return Ok;
}

template <typename ARCH>
status riscv_target_adapter<ARCH>::threadextrainfo_query(const rp_thread_ref &thread, std::string &out_buf) {
    char buf[20];
    memset(buf, 0, 20);
    sprintf(buf, "%02x%02x%02x%02x%02x%02x%02x%02x%02x", 'R', 'u', 'n', 'n', 'a', 'b', 'l', 'e', 0);
    out_buf = buf;
    return Ok;
}

template <typename ARCH> status riscv_target_adapter<ARCH>::packetsize_query(std::string &out_buf) {
    out_buf = "PacketSize=1000";
    return Ok;
}

template <typename ARCH> status riscv_target_adapter<ARCH>::add_break(int type, uint64_t addr, unsigned int length) {
    auto saddr = map_addr({iss::CODE, iss::PHYSICAL, addr});
    auto eaddr = map_addr({iss::CODE, iss::PHYSICAL, addr + length});
    target_adapter_base::bp_lut.addEntry(++target_adapter_base::bp_count, saddr.val, eaddr.val - saddr.val);
    LOG(TRACE) << "Adding breakpoint with handle " << target_adapter_base::bp_count << " for addr 0x" << std::hex
               << saddr.val << std::dec;
    LOG(TRACE) << "Now having " << target_adapter_base::bp_lut.size() << " breakpoints";
    return Ok;
}

template <typename ARCH> status riscv_target_adapter<ARCH>::remove_break(int type, uint64_t addr, unsigned int length) {
    auto saddr = map_addr({iss::CODE, iss::PHYSICAL, addr});
    unsigned handle = target_adapter_base::bp_lut.getEntry(saddr.val);
    if (handle) {
        LOG(TRACE) << "Removing breakpoint with handle " << handle << " for addr 0x" << std::hex << saddr.val
                   << std::dec;
        // TODO: check length of addr range
        target_adapter_base::bp_lut.removeEntry(handle);
        LOG(TRACE) << "Now having " << target_adapter_base::bp_lut.size() << " breakpoints";
        return Ok;
    }
    LOG(TRACE) << "Now having " << target_adapter_base::bp_lut.size() << " breakpoints";
    return Err;
}

template <typename ARCH> status riscv_target_adapter<ARCH>::resume_from_addr(bool step, int sig, uint64_t addr, rp_thread_ref thread,
        std::function<void(unsigned)> stop_callback) {
    unsigned reg_no = arch::traits<ARCH>::PC;
    std::vector<uint8_t> data(8);
    *(reinterpret_cast<uint64_t *>(&data[0])) = addr;
    core->set_reg(reg_no, data);
    return resume_from_current(step, sig, thread, stop_callback);
}
template <typename ARCH> status riscv_target_adapter<ARCH>::target_xml_query(std::string& out_buf) {
    const std::string res{
        "<?xml version=\"1.0\"?><!DOCTYPE target SYSTEM \"gdb-target.dtd\">"
"<target><architecture>riscv:rv32</architecture>"
//"  <feature name=\"org.gnu.gdb.riscv.rv32i\">\n"
//"    <reg name=\"x0\"  bitsize=\"32\" group=\"general\"/>\n"
//"    <reg name=\"x1\"  bitsize=\"32\" group=\"general\"/>\n"
//"    <reg name=\"x2\"  bitsize=\"32\" group=\"general\"/>\n"
//"    <reg name=\"x3\"  bitsize=\"32\" group=\"general\"/>\n"
//"    <reg name=\"x4\"  bitsize=\"32\" group=\"general\"/>\n"
//"    <reg name=\"x5\"  bitsize=\"32\" group=\"general\"/>\n"
//"    <reg name=\"x6\"  bitsize=\"32\" group=\"general\"/>\n"
//"    <reg name=\"x7\"  bitsize=\"32\" group=\"general\"/>\n"
//"    <reg name=\"x8\"  bitsize=\"32\" group=\"general\"/>\n"
//"    <reg name=\"x9\"  bitsize=\"32\" group=\"general\"/>\n"
//"    <reg name=\"x10\" bitsize=\"32\" group=\"general\"/>\n"
//"    <reg name=\"x11\" bitsize=\"32\" group=\"general\"/>\n"
//"    <reg name=\"x12\" bitsize=\"32\" group=\"general\"/>\n"
//"    <reg name=\"x13\" bitsize=\"32\" group=\"general\"/>\n"
//"    <reg name=\"x14\" bitsize=\"32\" group=\"general\"/>\n"
//"    <reg name=\"x15\" bitsize=\"32\" group=\"general\"/>\n"
//"    <reg name=\"x16\" bitsize=\"32\" group=\"general\"/>\n"
//"    <reg name=\"x17\" bitsize=\"32\" group=\"general\"/>\n"
//"    <reg name=\"x18\" bitsize=\"32\" group=\"general\"/>\n"
//"    <reg name=\"x19\" bitsize=\"32\" group=\"general\"/>\n"
//"    <reg name=\"x20\" bitsize=\"32\" group=\"general\"/>\n"
//"    <reg name=\"x21\" bitsize=\"32\" group=\"general\"/>\n"
//"    <reg name=\"x22\" bitsize=\"32\" group=\"general\"/>\n"
//"    <reg name=\"x23\" bitsize=\"32\" group=\"general\"/>\n"
//"    <reg name=\"x24\" bitsize=\"32\" group=\"general\"/>\n"
//"    <reg name=\"x25\" bitsize=\"32\" group=\"general\"/>\n"
//"    <reg name=\"x26\" bitsize=\"32\" group=\"general\"/>\n"
//"    <reg name=\"x27\" bitsize=\"32\" group=\"general\"/>\n"
//"    <reg name=\"x28\" bitsize=\"32\" group=\"general\"/>\n"
//"    <reg name=\"x29\" bitsize=\"32\" group=\"general\"/>\n"
//"    <reg name=\"x30\" bitsize=\"32\" group=\"general\"/>\n"
//"    <reg name=\"x31\" bitsize=\"32\" group=\"general\"/>\n"
//"  </feature>\n"
"</target>"};
    out_buf=res;
    return Ok;
}

/*
 *
<?xml version="1.0"?>
<!DOCTYPE target SYSTEM "gdb-target.dtd">
<target>
  <architecture>riscv:rv32</architecture>

  <feature name="org.gnu.gdb.riscv.rv32i">
    <reg name="x0"  bitsize="32" group="general"/>
    <reg name="x1"  bitsize="32" group="general"/>
    <reg name="x2"  bitsize="32" group="general"/>
    <reg name="x3"  bitsize="32" group="general"/>
    <reg name="x4"  bitsize="32" group="general"/>
    <reg name="x5"  bitsize="32" group="general"/>
    <reg name="x6"  bitsize="32" group="general"/>
    <reg name="x7"  bitsize="32" group="general"/>
    <reg name="x8"  bitsize="32" group="general"/>
    <reg name="x9"  bitsize="32" group="general"/>
    <reg name="x10" bitsize="32" group="general"/>
    <reg name="x11" bitsize="32" group="general"/>
    <reg name="x12" bitsize="32" group="general"/>
    <reg name="x13" bitsize="32" group="general"/>
    <reg name="x14" bitsize="32" group="general"/>
    <reg name="x15" bitsize="32" group="general"/>
    <reg name="x16" bitsize="32" group="general"/>
    <reg name="x17" bitsize="32" group="general"/>
    <reg name="x18" bitsize="32" group="general"/>
    <reg name="x19" bitsize="32" group="general"/>
    <reg name="x20" bitsize="32" group="general"/>
    <reg name="x21" bitsize="32" group="general"/>
    <reg name="x22" bitsize="32" group="general"/>
    <reg name="x23" bitsize="32" group="general"/>
    <reg name="x24" bitsize="32" group="general"/>
    <reg name="x25" bitsize="32" group="general"/>
    <reg name="x26" bitsize="32" group="general"/>
    <reg name="x27" bitsize="32" group="general"/>
    <reg name="x28" bitsize="32" group="general"/>
    <reg name="x29" bitsize="32" group="general"/>
    <reg name="x30" bitsize="32" group="general"/>
    <reg name="x31" bitsize="32" group="general"/>
  </feature>

</target>

 */
}
}

#endif /* _ISS_DEBUGGER_RISCV_TARGET_ADAPTER_H_ */
