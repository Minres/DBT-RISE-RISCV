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
#ifndef _MSTATUS_TYPE
#define _MSTATUS_TYPE

#include <cstdint>
#include <type_traits>
#include <util/bit_field.h>
#include <util/ities.h>

namespace iss {
namespace arch {

template <class T, class Enable = void> struct status {};
// specialization 32bit
template <typename T> struct status<T, typename std::enable_if<std::is_same<T, uint32_t>::value>::type> {
    static inline unsigned SD(T v) { return bit_sub<63, 1>(v); };
    // Machine mode big endian
    static inline unsigned MBE(T v) { return bit_sub<37, 1>(v); };
    // Supervisor mode big endian
    static inline unsigned SBE(T v) { return bit_sub<36, 1>(v); };
    // value of XLEN for S-mode
    static inline unsigned SXL(T v) { return bit_sub<34, 2>(v); };
    // value of XLEN for U-mode
    static inline unsigned UXL(T v) { return bit_sub<32, 2>(v); };
    // Trap SRET
    static inline unsigned TSR(T v) { return bit_sub<22, 1>(v); };
    // Timeout Wait
    static inline unsigned TW(T v) { return bit_sub<21, 1>(v); };
    // Trap Virtual Memory
    static inline unsigned TVM(T v) { return bit_sub<20, 1>(v); };
    // Make eXecutable Readable
    static inline unsigned MXR(T v) { return bit_sub<19, 1>(v); };
    // permit Supervisor User Memory access
    static inline unsigned SUM(T v) { return bit_sub<18, 1>(v); };
    // Modify PRiVilege
    static inline unsigned MPRV(T v) { return bit_sub<17, 1>(v); };
    // status of additional user-mode extensions and associated state, All off/None dirty or clean, some on/None
    // dirty, some clean/Some dirty
    static inline unsigned XS(T v) { return bit_sub<15, 2>(v); };
    // floating-point unit status Off/Initial/Clean/Dirty
    static inline unsigned FS(T v) { return bit_sub<13, 2>(v); };
    // machine previous privilege
    static inline unsigned MPP(T v) { return bit_sub<11, 2>(v); };
    // supervisor previous privilege
    static inline unsigned SPP(T v) { return bit_sub<8, 1>(v); };
    // previous machine interrupt-enable
    static inline unsigned MPIE(T v) { return bit_sub<7, 1>(v); };
    // previous supervisor interrupt-enable
    static inline unsigned SPIE(T v) { return bit_sub<5, 1>(v); };
    // previous user interrupt-enable
    static inline unsigned UPIE(T v) { return bit_sub<4, 1>(v); };
    // machine interrupt-enable
    static inline unsigned MIE(T v) { return bit_sub<3, 1>(v); };
    // supervisor interrupt-enable
    static inline unsigned SIE(T v) { return bit_sub<1, 1>(v); };
    // user interrupt-enable
    static inline unsigned UIE(T v) { return bit_sub<0, 1>(v); };
};

template <typename T> struct status<T, typename std::enable_if<std::is_same<T, uint64_t>::value>::type> {
public:
    // SD bit is read-only and is set when either the FS or XS bits encode a Dirty state (i.e., SD=((FS==11) OR
    // XS==11)))
    static inline unsigned SD(T v) { return bit_sub<63, 1>(v); };
    // Machine mode big endian
    static inline unsigned MBE(T v) { return bit_sub<37, 1>(v); };
    // Supervisor mode big endian
    static inline unsigned SBE(T v) { return bit_sub<36, 1>(v); };
    // value of XLEN for S-mode
    static inline unsigned SXL(T v) { return bit_sub<34, 2>(v); };
    // value of XLEN for U-mode
    static inline unsigned UXL(T v) { return bit_sub<32, 2>(v); };
    // Trap SRET
    static inline unsigned TSR(T v) { return bit_sub<22, 1>(v); };
    // Timeout Wait
    static inline unsigned TW(T v) { return bit_sub<21, 1>(v); };
    // Trap Virtual Memory
    static inline unsigned TVM(T v) { return bit_sub<20, 1>(v); };
    // Make eXecutable Readable
    static inline unsigned MXR(T v) { return bit_sub<19, 1>(v); };
    // permit Supervisor User Memory access
    static inline unsigned SUM(T v) { return bit_sub<18, 1>(v); };
    // Modify PRiVilege
    static inline unsigned MPRV(T v) { return bit_sub<17, 1>(v); };
    // status of additional user-mode extensions and associated state, All off/None dirty or clean, some on/None
    // dirty, some clean/Some dirty
    static inline unsigned XS(T v) { return bit_sub<15, 2>(v); };
    // floating-point unit status Off/Initial/Clean/Dirty
    static inline unsigned FS(T v) { return bit_sub<13, 2>(v); };
    // machine previous privilege
    static inline unsigned MPP(T v) { return bit_sub<11, 2>(v); };
    // supervisor previous privilege
    static inline unsigned SPP(T v) { return bit_sub<8, 1>(v); };
    // previous machine interrupt-enable
    static inline unsigned MPIE(T v) { return bit_sub<7, 1>(v); };
    // previous supervisor interrupt-enable
    static inline unsigned SPIE(T v) { return bit_sub<5, 1>(v); };
    // previous user interrupt-enable
    static inline unsigned UPIE(T v) { return bit_sub<4, 1>(v); };
    // machine interrupt-enable
    static inline unsigned MIE(T v) { return bit_sub<3, 1>(v); };
    // supervisor interrupt-enable
    static inline unsigned SIE(T v) { return bit_sub<1, 1>(v); };
    // user interrupt-enable
    static inline unsigned UIE(T v) { return bit_sub<0, 1>(v); };
};

// primary template
template <class T, class Enable = void> struct hart_state {};
// specialization 32bit
template <typename T> class hart_state<T, typename std::enable_if<std::is_same<T, uint32_t>::value>::type> {
public:
    BEGIN_BF_DECL(mstatus_t, T);
    // SD bit is read-only and is set when either the FS or XS bits encode a Dirty state (i.e., SD=((FS==11) OR
    // XS==11)))
    BF_FIELD(SD, 31, 1);
    // Trap SRET
    BF_FIELD(TSR, 22, 1);
    // Timeout Wait
    BF_FIELD(TW, 21, 1);
    // Trap Virtual Memory
    BF_FIELD(TVM, 20, 1);
    // Make eXecutable Readable
    BF_FIELD(MXR, 19, 1);
    // permit Supervisor User Memory access
    BF_FIELD(SUM, 18, 1);
    // Modify PRiVilege
    BF_FIELD(MPRV, 17, 1);
    // status of additional user-mode extensions and associated state, All off/None dirty or clean, some on/None
    // dirty, some clean/Some dirty
    BF_FIELD(XS, 15, 2);
    // floating-point unit status Off/Initial/Clean/Dirty
    BF_FIELD(FS, 13, 2);
    // machine previous privilege
    BF_FIELD(MPP, 11, 2);
    // supervisor previous privilege
    BF_FIELD(SPP, 8, 1);
    // previous machine interrupt-enable
    BF_FIELD(MPIE, 7, 1);
    // previous supervisor interrupt-enable
    BF_FIELD(SPIE, 5, 1);
    // previous user interrupt-enable
    BF_FIELD(UPIE, 4, 1);
    // machine interrupt-enable
    BF_FIELD(MIE, 3, 1);
    // supervisor interrupt-enable
    BF_FIELD(SIE, 1, 1);
    // user interrupt-enable
    BF_FIELD(UIE, 0, 1);
    END_BF_DECL();

    mstatus_t mstatus;

    static const T mstatus_reset_val = 0x1800;
};

// specialization 64bit
template <typename T> class hart_state<T, typename std::enable_if<std::is_same<T, uint64_t>::value>::type> {
public:
    BEGIN_BF_DECL(mstatus_t, T);
    // SD bit is read-only and is set when either the FS or XS bits encode a Dirty state (i.e., SD=((FS==11) OR
    // XS==11)))
    BF_FIELD(SD, 63, 1);
    // Machine mode big endian
    BF_FIELD(MBE, 37, 1);
    // Supervisor mode big endian
    BF_FIELD(SBE, 36, 1);
    // value of XLEN for S-mode
    BF_FIELD(SXL, 34, 2);
    // value of XLEN for U-mode
    BF_FIELD(UXL, 32, 2);
    // Trap SRET
    BF_FIELD(TSR, 22, 1);
    // Timeout Wait
    BF_FIELD(TW, 21, 1);
    // Trap Virtual Memory
    BF_FIELD(TVM, 20, 1);
    // Make eXecutable Readable
    BF_FIELD(MXR, 19, 1);
    // permit Supervisor User Memory access
    BF_FIELD(SUM, 18, 1);
    // Modify PRiVilege
    BF_FIELD(MPRV, 17, 1);
    // status of additional user-mode extensions and associated state, All off/None dirty or clean, some on/None
    // dirty, some clean/Some dirty
    BF_FIELD(XS, 15, 2);
    // floating-point unit status Off/Initial/Clean/Dirty
    BF_FIELD(FS, 13, 2);
    // machine previous privilege
    BF_FIELD(MPP, 11, 2);
    // supervisor previous privilege
    BF_FIELD(SPP, 8, 1);
    // previous machine interrupt-enable
    BF_FIELD(MPIE, 7, 1);
    // previous supervisor interrupt-enable
    BF_FIELD(SPIE, 5, 1);
    // previous user interrupt-enable
    BF_FIELD(UPIE, 4, 1);
    // machine interrupt-enable
    BF_FIELD(MIE, 3, 1);
    // supervisor interrupt-enable
    BF_FIELD(SIE, 1, 1);
    // user interrupt-enable
    BF_FIELD(UIE, 0, 1);
    END_BF_DECL();

    mstatus_t mstatus;

    static const T mstatus_reset_val = 0x1800;
};
} // namespace arch
} // namespace iss
#endif // _MSTATUS_TYPE