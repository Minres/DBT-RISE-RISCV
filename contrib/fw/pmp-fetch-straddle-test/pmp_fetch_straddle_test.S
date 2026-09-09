// PMP sector iteration must cover every 4-byte sector an access touches.
//
// pmp_check walked sectors with `for(offset = 0; offset < len; offset += 4)`,
// which visits only floor(len/4) sectors starting at addr. When addr is not
// 4-byte aligned the final touched sector is never examined.
//
// Instruction fetch is the reachable case: the hart rejects misaligned data
// accesses with MISALIGNED_LOAD before they reach PMP, but fetch_ins always
// requests 4 bytes and the fetch alignment is 2 on a compressed ISA, so a
// 4-byte instruction at PC%4==2 spans two sectors.
//
// Layout at straddle_base (4-byte aligned):
//   base+0..base+1  c.nop                 sector base+0
//   base+2..base+5  4-byte j              sectors base+0 and base+4
// Entry 0 is an NA4 deny region covering only sector base+4, so the fetch at
// base+2 partially overlaps it. Partial overlap must be denied.
//
// Buggy: only sector base+0 is checked, no entry matches, the M-mode
//        fall-through allows the fetch, the jump runs -> FAIL.
// Fixed: sector base+4 matches, the overlap is partial -> denied -> PASS.
//
// Pass: the straddling fetch is denied and reports an instruction access fault -> j . (exit 0)
// Fail: any CSR trap, the fetch succeeds, or the cause is wrong -> semihosting SYS_EXIT (exit 2)

#include "pmp_test_common.h"

.section .text
.globl _start
_start:
    // A CSR trap here means PMP is unavailable -> FAIL
    la   t0, csr_trap
    csrw mtvec, t0

    // pmpaddr0 = (straddle_base + 4) >> 2: NA4 over the upper sector only
    la   t5, straddle_base
    addi t1, t5, 4
    srli t1, t1, 2
    csrw pmpaddr0, t1

    // pmpcfg0 byte 0 = 0x90 = NA4 (0x10) | L (0x80), no R/W/X
    li   t1, 0x90
    csrw pmpcfg0, t1

    la   t0, pmp_fault
    csrw mtvec, t0

    jr   t5             // enter at straddle_base, which is 4-byte aligned

trap_entry csr_trap
    semihosting_fail    // CSR trapped: PMP unavailable -> FAIL

trap_entry pmp_fault
    // A denied fetch is an instruction access fault, not a load access fault.
    csrr t1, mcause
    li   t2, 1          // RV_CAUSE_FETCH_ACCESS
    bne  t1, t2, fetch_allowed
    j    .              // straddling fetch denied as expected -> PASS

fetch_allowed:
    semihosting_fail    // the straddling fetch was allowed -> FAIL

    .align 2            // 4-byte align: straddle_base % 4 == 0
straddle_base:
    c.nop               // 2 bytes, leaves PC at base+2
    .option push
    .option norvc       // force a 4-byte jump so the fetch spans two sectors
    j    fetch_allowed  // occupies base+2..base+5
    .option pop
