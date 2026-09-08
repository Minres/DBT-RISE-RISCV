// PMP execute-permission enforcement on instruction fetch.
//
// Instruction fetches use the IMEM address space rather than MEM, and no other
// PMP test exercises the fetch path or the X permission bit at all. This test
// pins that behaviour so the address-space guard in pmp.h read_mem cannot be
// narrowed to MEM alone without a test failing.
//
// Entry 0 is an NA4 region covering the 4 bytes at target, locked and with no
// X permission. L=1 is required because pmp_check grants M-mode unconditionally
// when L=0.
//
// Pass: fetch of target is denied -> fault fires -> j . (exit 0)
// Fail: any CSR trap, OR the fetch succeeds -> semihosting SYS_EXIT (exit 2)

#include "pmp_test_common.h"

.section .text
.globl _start
_start:
    // A CSR trap here means PMP is unavailable -> FAIL
    la   t0, csr_trap
    csrw mtvec, t0

    // pmpaddr0 = target >> 2 (NA4: covers exactly the 4 bytes at target)
    la   t5, target
    srli t1, t5, 2
    csrw pmpaddr0, t1

    // pmpcfg0 byte 0 = 0x90 = NA4 (0x10) | L (0x80), no R/W/X
    li   t1, 0x90
    csrw pmpcfg0, t1

    la   t0, pmp_fault
    csrw mtvec, t0

    jr   t5             // fetch from target: PMP must deny (no X)

csr_trap:
    semihosting_fail    // CSR trapped: PMP unavailable -> FAIL

pmp_fault:
    j    .              // fetch denied as expected -> PASS

    .align 2            // 4-byte align so the NA4 region covers this exactly
target:
    semihosting_fail    // executed only if the fetch was allowed -> FAIL
