// Verify pmpaddr0 is accessible as a CSR (no illegal-instruction trap on read/write).
// Pass: j . detected as JUMP_TO_SELF by ISS, exits 0.
// Fail: semihosting SYS_EXIT sequence, ISS exits non-zero.

#include "pmp_test_common.h"

.section .text
.globl _start
_start:
    // Install fail handler: any unexpected trap before the ecall path means the CSR
    // is not available or behaves incorrectly
    la   t0, fail
    csrw mtvec, t0

    // Write a distinctive value to pmpaddr0 and verify the readback matches
    li   t0, 0xDEAD
    csrw pmpaddr0, t0
    csrr t1, pmpaddr0
    bne  t0, t1, fail

    j    .              // pass: ISS detects j . and exits 0

fail:
    // Semihosting SYS_EXIT: the ISS checks for slli+ebreak+srai and intercepts
    // before dispatching to mtvec. Must use 4-byte ebreak (not c.ebreak) so the
    // check lands at the correct offsets (-4 and +4 from the ebreak address).
    semihosting_fail
