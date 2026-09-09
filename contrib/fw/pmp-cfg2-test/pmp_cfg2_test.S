// Verify pmpcfg2 (entries 8-15) is accessible as a CSR on RV64.
// On RV64 valid pmpcfg CSRs are pmpcfg0 (entries 0-7) and pmpcfg2 (entries 8-15).
//
// Pass: pmpcfg2 readback == written value -> j . (exit 0)
// Fail: pmpcfg2 write traps OR readback wrong -> semihosting SYS_EXIT (exit 2)

#include "pmp_test_common.h"

.section .text
.globl _start
_start:
    la   t0, fail
    csrw mtvec, t0

    // 0x18 per byte = NAPOT mode, no RWX, no lock; survives 0x9f byte mask
    li   t0, 0x1818181818181818
    csrw pmpcfg2, t0
    csrr t1, pmpcfg2
    bne  t0, t1, fail

    j    .              // pass: ISS detects j . and exits 0

trap_entry fail
    semihosting_fail
