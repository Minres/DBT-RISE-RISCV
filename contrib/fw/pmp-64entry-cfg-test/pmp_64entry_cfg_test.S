// Verify pmpcfg4 (CSR 0x3A4, holds entries 32-39 on RV64) is accessible and persistent.
// Tests 64-entry pmpcfg registration.
// The probe value deliberately leaves L (0x80) clear: locking entry 32 would freeze its
// config byte, so the clear below could not be observed.
// Pass: write 0x1F -> read back 0x1F -> write 0 -> read back 0 -> j . (exit 0)
// Fail: any trap or readback mismatch -> semihosting SYS_EXIT (exit 2)

#include "pmp_test_common.h"

.section .text
.globl _start
_start:
    la   t0, fail
    csrw mtvec, t0

    // Write and verify a non-zero value
    li   t0, 0x1F
    csrw 0x3A4, t0          // pmpcfg4 write
    csrr t1, 0x3A4          // pmpcfg4 read
    bne  t0, t1, fail       // must match

    // Clear and verify zero
    li   t0, 0
    csrw 0x3A4, t0
    csrr t1, 0x3A4
    bne  t0, t1, fail

    j    .                  // pass

trap_entry fail
    semihosting_fail
