// With PMP entries implemented but none enabled, an S/U-mode access must fail.
//
// The privileged spec says that if no PMP entry matches an M-mode access the
// access succeeds, but if no entry matches an S/U-mode access while at least one
// entry is implemented, the access fails. pmp_check short-circuited on
// `if(!any_active) return true;`, which allowed the access at any privilege
// level, contradicting the fall-through at the end of the function that already
// returned PRIV == PRIV_M.
//
// This needs a hart that implements U mode: the M-only wrappers pin PRIV to
// PRIV_M, where the buggy and correct results coincide. Run with rv64gc_mup.
//
// No PMP entry is programmed here, so any_active stays false. After mret the
// first U-mode instruction fetch must already be denied.
//
// Pass: the U-mode access traps back to M mode -> j . (exit 0)
// Fail: U mode runs unrestricted -> semihosting SYS_EXIT (exit 2)

#include "pmp_test_common.h"

.section .text
.globl _start
_start:
    // Any trap from U mode lands here and means PMP denied the access -> PASS
    la   t0, pmp_fault
    csrw mtvec, t0

    // mstatus.MPP = U (0) so mret drops to user mode
    li   t0, 3
    slli t0, t0, 11         // MPP mask, bits 12:11
    csrc mstatus, t0

    la   t0, umode_entry
    csrw mepc, t0
    mret

    // Only reached if mret did not transfer control at all
    semihosting_fail

pmp_fault:
    j    .                  // U-mode access denied as expected -> PASS

umode_entry:
    // Reached only if U mode was allowed to run with no PMP entry enabled.
    semihosting_fail
