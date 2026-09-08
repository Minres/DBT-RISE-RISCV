// PMP lock bit (L, bit 7 of a cfg byte) must be enforced.
//
// Per the RISC-V privileged spec, once a PMP entry is locked only a reset can
// unlock it: L=1 freezes both pmpicfg and the associated pmpiaddr, and if a
// locked entry uses A=TOR it additionally freezes pmpaddr[i-1], the region's
// lower bound. write_pmpcfg and write_pmpaddr overwrote their targets
// unconditionally, so a locked configuration could be relaxed at will.
//
// The locked entries here use A=OFF (and an empty TOR range) so no region is
// ever matched. This keeps the test purely about CSR retention and removes any
// chance of the test denying its own instruction fetches.
//
// Writes to locked entries are ignored silently rather than trapping, so every
// check is a CSR write followed by a readback compare.
//
// Pass: all three locked values are retained -> j . (exit 0)
// Fail: any CSR trap, OR any locked value changed -> semihosting SYS_EXIT (exit 2)

#include "pmp_test_common.h"

.section .text
.globl _start
_start:
    // A CSR trap here means PMP is unavailable -> FAIL
    la   t0, fail
    csrw mtvec, t0

    // Entry 0: set an address, then lock the entry with A=OFF so it is locked
    // but inactive.
    li   t1, 0x1111
    csrw pmpaddr0, t1
    li   t1, 0x80               // L (0x80), A=OFF
    csrw pmpcfg0, t1
    csrr t2, pmpcfg0
    li   t3, 0x80
    bne  t2, t3, fail           // lock bit did not store at all

    // A locked cfg byte must survive an attempt to clear it.
    csrw pmpcfg0, zero
    csrr t2, pmpcfg0
    bne  t2, t3, fail

    // A locked entry's pmpaddr must survive an attempt to change it.
    li   t1, 0x2222
    csrw pmpaddr0, t1
    csrr t2, pmpaddr0
    li   t3, 0x1111
    bne  t2, t3, fail

    // Entry 3 as locked TOR freezes pmpaddr2, its lower bound, even though
    // entry 2 is itself unlocked. pmpaddr3 stays 0 so the range is empty.
    li   t1, 0x3333
    csrw pmpaddr2, t1
    li   t1, 0x88               // L (0x80) | A=TOR (0x08)
    slli t1, t1, 24             // place in cfg byte 3 -> entry 3
    csrw pmpcfg0, t1
    li   t1, 0x4444
    csrw pmpaddr2, t1
    csrr t2, pmpaddr2
    li   t3, 0x3333
    bne  t2, t3, fail

    j    .                      // every locked value retained -> PASS

fail:
    semihosting_fail
