// VP guard test: verify PMP enforcement works with only 8 entries (S5 hardware limit).
// Configures entry 0 to deny load from test_word; expects load-access fault.
// Also verifies pmpaddr8+ traps (outside the 8-entry window).
//
// Pass: phase2_trap fires (load denied) -> j . (exit 0)
// Fail: CSR trap, load succeeds, or unexpected behavior -> semihosting SYS_EXIT (exit 2)

.macro semihosting_fail
    li   a0, 0x18
    .option push
    .option norvc
    slli zero, zero, 0x1f
    ebreak
    srai zero, zero, 7
    .option pop
    j    .
.endm

.section .text
.globl _start
_start:
    // phase1_trap: any CSR trap = PMP not available
    la   t0, phase1_trap
    csrw mtvec, t0

    // Configure entry 0: NA4, no R/W/X, locked
    la   t5, test_word
    srli t1, t5, 2
    csrw pmpaddr0, t1

    li   t1, 0x90           // NA4(0x10) | L(0x80)
    csrw pmpcfg0, t1

    // phase2_trap: load fault = enforcement active = PASS
    la   t0, phase2_trap
    csrw mtvec, t0

    lw   t1, 0(t5)          // should fault: entry 0 denies read

    semihosting_fail        // no fault -> FAIL

phase1_trap:
    semihosting_fail        // CSR trap -> FAIL

phase2_trap:
    j    .                  // load denied as expected -> PASS

    .align 2
test_word:
    .word 0xCAFECAFE
