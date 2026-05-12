// PMP enforcement test - entry 0 (pmpaddr0 + pmpcfg0 byte 0).
// Tests that pmpcfg naming bugs 1+2 in pmp.h are fixed:
//   Bug 1: read_pmpcfg reads pmpaddr[] instead of pmpcfg[]
//   Bug 2: write_pmpcfg writes pmpaddr[] instead of pmpcfg[]
//
// Pass: PMP denies load -> load fault fires -> j . (exit 0)
// Fail: any CSR trap, OR load succeeds without fault -> semihosting SYS_EXIT (exit 2)

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
    // phase1_trap: CSR write trapped - PMP not available
    la   t0, phase1_trap
    csrw mtvec, t0

    // Configure pmpaddr0 = test_word >> 2 (NA4: covers exactly 4 bytes at test_word)
    la   t5, test_word
    srli t1, t5, 2
    csrw pmpaddr0, t1

    // pmpcfg0 byte 0 = 0x90 = NA4 (0x10) | L (0x80), no R/W/X permissions
    li   t1, 0x90
    csrw pmpcfg0, t1

    // phase2_trap: load fault fired - PMP enforcement active = PASS
    la   t0, phase2_trap
    csrw mtvec, t0

    lw   t1, 0(t5)      // should fault: PMP denies read

    semihosting_fail    // no fault: enforcement did not fire -> FAIL

phase1_trap:
    semihosting_fail    // CSR trapped: PMP unavailable -> FAIL

phase2_trap:
    j    .              // load denied as expected -> PASS

    .align 2
test_word:
    .word 0xCAFECAFE
