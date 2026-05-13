// PMP shift test - entry 1 (pmpaddr1 + pmpcfg0 byte 1).
// Verifies that pmpcfg byte extraction uses the correct shift: (i % cfg_reg_size) * 8.
//
// For entry 1, byte 1 of pmpcfg0 = 0x90 (NA4|L), written as 0x9000 to pmpcfg0.
// A wrong shift of 1 bit instead of 8 bits reads 0x4800, which has no PMP_A bits
// set, so any_active stays false and enforcement is skipped.
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

    // Configure pmpaddr1 = test_word2 >> 2 (NA4)
    la   t5, test_word2
    srli t1, t5, 2
    csrw pmpaddr1, t1

    // pmpcfg0 byte 1 = 0x90 -> write 0x9000 to pmpcfg0
    li   t1, 0x9000
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
test_word2:
    .word 0xCAFECAFE
