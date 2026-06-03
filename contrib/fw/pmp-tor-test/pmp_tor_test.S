// Verify that a partial-overlap load (sectors spanning a TOR region boundary) is denied.
// pmpaddr0 sets the TOR base; pmpaddr1/pmpcfg0[1] set a locked read-only TOR region.
// An 8-byte load at test_region covers two 4-byte sectors:
//   sector 0: test_region+0 (below TOR base -> outside range)
//   sector 1: test_region+4 (inside TOR range)
// Correct behaviour: partial overlap -> PMP denies -> load fault fires.
//
// Pass: load fault fires -> j . (exit 0)
// Fail: load succeeds without fault -> semihosting SYS_EXIT (exit 2)

.section .text
.globl _start
_start:
    la   t0, fail
    csrw mtvec, t0

    // pmpaddr0 = (test_region+4) >> 2  (TOR base for entry 1)
    la   t5, test_region
    addi t1, t5, 4
    srli t1, t1, 2
    csrw pmpaddr0, t1

    // pmpaddr1 = (test_region+12) >> 2  (TOR top for entry 1)
    addi t1, t5, 12
    srli t1, t1, 2
    csrw pmpaddr1, t1

    // pmpcfg0: byte 1 = 0x89 (TOR|L|R) -> locked read-only TOR [test_region+4, test_region+12)
    li   t1, 0x8900
    csrw pmpcfg0, t1

    la   t0, pmp_fault
    csrw mtvec, t0

    // 8-byte load at test_region: sector 0 is outside range, sector 1 is inside.
    // PMP must deny (partial overlap).
    ld   t2, 0(t5)

    // No fault fired -> bug present
fail:
    li   a0, 0x18
    .option push
    .option norvc
    slli zero, zero, 0x1f
    ebreak
    srai zero, zero, 7
    .option pop
    j    .

pmp_fault:
    j    .              // load denied as expected -> PASS

    .align 3            // 8-byte align for ld
test_region:
    .quad 0x1234567890ABCDEF
    .quad 0xC001CAFEDEADBEEF
