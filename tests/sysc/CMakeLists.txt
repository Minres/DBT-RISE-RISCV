find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(sysc_tests
    test_main.cpp
    core_complex_test.cpp
    aia_csr_test.cpp
)
target_link_libraries(sysc_tests PRIVATE scc::components GTest::gtest)
# whole-archive so the static core registrations in register_cores.cpp are kept
target_force_link_libraries(sysc_tests PRIVATE dbt-rise-riscv_sc_tlm)
gtest_discover_tests(sysc_tests PROPERTIES TIMEOUT 60)
