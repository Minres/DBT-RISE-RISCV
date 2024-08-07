cmake_minimum_required(VERSION 3.12)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/../cmake) # main (top) cmake dir
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake) # project specific cmake dir

# Set the name of your project here
project("sotfloat" VERSION 3.0.0)

# Set the version number of your project here (format is MAJOR.MINOR.PATCHLEVEL - e.g. 1.0.0)
set(VERSION "3e")

include(GNUInstallDirs)

set(SPECIALIZATION RISCV)

set(LIB_HEADERS source/include/softfloat.h source/include/softfloat_types.h)
set(PRIMITIVES
	source/s_eq128.c
	source/s_le128.c
	source/s_lt128.c
	source/s_shortShiftLeft128.c
	source/s_shortShiftRight128.c
	source/s_shortShiftRightJam64.c
	source/s_shortShiftRightJam64Extra.c
	source/s_shortShiftRightJam128.c
	source/s_shortShiftRightJam128Extra.c
	source/s_shiftRightJam32.c
	source/s_shiftRightJam64.c
	source/s_shiftRightJam64Extra.c
	source/s_shiftRightJam128.c
	source/s_shiftRightJam128Extra.c
	source/s_shiftRightJam256M.c
	source/s_countLeadingZeros8.c
	source/s_countLeadingZeros16.c
	source/s_countLeadingZeros32.c
	source/s_countLeadingZeros64.c
	source/s_add128.c
	source/s_add256M.c
	source/s_sub128.c
	source/s_sub256M.c
	source/s_mul64ByShifted32To128.c
	source/s_mul64To128.c
	source/s_mul128By32.c
	source/s_mul128To256M.c
	source/s_approxRecip_1Ks.c
	source/s_approxRecip32_1.c
	source/s_approxRecipSqrt_1Ks.c
	source/s_approxRecipSqrt32_1.c
)

set(SPECIALIZE
	source/${SPECIALIZATION}/softfloat_raiseFlags.c
	source/${SPECIALIZATION}/s_f16UIToCommonNaN.c
	source/${SPECIALIZATION}/s_commonNaNToF16UI.c
	source/${SPECIALIZATION}/s_propagateNaNF16UI.c
	source/${SPECIALIZATION}/s_f32UIToCommonNaN.c
	source/${SPECIALIZATION}/s_commonNaNToF32UI.c
	source/${SPECIALIZATION}/s_propagateNaNF32UI.c
	source/${SPECIALIZATION}/s_f64UIToCommonNaN.c
	source/${SPECIALIZATION}/s_commonNaNToF64UI.c
	source/${SPECIALIZATION}/s_propagateNaNF64UI.c
	source/${SPECIALIZATION}/extF80M_isSignalingNaN.c
	source/${SPECIALIZATION}/s_extF80UIToCommonNaN.c
	source/${SPECIALIZATION}/s_commonNaNToExtF80UI.c
	source/${SPECIALIZATION}/s_propagateNaNExtF80UI.c
	source/${SPECIALIZATION}/f128M_isSignalingNaN.c
	source/${SPECIALIZATION}/s_f128UIToCommonNaN.c
	source/${SPECIALIZATION}/s_commonNaNToF128UI.c
	source/${SPECIALIZATION}/s_propagateNaNF128UI.c
)

set(OTHERS
	source/s_roundToUI32.c
	source/s_roundToUI64.c
	source/s_roundToI32.c
	source/s_roundToI64.c
	source/s_normSubnormalF16Sig.c
	source/s_roundPackToF16.c
	source/s_normRoundPackToF16.c
	source/s_addMagsF16.c
	source/s_subMagsF16.c
	source/s_mulAddF16.c
	source/s_normSubnormalF32Sig.c
	source/s_roundPackToF32.c
	source/s_normRoundPackToF32.c
	source/s_addMagsF32.c
	source/s_subMagsF32.c
	source/s_mulAddF32.c
	source/s_normSubnormalF64Sig.c
	source/s_roundPackToF64.c
	source/s_normRoundPackToF64.c
	source/s_addMagsF64.c
	source/s_subMagsF64.c
	source/s_mulAddF64.c
	source/s_normSubnormalExtF80Sig.c
	source/s_roundPackToExtF80.c
	source/s_normRoundPackToExtF80.c
	source/s_addMagsExtF80.c
	source/s_subMagsExtF80.c
	source/s_normSubnormalF128Sig.c
	source/s_roundPackToF128.c
	source/s_normRoundPackToF128.c
	source/s_addMagsF128.c
	source/s_subMagsF128.c
	source/s_mulAddF128.c
	source/softfloat_state.c
	source/ui32_to_f16.c
	source/ui32_to_f32.c
	source/ui32_to_f64.c
	source/ui32_to_extF80.c
	source/ui32_to_extF80M.c
	source/ui32_to_f128.c
	source/ui32_to_f128M.c
	source/ui64_to_f16.c
	source/ui64_to_f32.c
	source/ui64_to_f64.c
	source/ui64_to_extF80.c
	source/ui64_to_extF80M.c
	source/ui64_to_f128.c
	source/ui64_to_f128M.c
	source/i32_to_f16.c
	source/i32_to_f32.c
	source/i32_to_f64.c
	source/i32_to_extF80.c
	source/i32_to_extF80M.c
	source/i32_to_f128.c
	source/i32_to_f128M.c
	source/i64_to_f16.c
	source/i64_to_f32.c
	source/i64_to_f64.c
	source/i64_to_extF80.c
	source/i64_to_extF80M.c
	source/i64_to_f128.c
	source/i64_to_f128M.c
	source/f16_to_ui32.c
	source/f16_to_ui64.c
	source/f16_to_i32.c
	source/f16_to_i64.c
	source/f16_to_ui32_r_minMag.c
	source/f16_to_ui64_r_minMag.c
	source/f16_to_i32_r_minMag.c
	source/f16_to_i64_r_minMag.c
	source/f16_to_f32.c
	source/f16_to_f64.c
	source/f16_to_extF80.c
	source/f16_to_extF80M.c
	source/f16_to_f128.c
	source/f16_to_f128M.c
	source/f16_roundToInt.c
	source/f16_add.c
	source/f16_sub.c
	source/f16_mul.c
	source/f16_mulAdd.c
	source/f16_div.c
	source/f16_rem.c
	source/f16_sqrt.c
	source/f16_eq.c
	source/f16_le.c
	source/f16_lt.c
	source/f16_eq_signaling.c
	source/f16_le_quiet.c
	source/f16_lt_quiet.c
	source/f16_isSignalingNaN.c
	source/f32_to_ui32.c
	source/f32_to_ui64.c
	source/f32_to_i32.c
	source/f32_to_i64.c
	source/f32_to_ui32_r_minMag.c
	source/f32_to_ui64_r_minMag.c
	source/f32_to_i32_r_minMag.c
	source/f32_to_i64_r_minMag.c
	source/f32_to_f16.c
	source/f32_to_f64.c
	source/f32_to_extF80.c
	source/f32_to_extF80M.c
	source/f32_to_f128.c
	source/f32_to_f128M.c
	source/f32_roundToInt.c
	source/f32_add.c
	source/f32_sub.c
	source/f32_mul.c
	source/f32_mulAdd.c
	source/f32_div.c
	source/f32_rem.c
	source/f32_sqrt.c
	source/f32_eq.c
	source/f32_le.c
	source/f32_lt.c
	source/f32_eq_signaling.c
	source/f32_le_quiet.c
	source/f32_lt_quiet.c
	source/f32_isSignalingNaN.c
	source/f64_to_ui32.c
	source/f64_to_ui64.c
	source/f64_to_i32.c
	source/f64_to_i64.c
	source/f64_to_ui32_r_minMag.c
	source/f64_to_ui64_r_minMag.c
	source/f64_to_i32_r_minMag.c
	source/f64_to_i64_r_minMag.c
	source/f64_to_f16.c
	source/f64_to_f32.c
	source/f64_to_extF80.c
	source/f64_to_extF80M.c
	source/f64_to_f128.c
	source/f64_to_f128M.c
	source/f64_roundToInt.c
	source/f64_add.c
	source/f64_sub.c
	source/f64_mul.c
	source/f64_mulAdd.c
	source/f64_div.c
	source/f64_rem.c
	source/f64_sqrt.c
	source/f64_eq.c
	source/f64_le.c
	source/f64_lt.c
	source/f64_eq_signaling.c
	source/f64_le_quiet.c
	source/f64_lt_quiet.c
	source/f64_isSignalingNaN.c
	source/extF80_to_ui32.c
	source/extF80_to_ui64.c
	source/extF80_to_i32.c
	source/extF80_to_i64.c
	source/extF80_to_ui32_r_minMag.c
	source/extF80_to_ui64_r_minMag.c
	source/extF80_to_i32_r_minMag.c
	source/extF80_to_i64_r_minMag.c
	source/extF80_to_f16.c
	source/extF80_to_f32.c
	source/extF80_to_f64.c
	source/extF80_to_f128.c
	source/extF80_roundToInt.c
	source/extF80_add.c
	source/extF80_sub.c
	source/extF80_mul.c
	source/extF80_div.c
	source/extF80_rem.c
	source/extF80_sqrt.c
	source/extF80_eq.c
	source/extF80_le.c
	source/extF80_lt.c
	source/extF80_eq_signaling.c
	source/extF80_le_quiet.c
	source/extF80_lt_quiet.c
	source/extF80_isSignalingNaN.c
	source/extF80M_to_ui32.c
	source/extF80M_to_ui64.c
	source/extF80M_to_i32.c
	source/extF80M_to_i64.c
	source/extF80M_to_ui32_r_minMag.c
	source/extF80M_to_ui64_r_minMag.c
	source/extF80M_to_i32_r_minMag.c
	source/extF80M_to_i64_r_minMag.c
	source/extF80M_to_f16.c
	source/extF80M_to_f32.c
	source/extF80M_to_f64.c
	source/extF80M_to_f128M.c
	source/extF80M_roundToInt.c
	source/extF80M_add.c
	source/extF80M_sub.c
	source/extF80M_mul.c
	source/extF80M_div.c
	source/extF80M_rem.c
	source/extF80M_sqrt.c
	source/extF80M_eq.c
	source/extF80M_le.c
	source/extF80M_lt.c
	source/extF80M_eq_signaling.c
	source/extF80M_le_quiet.c
	source/extF80M_lt_quiet.c
	source/f128_to_ui32.c
	source/f128_to_ui64.c
	source/f128_to_i32.c
	source/f128_to_i64.c
	source/f128_to_ui32_r_minMag.c
	source/f128_to_ui64_r_minMag.c
	source/f128_to_i32_r_minMag.c
	source/f128_to_i64_r_minMag.c
	source/f128_to_f16.c
	source/f128_to_f32.c
	source/f128_to_extF80.c
	source/f128_to_f64.c
	source/f128_roundToInt.c
	source/f128_add.c
	source/f128_sub.c
	source/f128_mul.c
	source/f128_mulAdd.c
	source/f128_div.c
	source/f128_rem.c
	source/f128_sqrt.c
	source/f128_eq.c
	source/f128_le.c
	source/f128_lt.c
	source/f128_eq_signaling.c
	source/f128_le_quiet.c
	source/f128_lt_quiet.c
	source/f128_isSignalingNaN.c
	source/f128M_to_ui32.c
	source/f128M_to_ui64.c
	source/f128M_to_i32.c
	source/f128M_to_i64.c
	source/f128M_to_ui32_r_minMag.c
	source/f128M_to_ui64_r_minMag.c
	source/f128M_to_i32_r_minMag.c
	source/f128M_to_i64_r_minMag.c
	source/f128M_to_f16.c
	source/f128M_to_f32.c
	source/f128M_to_extF80M.c
	source/f128M_to_f64.c
	source/f128M_roundToInt.c
	source/f128M_add.c
	source/f128M_sub.c
	source/f128M_mul.c
	source/f128M_mulAdd.c
	source/f128M_div.c
	source/f128M_rem.c
	source/f128M_sqrt.c
	source/f128M_eq.c
	source/f128M_le.c
	source/f128M_lt.c
	source/f128M_eq_signaling.c
	source/f128M_le_quiet.c
	source/f128M_lt_quiet.c
)

set(LIB_SOURCES ${PRIMITIVES} ${SPECIALIZE} ${OTHERS})

add_library(softfloat STATIC ${LIB_SOURCES})
set_property(TARGET softfloat PROPERTY C_STANDARD 99)
target_compile_definitions(softfloat PRIVATE 
	SOFTFLOAT_ROUND_ODD 
	INLINE_LEVEL=5 
	SOFTFLOAT_FAST_DIV32TO16
  	SOFTFLOAT_FAST_DIV64TO32
  	SOFTFLOAT_FAST_INT64
#  	THREAD_LOCAL=__thread
)
target_include_directories(softfloat PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/build/Linux-x86_64-GCC)
target_include_directories(softfloat PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/source/include ${CMAKE_CURRENT_SOURCE_DIR}/source/${SPECIALIZATION})
set_target_properties(softfloat PROPERTIES
  VERSION ${VERSION}
  FRAMEWORK FALSE
  PUBLIC_HEADER "${LIB_HEADERS}"
)

install(TARGETS softfloat
  EXPORT ${PROJECT_NAME}Targets            # for downstream dependencies
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/static COMPONENT libs   # static lib
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT libs   # shared lib
  FRAMEWORK DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT libs # for mac
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT devel   # headers for mac (note the different component -> different package)
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}                # headers
)
