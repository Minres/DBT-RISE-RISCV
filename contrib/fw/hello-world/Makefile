
TARGET  = hello
C_SRCS  = $(wildcard *.c) 
HEADERS = $(wildcard *.h)
CFLAGS += -Og -g 

BOARD=iss
LINK_TARGET=link
RISCV_ARCH:=rv32imc
RISCV_ABI:=ilp32
#RISCV_ARCH:=rv64imc
#RISCV_ABI:=lp64
LDFLAGS := -g -march=$(RISCV_ARCH) -mabi=$(RISCV_ABI)

compiler := $(shell which riscv64-unknown-elf-gcc)
TOOL_DIR=$(dir $(compiler))

BSP_BASE ?= ../bsp
include $(BSP_BASE)/env/common-gcc.mk
