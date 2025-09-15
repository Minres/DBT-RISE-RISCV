#!/bin/bash
##

# Absolute path to this script, e.g. /home/user/bin/foo.sh
SCRIPT=`readlink -f "$0"`
# Absolute path this script is in, thus /home/user/bin
MYSCRIPTDIR=`dirname "$SCRIPT"`
SCRIPTNAME=`basename "$SCRIPT"`

ROOT_DIR=$(realpath $MYSCRIPTDIR/../..)
SCRIPTDIR=$ROOT_DIR/build/TGC-GEN/scripts
CORE_DSL_DIR=$(realpath ${SCRIPTDIR}/../CoreDSL)

print_help() {
    echo "Usage: $SCRIPTNAME [-h] -c <core name> -v <core variant>"
    echo "Generate RTL for TGC & custom cores"
    echo "  -b <backend list> list of backends to generate code for, default: '$BACKENDS'"
    echo "  -x                use extended list of cores"
    exit 0
}

OUTPUT_ROOT=${ROOT_DIR}/dbt-rise-riscv
TMPL_DIR=${OUTPUT_ROOT}/gen_input/templates

BACKENDS="interp asmjit"
EXT_LIST=0
while getopts 'b:hx' c; do
  case $c in
    x) EXT_LIST=1 ;;
    b) BACKENDS=$OPTARG ;;
    h) print_help ;;
    ?) die "Unknown CLI option!" 255
  esac
done

[[ -d $ROOT_DIR/build/TGC-GEN ]] || (cd build; git clone --recursive -b develop https://git.minres.com/TGFS/TGC-GEN.git )

Generate standard cores
for core in RV32I RV32IMAC RV32GC RV64I RV64GC; do
	for backend in ${BACKENDS}; do 
		${SCRIPTDIR}/generate_iss.sh -o ${OUTPUT_ROOT} -t ${TMPL_DIR} -m src -c $core -b ${backend} ${TMPL_DIR}/../../gen_input/cores.core_desc
	done
	python3 ${SCRIPTDIR}/update_cycle_yaml.py -i ${OUTPUT_ROOT}/contrib/instr/${core}_instr.yaml -o ${OUTPUT_ROOT}/contrib/instr/${core}_fast.yaml -s fast
	python3 ${SCRIPTDIR}/update_cycle_yaml.py -i ${OUTPUT_ROOT}/contrib/instr/${core}_instr.yaml -o ${OUTPUT_ROOT}/contrib/instr/${core}_slow.yaml -s slow
done

# Generate for TGC5C
core=TGC5C
for backend in ${BACKENDS}; do 
    echo ${SCRIPTDIR}/generate_iss.sh -o ${OUTPUT_ROOT} -t ${TMPL_DIR} -m src -c $core -b ${backend} ${CORE_DSL_DIR}/${core}.core_desc
    ${SCRIPTDIR}/generate_iss.sh -o ${OUTPUT_ROOT} -t ${TMPL_DIR} -m src -c $core -b ${backend} ${CORE_DSL_DIR}/${core}.core_desc
done
unset core
python3 ${SCRIPTDIR}/update_cycle_yaml.py -i ${OUTPUT_ROOT}/contrib/instr/TGC5C_instr.yaml -o ${OUTPUT_ROOT}/contrib/instr/TGC5C_fast.yaml -s fast
python3 ${SCRIPTDIR}/update_cycle_yaml.py -i ${OUTPUT_ROOT}/contrib/instr/TGC5C_instr.yaml -o ${OUTPUT_ROOT}/contrib/instr/TGC5C_slow.yaml -s slow
exit
# Generate for other TGCs
OUTPUT_ROOT=${ROOT_DIR}/dbt-rise-custom
for core in TGC5A TGC5B TGC5D TGC5E TGC6B TGC6C TGC6D TGC6E; do
	for backend in ${BACKENDS}; do 
		${SCRIPTDIR}/generate_iss.sh -o ${OUTPUT_ROOT} -t ${TMPL_DIR} -m src -s -c $core -b ${backend} ${CORE_DSL_DIR}/${core}.core_desc
	done
	python3 ${SCRIPTDIR}/update_cycle_yaml.py -i ${OUTPUT_ROOT}/contrib/instr/${core}_instr.yaml -o ${OUTPUT_ROOT}/contrib/instr/${core}_fast.yaml -s fast
	python3 ${SCRIPTDIR}/update_cycle_yaml.py -i ${OUTPUT_ROOT}/contrib/instr/${core}_instr.yaml -o ${OUTPUT_ROOT}/contrib/instr/${core}_slow.yaml -s slow
done
for core in TGC5C_XRB_NN; do
	for backend in interp; do 
		${SCRIPTDIR}/generate_iss.sh -o ${OUTPUT_ROOT} -t ${TMPL_DIR} -m src -s -c $core -b ${backend} ${CORE_DSL_DIR}/${core}.core_desc
		python3 ${SCRIPTDIR}/update_cycle_yaml.py -i ${OUTPUT_ROOT}/contrib/instr/${core}_instr.yaml -o ${OUTPUT_ROOT}/contrib/instr/${core}_fast.yaml -s fast
		python3 ${SCRIPTDIR}/update_cycle_yaml.py -i ${OUTPUT_ROOT}/contrib/instr/${core}_instr.yaml -o ${OUTPUT_ROOT}/contrib/instr/${core}_slow.yaml -s slow
	done
done
for core in RV32GCV RV64GCV; do
	echo ${SCRIPTDIR}/generate_iss.sh -o ${OUTPUT_ROOT} -t ${TMPL_DIR} -m src -s -c $core -b interp ${TMPL_DIR}/../../gen_input/cores.core_desc
	${SCRIPTDIR}/generate_iss.sh -o ${OUTPUT_ROOT} -t ${TMPL_DIR} -m src -s -c $core -b interp ${TMPL_DIR}/../../gen_input/cores.core_desc
	python3 ${SCRIPTDIR}/update_cycle_yaml.py -i ${OUTPUT_ROOT}/contrib/instr/${core}_instr.yaml -o ${OUTPUT_ROOT}/contrib/instr/${core}_fast.yaml -s fast
	python3 ${SCRIPTDIR}/update_cycle_yaml.py -i ${OUTPUT_ROOT}/contrib/instr/${core}_instr.yaml -o ${OUTPUT_ROOT}/contrib/instr/${core}_slow.yaml -s slow
done
# Generate extended list
if [ $EXT_LIST -ne 0 ]; then
	for core in TGC5Xe TGC5Xi TGC5Xec TGC5Xic TGC5Xem TGC5Xim TGC5Xemc TGC5Ximc; do
		for backend in ${BACKENDS}; do 
			${SCRIPTDIR}/generate_iss.sh -o ${OUTPUT_ROOT} -c $core -b ${backend} ${CORE_DSL_DIR}/TGC5.core_desc
		done
		python3 ${SCRIPTDIR}/update_cycle_yaml.py -i ${OUTPUT_ROOT}/contrib/instr/${core}_instr.yaml -o ${OUTPUT_ROOT}/contrib/instr/${core}_fast.yaml -s fast
		python3 ${SCRIPTDIR}/update_cycle_yaml.py -i ${OUTPUT_ROOT}/contrib/instr/${core}_instr.yaml -o ${OUTPUT_ROOT}/contrib/instr/${core}_slow.yaml -s slow
	done
fi
