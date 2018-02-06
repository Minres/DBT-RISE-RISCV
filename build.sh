mkdir -f build_rel
cd build_rel/
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo && \
	cmake --build . && \
	bin/riscv --reset=0x20400000 --verbose=4 $HOME/eclipse-workspace/RiscV-dhrystone/dhrystone

