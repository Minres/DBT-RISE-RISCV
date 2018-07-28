#!/bin/sh
##

if [ -n "$1" ]; then
	suffix=$1
else
	suffix=Debug
fi
cwd=`pwd`
for i in $*; do	
	if echo "$i" | grep 'CMAKE_BUILD_TYPE='; then
		suffix=`echo $i | sed 's/-DCMAKE_BUILD_TYPE=//'`
	fi
done
mkdir -p build/$suffix && cd build/$suffix
cmake $* $cwd
