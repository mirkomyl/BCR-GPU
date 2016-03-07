CXXFLAGS = -Wall -march=native -O3 -ansi
C_COMP = gcc
LINKER = gcc 
SRCTOCHAR = ./srcToChar
AR = ar

srcToChar:
	${C_COMP} -o ${SRCTOCHAR} ${CXXFLAGS} ${LDFLAGS} srcToChar.c

.PHONY: b4pfm
b4pfm:
	make srcToChar
	${SRCTOCHAR} kernels2D kernels2D.cl kernels2D.dat
	${SRCTOCHAR} kernels3D kernels3D.cl kernels3D.dat
	${C_COMP} -fpic ${CXXFLAGS} -c b4pfm2D.c
	${C_COMP} -fpic ${CXXFLAGS} -c b4pfm3D.c
	${C_COMP} -shared -o libb4pfm.so b4pfm2D.o b4pfm3D.o

clean:
	rm -f *.dat *.o *.so ${SRCTOCHAR} 
