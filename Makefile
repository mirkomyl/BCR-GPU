CXXFLAGS = -Wall -march=native -O3 -ansi
#CXXFLAGS = -Wall -g
C_COMP = gcc
LINKER = gcc 
SRCTOCHAR = ./srcToChar
AR = ar

srcToChar:
	${C_COMP} -o ${SRCTOCHAR} ${CXXFLAGS} ${LDFLAGS} srcToChar.c

.PHONY: test_double
test_double:
	make clean
	make b4pfm
	${C_COMP} -c test.c ${CXXFLAGS} -DDOUBLE=1 -DOPENCL=1 -o test.o
	${C_COMP} -c verokki.c ${CXXFLAGS} -fopenmp -DDOUBLE=1 -o verokki.o
	${LINKER} -L./ -lm -lOpenCL -fopenmp -lb4pfm -o test test.o verokki.o
	LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH ./test

.PHONY: cpu_test_double
cpu_test_double:
	make clean
	${C_COMP} -c test.c ${CXXFLAGS} -DDOUBLE=1 -DOPENCL=0  -o test.o
	${C_COMP} -c verokki.c ${CXXFLAGS} -fopenmp -DDOUBLE=1 -o verokki.o
	${LINKER} -o test -lm -fopenmp test.o verokki.o
	./test

.PHONY: test_float
test_float:
	make clean
	make b4pfm
	${C_COMP} -c test.c ${CXXFLAGS} -DDOUBLE=0 -DOPENCL=1  -o test.o
	${C_COMP} -c verokki.c ${CXXFLAGS} -fopenmp -DDOUBLE=0 -o verokki.o
	${LINKER} -L./ -lm -lOpenCL -fopenmp -lb4pfm -o test test.o verokki.o
	LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH ./test

.PHONY: b4pfm
b4pfm:
	make srcToChar
	${SRCTOCHAR} kernels2D kernels2D.cl kernels2D.dat
	${SRCTOCHAR} kernels3D kernels3D.cl kernels3D.dat
	${C_COMP} -fpic ${CXXFLAGS} -c b4pfm2D.c
	${C_COMP} -fpic ${CXXFLAGS} -c b4pfm3D.c
	${C_COMP} -shared -o libb4pfm.so b4pfm2D.o b4pfm3D.o

clean:
	rm -f *.dat *.o *.so test ${SRCTOCHAR} 
