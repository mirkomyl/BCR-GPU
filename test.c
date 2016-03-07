/*
 *  Created on: May 27, 2013
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <sys/time.h>  

#if OPENCL
#include "b4pfm.h"
#endif

/* Settings begin */

/*#define DOUBLE 1*/
#define K1 KKK
#define K2 KKK
#define K3 KKK

#define FR2 0

//#define TTT 1
#define TTT POW2(2*MAX(1,9-KKK))
#define PRINT 0

/* Setting end */

#ifndef DOUBLE
#error "DOUBLE is not defined."
#endif

#if DOUBLE
typedef double _var;
#else
typedef float _var;
#endif

/* 2^j  */
#define POW2(j) (1<<(j))

#define MIN(a,b) (a < b ? a : b)
#define MAX(a,b) (a > b ? a : b)

#define N1  (POW2(K1)-1)
#define N2  (POW2(K2)-1)
#define N3  (POW2(K3)-1)
#define LDF MAX(1,POW2(K3))

/* verokki.c  */
void base2solver(_var*, _var*, _var, int, int, int);
void base4solver(_var*, _var*, _var, int, int, int);
void base2solver3d(_var*, _var*, int, int, int, int);
void base4solver3d(_var*, _var*, int, int, int, int);

_var randf(_var l, _var h) {
	_var low = MIN(l,h);
	_var high = MAX(l,h);
	_var ran = (high-low)*((_var)rand()/(_var)RAND_MAX) + low;
	return ran;
}


double check_2D_error(_var* f, _var* org_f, int n1, int n2, int ldf) {
	double err = 0.0;
        
	int i,j;
	for(i = 0; i < n1; i++) {
		for(j = 0; j < n2; j++) {
			double p1, p2, p3, p4;
			p1 = p2 = p3 = p4 = 0.0;
			if(i != 0)    p1 = f[(i-1)*ldf+j];
			if(i != n1-1) p2 = f[(i+1)*ldf+j];
			if(j != 0)    p3 = f[i*ldf+j-1];
			if(j != n2-1) p4 = f[i*ldf+j+1];
                        
			#define g(a) ((a)*(a))
			err += g((double)org_f[i*ldf+j] - (4.0*((double)f[i*ldf+j])-p1-p2-p3-p4));
			#undef g
		}
	}
        
	return sqrt(err);
}

double check_2D_block_error(_var* f, _var* org_f, int n1, int n2, int ldf) {
	double max_error = 0.0;

	int i,j;
	for(i = 0; i < n1; i++) {
		double err = 0.0;
		for(j = 0; j < n2; j++) {
			double p1, p2, p3, p4;
			p1 = p2 = p3 = p4 = 0.0;
			if(i != 0)    p1 = f[(i-1)*ldf+j];
			if(i != n1-1) p2 = f[(i+1)*ldf+j];
			if(j != 0)    p3 = f[i*ldf+j-1];
			if(j != n2-1) p4 = f[i*ldf+j+1];
                        
			#define g(a) ((a)*(a))
			err += g((double)org_f[i*ldf+j] - (4.0*((double)f[i*ldf+j])-p1-p2-p3-p4));
			#undef g
		}

		max_error = MAX(max_error, sqrt(err));
	}
        
	return max_error;
}

double check_3D_error(_var* f, _var* org_f, int n1, int n2, int n3, int ldf) {
	double err = 0.0;
        
	int i,j,k;
	for(i = 0; i < n1; i++) {
		for(j = 0; j < n2; j++) {
			for(k = 0; k < n3; k++) {
				double p = 0.0;
				if(i != 0)    p += f[((i-1)*n2+j)*ldf+k];
				if(i != n1-1) p += f[((i+1)*n2+j)*ldf+k];
				if(j != 0)    p += f[(i*n2+j-1)*ldf+k];
				if(j != n2-1) p += f[(i*n2+j+1)*ldf+k];
				if(k != 0)    p += f[(i*n2+j)*ldf+k-1];
				if(k != n3-1) p += f[(i*n2+j)*ldf+k+1];
                        
				#define g(a) ((a)*(a))
				err += g((double)org_f[(i*n2+j)*ldf+k] - (6.0*((double)f[(i*n2+j)*ldf+k])-p));
				#undef g
			}
		}
	}
        
	return sqrt(err);
}

double check_3D_block_error(_var* f, _var* org_f, int n1, int n2, int n3, int ldf) {
	double max_error = 0.0;
        
	int i,j,k;
	for(i = 0; i < n1; i++) {
		double err = 0.0;
		for(j = 0; j < n2; j++) {
			for(k = 0; k < n3; k++) {
				double p = 0.0;
				if(i != 0)    p += f[((i-1)*n2+j)*ldf+k];
				if(i != n1-1) p += f[((i+1)*n2+j)*ldf+k];
				if(j != 0)    p += f[(i*n2+j-1)*ldf+k];
				if(j != n2-1) p += f[(i*n2+j+1)*ldf+k];
				if(k != 0)    p += f[(i*n2+j)*ldf+k-1];
				if(k != n3-1) p += f[(i*n2+j)*ldf+k+1];
                        
				#define g(a) ((a)*(a))
				err += g((double)org_f[(i*n2+j)*ldf+k] - (6.0*((double)f[(i*n2+j)*ldf+k])-p));
				#undef g
			}
		}
		max_error = MAX(max_error, sqrt(err));
	}
        
	return max_error;
}

void print_diff(_var *f1, _var *f2, int n1, int n2, int n3, int ldf, int print) {
	int i, j, k;
	_var max_diff = 0.0;
	for(i = 0; i < n1; i++) {
		for(j = 0; j < n2; j++) {
			for(k = 0; k < n3; k++) {
				_var diff = fabs(f1[(i*n2+j)*ldf+k] - f2[(i*n2+j)*ldf+k]);
				char mark = f1[(i*n2+j)*ldf+k] < f2[(i*n2+j)*ldf+k] ? '+' : '-';
				max_diff = MAX(max_diff,diff);
				if(print) {
					if(diff > 0.000005 || isnan(f1[(i*n2+j)*ldf+k]) || isnan(f2[(i*n2+j)*ldf+k]));
					printf("(%5d,%5d,%5d) === %+.6f vs %+.6f, diff: %.17f || ", i,j,k,f1[(i*n2+j)*ldf+k],f2[(i*n2+j)*ldf+k], diff);
					double tres = 0.00000000000001;
					for(; tres < 1.1 && diff > tres; tres *= 2)
						printf("%c", mark);
					if(tres > 1.1) printf(">");
					printf("\n");
				}
			}
			if(print) printf("=======================================================================================\n");
		}
	}
	printf("Max diff: %f\n", max_diff);
}

#if OPENCL

void compare_gpu_2d(int k1, int k2, int ldf) {
	int err;

	b4pfm2D_params params;

/*
	b4pfm2D_auto_opts(0, 0, 0, &params, FR2, k1, k2, ldf, DOUBLE, B4PFM_DEBUG_FULL); 
 	b4pfm2D solver = b4pfm2D_init_solver(0, &params, k1, k2, ldf, DOUBLE, FR2,B4PFM_DEBUG_FULL, &err);
*/

	b4pfm2D solver = b4pfm2D_init_solver(0, 0, k1, k2, ldf, DOUBLE, FR2, B4PFM_DEBUG_FULL, &err);	 

	if(err != B4PFM_OK) 
		return;

	int n1 = POW2(k1)-1;
	int n2 = POW2(k2)-1;

	srand ( time(NULL) );
 	_var* f1   = malloc(n1*ldf*sizeof(_var));
	_var* f2 = malloc(n1*ldf*sizeof(_var)); 
	_var* dd = malloc(n2*sizeof(_var));
	int i,j;
	for(i = 0; i < n1; i++) 
		for(j = 0; j < n2; j++) 
			f1[i*ldf+j] = f2[i*ldf+j] = randf(-1, 1);

	for(i = 0; i < n2; i++)
		dd[i] = 4.0;

	printf("n1=%d, n2=%d \n", n1, n2);

	struct timeval start, end;
	gettimeofday(&start, NULL);

	/* GPU */
#if DOUBLE
	err = b4pfm2D_load_and_run_solver_double(solver, f2, B4PFM_DEBUG_TIMING);	
#else
	err = b4pfm2D_load_and_run_solver_float(solver, f2, B4PFM_DEBUG_TIMING);	
#endif

	gettimeofday(&end, NULL);
	double time = 1.0*((end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)*1.0E-6) + 0.5E-6;

	b4pfm2D_free_solver(solver);

	printf("GPU time: %f\n", time);

	gettimeofday(&start, NULL);

	/* CPU */
#if FR2
	base2solver(f1, dd, 0.0, k1, n2, ldf);
#else
	base4solver(f1, dd, 0.0, k1, n2, ldf);
#endif	

	gettimeofday(&end, NULL);
	time = 1.0*((end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)*1.0E-6) + 0.5E-6;
	
	printf("CPU time: %f\n", time);

	if(err == B4PFM_OK)
		print_diff(f1,f2,1,n1,n2,ldf,PRINT);

	free(dd);
	free(f1);
	free(f2);
}

void compare_gpu_3d(int k1, int k2, int k3, int ldf) {
	int err;

	b4pfm3D_params params;


	b4pfm3D_auto_opts(0, 0, 0, 0, &params, FR2, k1, k2, k3, ldf, DOUBLE, B4PFM_DEBUG_FULL);
	b4pfm3D solver = b4pfm3D_init_solver(0, &params, k1, k2, k3, ldf, DOUBLE, FR2, B4PFM_DEBUG_FULL, &err);
	
	/*b4pfm3D solver = b4pfm3D_init_solver(0, 0, k1, k2, k3, ldf, DOUBLE, FR2, B4PFM_DEBUG_FULL, &err);	 */

	if(err != B4PFM_OK) 
		return;

	int n1 = POW2(k1)-1;
	int n2 = POW2(k2)-1;
	int n3 = POW2(k3)-1;

	srand ( time(NULL) );
 	_var* f1   = malloc(n1*n2*ldf*sizeof(_var));
	_var* f2 = malloc(n1*n2*ldf*sizeof(_var)); 
	_var* dd = malloc(n3*sizeof(_var));
	int i,j,k;
	for(i = 0; i < n1; i++) 
		for(j = 0; j < n2; j++) 
			for(k = 0; k < n3; k++)
				f1[(i*n2+j)*ldf+k] = f2[(i*n2+j)*ldf+k] = randf(-1, 1);

	for(i = 0; i < n3; i++)
		dd[i] = 6.0;

	printf("n1=%d, n2=%d, n3=%d \n", n1, n2, n3);

	struct timeval start, end;
	gettimeofday(&start, NULL);

	/* GPU */
#if DOUBLE
	err = b4pfm3D_load_and_run_solver_double(solver, f2, B4PFM_DEBUG_TIMING);	
#else
	err = b4pfm3D_load_and_run_solver_float(solver, f2, B4PFM_DEBUG_TIMING);
#endif

	gettimeofday(&end, NULL);
	double time = 1.0*((end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)*1.0E-6) + 0.5E-6;

	printf("GPU time: %f\n", time);

	b4pfm3D_free_solver(solver);

	/* CPU */

	gettimeofday(&start, NULL);

#if FR2
	base2solver3d(f1, dd, k1, k2, n3, ldf);
#else
	base4solver3d(f1, dd, k1, k2, n3, ldf);
#endif

	gettimeofday(&end, NULL);
	time = 1.0*((end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)*1.0E-6) + 0.5E-6;
	
	printf("CPU time: %f\n", time);


	if(err == B4PFM_OK)
		print_diff(f1,f2,n1,n2,n3,ldf,PRINT);

	free(dd);
	free(f1);
	free(f2);


}

#endif

void test_cpu_2d(int k1, int n2, int ldf) {
	int n1 = POW2(k1)-1;

	srand ( time(NULL) );
 	_var* f   = malloc(n1*ldf*sizeof(_var));
	_var* o_f = malloc(n1*ldf*sizeof(_var)); 
	_var* dd = malloc(n2*sizeof(_var));
	int i,j;
	for(i = 0; i < n1; i++) 
		for(j = 0; j < n2; j++) 
			f[i*ldf+j] = o_f[i*ldf+j] = randf(-1, 1);

	for(i = 0; i < n2; i++)
		dd[i] = 4.0;

	printf("n1=%d, n2=%d \n", n1, n2);

	/* Radix 2 */


	double q2_time = 1.0/0.0;

	for(i = 0; i < TTT; i++) {
		struct timeval start, end;
		gettimeofday(&start, NULL);

		base2solver(f, dd, 0.0, k1, n2, ldf);

		gettimeofday(&end, NULL);
		q2_time = MIN(q2_time, 1.0*((end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)*1.0E-6) + 0.5E-6);
	}
	
	printf("Radix-2 time: %.12f\n", q2_time);
#if TTT == 1
	printf("Radix-2 diff: %.20f\n", check_2D_error(f, o_f, n1, n2, ldf));
#endif

	/* Radix-4 */

	for(i = 0; i < n1; i++) 
		for(j = 0; j < n2; j++) 
			f[i*ldf+j] = o_f[i*ldf+j];

	double q4_time = 1.0/0.0;

	for(i = 0; i < TTT; i++) {
		struct timeval start, end;
		gettimeofday(&start, NULL);
		
		base4solver(f, dd, 0.0, k1, n2, ldf);

		gettimeofday(&end, NULL);
		q4_time = MIN(q4_time, 1.0*((end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)*1.0E-6) + 0.5E-6);
	}

        
	printf("Radix-4 time: %.12f\n", q4_time);
#if TTT == 1
	printf("Radix-4 diff: %.20f\n", check_2D_error(f, o_f, n1, n2, ldf));
#endif

	printf("Speed diff q2/q4: %f\n", q2_time / q4_time);

	free(dd);
	free(f);
	free(o_f);
}

void test_avg_err_cpu_2d(int k1, int n2, int ldf) {
	int n1 = POW2(k1)-1;

	srand ( time(NULL) );
 	_var* f   = malloc(n1*ldf*sizeof(_var));
	_var* o_f = malloc(n1*ldf*sizeof(_var)); 
	_var* dd = malloc(n2*sizeof(_var));

	printf("n1=%d, n2=%d \n", n1, n2);

	double cum_err_r2 = 0.0;
	double cum_err_r4 = 0.0;

	double max_err_r2 = 0.0;
	double max_err_r4 = 0.0;

	int t;
	for(t = 0; t < TTT; t++) {
		int i,j;

		for(i = 0; i < n2; i++)
			dd[i] = 4.0;

		for(i = 0; i < n1; i++) 
			for(j = 0; j < n2; j++) 
				f[i*ldf+j] = o_f[i*ldf+j] = randf(-1, 1);

		base2solver(f, dd, 0.0, k1, n2, ldf);
		double err = check_2D_block_error(f, o_f, n1, n2, ldf);
		cum_err_r2 += err;
		max_err_r2 = MAX(max_err_r2, err);

		for(i = 0; i < n1; i++) 
			for(j = 0; j < n2; j++) 
				f[i*ldf+j] = o_f[i*ldf+j];

		base4solver(f, dd, 0.0, k1, n2, ldf);
		err = check_2D_block_error(f, o_f, n1, n2, ldf);
		cum_err_r4 += err;
		max_err_r4 = MAX(max_err_r4, err);
	}

	
	printf("Radix-2 avg-diff: %.20f, max-diff: %.20f \n", cum_err_r2 / TTT, max_err_r2);
        
	printf("Radix-4 avg-diff: %.20f, max-diff: %.20f \n", cum_err_r4 / TTT, max_err_r4);

	free(dd);
	free(f);
	free(o_f);
}

void test_avg_err_3d(int k1, int k2, int n3, int ldf) {
	int n1 = POW2(k1)-1;
	int n2 = POW2(k2)-1;

	srand ( time(NULL) );
 	_var* f   = malloc(n1*n2*ldf*sizeof(_var));
	_var* o_f = malloc(n1*n2*ldf*sizeof(_var)); 
	_var* dd = malloc(n3*sizeof(_var));
	int i,j,k;

	for(i = 0; i < n3; i++)
		dd[i] = 6.0;

	printf("n1=%d, n2=%d, n3=%d, ldf=%d \n", n1, n2, n3, ldf);

	double cum_err_r2 = 0.0;
	double cum_err_r4 = 0.0;

	double max_err_r2 = 0.0;
	double max_err_r4 = 0.0;
	
	int t;
	for(t = 0; t < TTT; t++) {

		for(i = 0; i < n1; i++) 
			for(j = 0; j < n2; j++)
				for(k = 0; k < n3; k++) 
					f[(i*n2+j)*ldf+k] = o_f[(i*n2+j)*ldf+k] = randf(-1, 1);

		/* Radix 2 */
		base2solver3d(f, dd, k1, k2, n3, ldf);
		double err = check_3D_block_error(f, o_f, n1, n2, n3, ldf);
		cum_err_r2 += err;
		max_err_r2 = MAX(max_err_r2, err);

		for(i = 0; i < n1; i++) 
			for(j = 0; j < n2; j++)
				for(k = 0; k < n3; k++) 
					f[(i*n2+j)*ldf+k] = o_f[(i*n2+j)*ldf+k];

		/* Radix-4 */
		base4solver3d(f, dd, k1, k2, n3, ldf);
		err = check_3D_block_error(f, o_f, n1, n2, n3, ldf);
		cum_err_r4 += err;
		max_err_r4 = MAX(max_err_r4, err);
	}

	printf("Radix-2 avg-diff: %.20f, max-diff: %.20f \n", cum_err_r2 / TTT, max_err_r2);
        
	printf("Radix-4 avg-diff: %.20f, max-diff: %.20f \n", cum_err_r4 / TTT, max_err_r4);

	free(dd);
	free(f);
	free(o_f);
}

void test_cpu_3d(int k1, int k2, int n3, int ldf) {
	int n1 = POW2(k1)-1;
	int n2 = POW2(k2)-1;

	srand ( time(NULL) );
 	_var* f   = malloc(n1*n2*ldf*sizeof(_var));
#if TTT == 1
	_var* o_f = malloc(n1*n2*ldf*sizeof(_var)); 
#endif
	_var* dd = malloc(n3*sizeof(_var));
	int i,j,k;
	for(i = 0; i < n1; i++) 
		for(j = 0; j < n2; j++)
			for(k = 0; k < n3; k++) 
#if TTT == 1
				f[(i*n2+j)*ldf+k] = o_f[(i*n2+j)*ldf+k] = randf(-1, 1);
#else
				f[(i*n2+j)*ldf+k] = randf(-1, 1);
#endif

	for(i = 0; i < n3; i++)
		dd[i] = 6.0;

	printf("n1=%d, n2=%d, n3=%d, ldf=%d \n", n1, n2, n3, ldf);

	


	/* Radix 2 */

	double q2_time = 1.0/0.0;

	for(i = 0; i < TTT; i++) {
		struct timeval start, end;
		gettimeofday(&start, NULL);

		base2solver3d(f, dd, k1, k2, n3, ldf);

		gettimeofday(&end, NULL);
		q2_time = MIN(q2_time, 1.0*((end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)*1.0E-6) + 0.5E-6);
	}

	
	printf("Radix-2 time: %.12f\n", q2_time);
#if TTT == 1
	printf("Radix-2 diff: %.20f\n", check_3D_error(f, o_f, n1, n2, n3, ldf));
#endif

	/* Radix-4 */

#if TTT == 1
	for(i = 0; i < n1; i++) 
		for(j = 0; j < n2; j++)
			for(k = 0; k < n3; k++) 
				f[(i*n2+j)*ldf+k] = o_f[(i*n2+j)*ldf+k];
#endif

	double q4_time = 1.0/0.0;

	for(i = 0; i < TTT; i++) {
		struct timeval start, end;
		gettimeofday(&start, NULL);

		base4solver3d(f, dd, k1, k2, n3, ldf);

		gettimeofday(&end, NULL);
		q4_time = MIN(q4_time, 1.0*((end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)*1.0E-6) + 0.5E-6);
	}

        
	printf("Radix-4 time: %.12f\n", q4_time);
#if TTT == 1
	printf("Radix-4 diff: %.20f\n", check_3D_error(f, o_f, n1, n2, n3, ldf));
#endif

	printf("Speed diff q2/q4: %f\n", q2_time / q4_time);

	free(dd);
	free(f);
#if TTT == 1
	free(o_f);
#endif
}


int main() {
	/*test_avg_err_cpu_2d(K2, N3, LDF); */
	/*test_avg_err_3d(K1, K2, N3, LDF); */

	test_cpu_2d(K2, N3, LDF); 
	test_cpu_3d(K1, K2, N3, LDF); 
	/* compare_gpu_2d(K2, K3, LDF); */ 
	/*compare_gpu_3d(K1, K2, K3, LDF); */
	return 0;
}
