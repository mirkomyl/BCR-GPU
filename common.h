/*
 *  Created on: May 27, 2013
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#ifndef B4PFM_COMMON
#define B4PFM_COMMON

#include <CL/cl.h>

#include "b4pfm.h"


#define MAX_SOLVERS 	20
#define MAX_K1			30
#define MAX_K2			30
#define MAX_K3			30

#define	AUTO_OPT_MAX_ITER 	100
#define AUTO_OPT_BREAK		0.01
#define AUTO_OPT_MAX_SEC_MUL	16

#define INV_SOLVER		0

/* Remember to update this when you change kernel args!!! */
#define CR_LOCAL_MEM_BIAS (3*4)

/* 2^j */
#define POW2(j) (1<<(j))

#define MIN(a,b) (a < b ? a : b)
#define MAX(a,b) (a > b ? a : b)

typedef struct {
	cl_context context;

	int force_radix2;

	cl_kernel kernel_2D11;
	cl_kernel kernel_2D12;
	cl_kernel kernel_2D21;
	cl_kernel kernel_2D22A;
	cl_kernel kernel_2D22B;
	cl_kernel kernel_2DCR;

	cl_kernel kernel_Q2_2D11;
	cl_kernel kernel_Q2_2D12;
	cl_kernel kernel_Q2_2D21;
	cl_kernel kernel_Q2_2D22;

	size_t kernel_2D11_threads;
	size_t kernel_2D12_threads;
	size_t kernel_2D21_threads;
	size_t kernel_2D22A_threads;
	size_t kernel_2D22B_threads;	
	size_t kernel_2DCR_threads;

	size_t kernel_Q2_2D11_threads;
	size_t kernel_Q2_2D12_threads;
	size_t kernel_Q2_2D21_threads;
	size_t kernel_Q2_2D22_threads;

	int k1;
	int k2;
	int k3;
	int ldf;
	int double_prec;

	int max_sum_size1;
	int max_sum_size2a;
	int max_sum_size2b;

	int max_sum_size_q2_1;
	int max_sum_size_q2_2;
} r_b4pfm2D;

typedef struct {
	cl_context context;

	r_b4pfm2D *solver_2d;

	int force_radix2;

	cl_kernel kernel_3D11;
	cl_kernel kernel_3D12;
	cl_kernel kernel_3D21;
	cl_kernel kernel_3D22A;
	cl_kernel kernel_3D22B;

	int wg_per_2d_vect_11;
	int wg_per_2d_vect_12;
	int wg_per_2d_vect_21;
	int wg_per_2d_vect_22a;
	int wg_per_2d_vect_22b;

	size_t kernel_3D11_threads;
	size_t kernel_3D12_threads;
	size_t kernel_3D21_threads;
	size_t kernel_3D22A_threads;
	size_t kernel_3D22B_threads;	

	cl_kernel kernel_Q2_3D11;
	cl_kernel kernel_Q2_3D12;
	cl_kernel kernel_Q2_3D21;
	cl_kernel kernel_Q2_3D22;

	int wg_per_2d_vect_q2_11;
	int wg_per_2d_vect_q2_12;
	int wg_per_2d_vect_q2_21;
	int wg_per_2d_vect_q2_22;

	size_t kernel_Q2_3D11_threads;
	size_t kernel_Q2_3D12_threads;
	size_t kernel_Q2_3D21_threads;
	size_t kernel_Q2_3D22_threads;

	int k1;
	int k2;
	int k3;
	int ldf;
	int double_prec;

	int max_sum_size1;
	int max_sum_size2a;
	int max_sum_size2b;

	int max_sum_size_q2_1;
	int max_sum_size_q2_2;
} r_b4pfm3D;

void print_opt_params_2d(b4pfm2D_params *opt_params);
int free_2d_solver(r_b4pfm2D *solver);
r_b4pfm2D* init_2d_solver(cl_context context, int k1, int k2, int k3, int ldf, int prec, b4pfm2D_params opt_params, int force_radix2, int debug, int *error, int *o_err_info);
int run_2d_solver(r_b4pfm2D *solver, cl_command_queue queue, cl_mem f, cl_mem tmp, int count, int r1, int debug, int *err_info);
int run_2d_q2_solver(r_b4pfm2D *solver, cl_command_queue o_queue, cl_mem f, cl_mem tmp, int count, int r1, int o_r2, int debug, int *err_info);
int default_opt_2d_solver(cl_context context, b4pfm2D_params* opt_params, int k2, int k3, int ldf, int prec, int debug, int *err_info);

#endif /* B4PFM_COMMON */


