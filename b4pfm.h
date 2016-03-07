/*
 *  Created on: May 27, 2013
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */


#ifndef B4PFM
#define B4PFM

#include <CL/cl.h>

#define B4PFM_OK							 0
#define B4PFM_OPENCL_ERROR					-1
#define B4PFM_INVALID_SOLVER				-2
#define B4PFM_INVALID_ARGS					-3
#define B4PFM_TOO_MANY_SOLVERS				-4
#define B4PFM_UNKNOWN_ERROR					-5
#define B4PFM_INVALID_OPTS					-6
#define B4PFM_CR_WG_TOO_BIG					-7
#define B4PFM_OTHER_WG_2D11_TOO_BIG			-8
#define B4PFM_OTHER_WG_2D12_TOO_BIG			-9
#define B4PFM_OTHER_WG_2D21_TOO_BIG			-10
#define B4PFM_OTHER_WG_2D22A_TOO_BIG		-11
#define B4PFM_OTHER_WG_2D22B_TOO_BIG		-12
#define B4PFM_OTHER_WG_3D11_TOO_BIG			-13
#define B4PFM_OTHER_WG_3D12_TOO_BIG			-14
#define B4PFM_OTHER_WG_3D21_TOO_BIG			-15
#define B4PFM_OTHER_WG_3D22A_TOO_BIG		-16
#define B4PFM_OTHER_WG_3D22B_TOO_BIG		-17

#define B4PFM_OTHER_WG_Q2_2D11_TOO_BIG		-18
#define B4PFM_OTHER_WG_Q2_2D12_TOO_BIG		-19
#define B4PFM_OTHER_WG_Q2_2D21_TOO_BIG		-20
#define B4PFM_OTHER_WG_Q2_2D22_TOO_BIG		-21
#define B4PFM_OTHER_WG_Q2_3D11_TOO_BIG		-22
#define B4PFM_OTHER_WG_Q2_3D12_TOO_BIG		-23
#define B4PFM_OTHER_WG_Q2_3D21_TOO_BIG		-24
#define B4PFM_OTHER_WG_Q2_3D22_TOO_BIG		-25

#define B4PFM_DEBUG_NONE		0
#define B4PFM_DEBUG_NORMAL		1
#define B4PFM_DEBUG_FULL		2
#define B4PFM_DEBUG_TIMING		3

typedef unsigned int b4pfm2D;
typedef unsigned int b4pfm3D;

typedef struct {
	int hilodouble;
	int use_local_coef;			/* { 0 = no, 1 = yes } */
	int force_priv_coef;		/* { 0 = no, 1 = yes }, use_local_coef + force_priv_coef <= 1 */
	int work_block_size;		/* { 32, 64 } */
	int cr_wg_size; 			/* { k*work_block_size : k=1,2,... } */
	int cr_local_mem_size;		/* { 2^k : k=1,2,... } */
	int other_wg_size_11;		/* { k*work_block_size : k=1,2,... }, 0 = use preferred wg_size */
	int other_wg_size_12;		/* { k*work_block_size : k=1,2,... }, 0 = use preferred wg_size */
	int other_wg_size_21;		/* { k*work_block_size : k=1,2,... }, 0 = use preferred wg_size */
	int other_wg_size_22a;		/* { k*work_block_size : k=1,2,... }, 0 = use preferred wg_size */
	int other_wg_size_22b;		/* { k*work_block_size : k=1,2,... }, 0 = use preferred wg_size */
	int vector_width;			/* { 1,2 } for double and { 1,2,4 } for float */
								/* POW2(k2) % (other_wg_size * vector_width) == 0	 */
	int max_sum_size1;			/* 3*2^j */
	int max_sum_size2a;			/* 3*2^j */ 
	int max_sum_size2b;			/* 3*2^j */

	int other_wg_size_q2_11;	/* { k*work_block_size : k=1,2,... }, 0 = use preferred wg_size */
	int other_wg_size_q2_12;	/* { k*work_block_size : k=1,2,... }, 0 = use preferred wg_size */
	int other_wg_size_q2_21;	/* { k*work_block_size : k=1,2,... }, 0 = use preferred wg_size */
	int other_wg_size_q2_22;	/* { k*work_block_size : k=1,2,... }, 0 = use preferred wg_size */
	int max_sum_size_q2_1;		/* 3*2^j */
	int max_sum_size_q2_2;		/* 3*2^j */
} b4pfm2D_params;

typedef struct {
	int work_block_size;		/* { 32, 64 } */
	int other_wg_size_11;		/* { k*work_block_size : k=1,2,... }, 0 = use preferred wg_size */
	int other_wg_size_12;		/* { k*work_block_size : k=1,2,... }, 0 = use preferred wg_size */
	int other_wg_size_21;		/* { k*work_block_size : k=1,2,... }, 0 = use preferred wg_size */
	int other_wg_size_22a;		/* { k*work_block_size : k=1,2,... }, 0 = use preferred wg_size */
	int other_wg_size_22b;		/* { k*work_block_size : k=1,2,... }, 0 = use preferred wg_size */
	int vector_width;			/* { 1,2 } for double and { 1,2,4 } for float */
								/* POW2(k2) % (other_wg_size * vector_width) == 0 */
	int wg_per_2d_vect_11;
	int wg_per_2d_vect_12;
	int wg_per_2d_vect_21;
	int wg_per_2d_vect_22a;
	int wg_per_2d_vect_22b;
	int max_sum_size1;		/* 3*2^j */
	int max_sum_size2a;		/* 3*2^j */
	int max_sum_size2b;		/* 3*2^j */

	int other_wg_size_q2_11;		/* { k*work_block_size : k=1,2,... }, 0 = use preferred wg_size */
	int other_wg_size_q2_12;		/* { k*work_block_size : k=1,2,... }, 0 = use preferred wg_size */
	int other_wg_size_q2_21;		/* { k*work_block_size : k=1,2,... }, 0 = use preferred wg_size */
	int other_wg_size_q2_22;		/* { k*work_block_size : k=1,2,... }, 0 = use preferred wg_size */

	int wg_per_2d_vect_q2_11;
	int wg_per_2d_vect_q2_12;
	int wg_per_2d_vect_q2_21;
	int wg_per_2d_vect_q2_22;
	int max_sum_size_q2_1;		
	int max_sum_size_q2_2;		


	b4pfm2D_params params_2d;
} b4pfm3D_params;

/* 2D */

b4pfm2D b4pfm2D_init_solver(cl_context context, b4pfm2D_params *opt_params, int k1, int k2, int ldf, int prec, int force_radix2, int debug, int *error);

int b4pfm2D_run_solver(b4pfm2D solver, cl_command_queue queue, cl_mem f, cl_mem tmp, int debug);

int b4pfm2D_load_and_run_solver_float(b4pfm2D solver, float *f, int debug);

int b4pfm2D_load_and_run_solver_double(b4pfm2D solver, double *f, int debug);

int b4pfm2D_default_opts(cl_context context, b4pfm2D_params *opt_params, int k1, int k2, int ldf, int prec, int debug);

int b4pfm2D_auto_opts(cl_context context, cl_mem tmp1, cl_mem tmp2, b4pfm2D_params *opt_params, int force_radix2, int k1, int k2, int ldf, int prec, int debug);

int b4pfm2D_free_solver(b4pfm2D solver);

/* 3D */

b4pfm3D b4pfm3D_init_solver(cl_context context, b4pfm3D_params *opt_params, int k1, int k2, int k3, int ldf, int prec, int force_radix2, int debug, int *error);

int b4pfm3D_run_solver(b4pfm3D solver, cl_command_queue queue, cl_mem f, cl_mem tmp1, cl_mem tmp2, int debug);

int b4pfm3D_load_and_run_solver_float(b4pfm3D solver, float *f, int debug);

int b4pfm3D_load_and_run_solver_double(b4pfm3D solver, double *f, int debug);

int b4pfm3D_default_opts(cl_context context, b4pfm3D_params *opt_params, int k1, int k2, int k3, int ldf, int prec, int debug);

int b4pfm3D_auto_opts(cl_context context, cl_mem tmp1, cl_mem tmp2, cl_mem tmp3, b4pfm3D_params *opt_params, int force_radix2, int k1, int k2, int k3, int ldf, int prec, int debug);

int b4pfm3D_free_solver(b4pfm3D solver);


#endif
