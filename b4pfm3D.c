/*
 *  Created on: May 27, 2013
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include "common.h"

#include "kernels3D.dat"

#define KARGS_3D11_O_F 			0
#define KARGS_3D11_O_G 			1
#define KARGS_3D11_R 			2


#define KARGS_3D12_O_F 		0
#define KARGS_3D12_O_G 		1
#define KARGS_3D12_R 			2
#define KARGS_3D12_PREV_COUNT 	3

#define KARGS_3D21_O_F 			0
#define KARGS_3D21_O_G 			1
#define KARGS_3D21_R 			2

#define KARGS_3D22A_O_G 		0
#define KARGS_3D22A_R 			1

#define KARGS_3D22B_O_F 		0
#define KARGS_3D22B_O_G 		1
#define KARGS_3D22B_R 			2
#define KARGS_3D22B_PREV_COUNT 	3

#define KARGS_Q2_3D11_O_F 			0
#define KARGS_Q2_3D11_O_G 			1
#define KARGS_Q2_3D11_R 			2

#define KARGS_Q2_3D12_O_F 		0
#define KARGS_Q2_3D12_O_G 		1
#define KARGS_Q2_3D12_R 			2
#define KARGS_Q2_3D12_PREV_COUNT 	3

#define KARGS_Q2_3D21_O_F 			0
#define KARGS_Q2_3D21_O_G 			1
#define KARGS_Q2_3D21_R 			2

#define KARGS_Q2_3D22_O_F 		0
#define KARGS_Q2_3D22_O_G 		1
#define KARGS_Q2_3D22_R 			2
#define KARGS_Q2_3D22_PREV_COUNT 	3

typedef struct {
	b4pfm3D_params params;
	double time;
} opt_struct_3d;

int initialized_3d = 0;

r_b4pfm3D *solvers_3d[MAX_SOLVERS];

int free_3d_solver(r_b4pfm3D *solver) {

	if(solver == NULL)
		return B4PFM_INVALID_SOLVER;

#define free_kernel(name) \
	if(solver->name) clReleaseKernel(solver->name); solver->name = 0

	free_kernel(kernel_Q2_3D22);
	free_kernel(kernel_Q2_3D21);
	free_kernel(kernel_Q2_3D12);
	free_kernel(kernel_Q2_3D11);

	free_kernel(kernel_3D22B);
	free_kernel(kernel_3D22A);
	free_kernel(kernel_3D21);
	free_kernel(kernel_3D12);
	free_kernel(kernel_3D11);

#undef free_kernel

	if(solver->context)
		clReleaseContext(solver->context);

	if(solver->solver_2d)
		free_2d_solver(solver->solver_2d);

	free(solver);

	return B4PFM_OK;
}

void print_opt_params_3d(b4pfm3D_params *opt_params) {
	printf(
		"{\n" \
		"     other_wg_size_11     = %d\n" \
		"     other_wg_size_12     = %d\n" \
		"     other_wg_size_21     = %d\n" \
		"     other_wg_size_22a    = %d\n" \
		"     other_wg_size_22b    = %d\n" \
		"     wg_per_2d_vect_11    = %d\n" \
		"     wg_per_2d_vect_12    = %d\n" \
		"     wg_per_2d_vect_21    = %d\n" \
		"     wg_per_2d_vect_22a   = %d\n" \
		"     wg_per_2d_vect_22b   = %d\n" \
		"     vector_width         = %d\n" \
		"     max_sum_size1        = %d\n" \
		"     max_sum_size2a       = %d\n" \
		"     max_sum_size2b       = %d\n" \
		"     other_wg_size_q2_11  = %d\n" \
		"     other_wg_size_q2_12  = %d\n" \
		"     other_wg_size_q2_21  = %d\n" \
		"     other_wg_size_q2_22  = %d\n" \
		"     wg_per_2d_vect_q2_11 = %d\n" \
		"     wg_per_2d_vect_q2_12 = %d\n" \
		"     wg_per_2d_vect_q2_21 = %d\n" \
		"     wg_per_2d_vect_q2_22 = %d\n" \
		"     max_sum_size_q2_1    = %d\n" \
		"     max_sum_size_q2_2    = %d\n",
		opt_params->other_wg_size_11,
		opt_params->other_wg_size_12,	
		opt_params->other_wg_size_21,
		opt_params->other_wg_size_22a,
		opt_params->other_wg_size_22b,	
		opt_params->wg_per_2d_vect_11,
		opt_params->wg_per_2d_vect_12,
		opt_params->wg_per_2d_vect_21,
		opt_params->wg_per_2d_vect_22a,
		opt_params->wg_per_2d_vect_22b,
		opt_params->vector_width,
		opt_params->max_sum_size1,
		opt_params->max_sum_size2a,
		opt_params->max_sum_size2b,
		opt_params->other_wg_size_q2_11,
		opt_params->other_wg_size_q2_12,
		opt_params->other_wg_size_q2_21,
		opt_params->other_wg_size_q2_22,
		opt_params->wg_per_2d_vect_q2_11,
		opt_params->wg_per_2d_vect_q2_12,
		opt_params->wg_per_2d_vect_q2_21,
		opt_params->wg_per_2d_vect_q2_22,
		opt_params->max_sum_size_q2_1,
		opt_params->max_sum_size_q2_2);


	printf(
		"}\n");
}

r_b4pfm3D* init_3d_solver(cl_context context, int k1, int k2, int k3, int ldf, int prec, b4pfm3D_params opt_params, int force_radix2, int debug, int *error, int *o_err_info) {

	int *err_info;
	int tmp_err_info;
	if(o_err_info == 0)
		err_info = &tmp_err_info;
	else
		err_info = o_err_info;

	if(debug == B4PFM_DEBUG_FULL || debug == B4PFM_DEBUG_TIMING) { 
		printf("(debug) b4pdf3D / init solver: Initializing B4PDF3D-solver. Args: k1=%d, k2=%d, k3=%d, ldf=%d, prec=%d, force_radix2=%d, opt_params = \n",
			k1, k2, k3, ldf, prec, force_radix2);
		print_opt_params_3d(&opt_params);
	} else if(debug != B4PFM_DEBUG_NONE) printf(
		"(debug) b4pdf3D / init solver: Initializing B4PDF3D-solver. \n");
	*error = B4PFM_UNKNOWN_ERROR;
	int err; 

	int n3 = POW2(k3)-1;

	/* Initial args checks */

	if(    1 > k1 || k1 > MAX_K1
		|| 1 > k2 || k2 > MAX_K2
		|| 1 > k3 || k3 > MAX_K3
		|| ldf < n3) {
		printf(
			"(error) b4pdf3D / init solver: Invalid arguments. \n");
		*error = B4PFM_INVALID_ARGS;
		return NULL;
	}


	/* Initial opt_params checks */

	opt_params.wg_per_2d_vect_11  = MIN(POW2(k2)-1, opt_params.wg_per_2d_vect_11);
	opt_params.wg_per_2d_vect_12  = MIN(POW2(k2)-1, opt_params.wg_per_2d_vect_12);
	opt_params.wg_per_2d_vect_21  = MIN(POW2(k2)-1, opt_params.wg_per_2d_vect_21);
	opt_params.wg_per_2d_vect_22a = MIN(POW2(k2)-1, opt_params.wg_per_2d_vect_22a);
	opt_params.wg_per_2d_vect_22b = MIN(POW2(k2)-1, opt_params.wg_per_2d_vect_22b);

	if(!force_radix2) {

		if(	opt_params.other_wg_size_11 % opt_params.work_block_size != 0 ||
			opt_params.other_wg_size_12 % opt_params.work_block_size != 0 ||
			opt_params.other_wg_size_21 % opt_params.work_block_size != 0 ||
			opt_params.other_wg_size_22a % opt_params.work_block_size != 0 ||
			opt_params.other_wg_size_22b % opt_params.work_block_size != 0 ||
			opt_params.other_wg_size_11 <= 0 ||
			opt_params.other_wg_size_12 <= 0 ||
			opt_params.other_wg_size_21 <= 0 ||
			opt_params.other_wg_size_22a <= 0 ||
			opt_params.other_wg_size_22b <= 0 ||
			opt_params.wg_per_2d_vect_11 <= 0 ||
			opt_params.wg_per_2d_vect_12 <= 0 ||
			opt_params.wg_per_2d_vect_21 <= 0 ||
			opt_params.wg_per_2d_vect_22a <= 0 ||
			opt_params.wg_per_2d_vect_22b <= 0 ||
			opt_params.max_sum_size1 < 4 || opt_params.max_sum_size1 % 4 != 0 ||
			opt_params.max_sum_size2a < 4 || opt_params.max_sum_size2a % 4 != 0 ||
			opt_params.max_sum_size2b < 4 || opt_params.max_sum_size2b % 4 != 0 ) {

			printf(
				"(error) b4pdf3D / init solver: Invalid opt_params. \n");
			*error = B4PFM_INVALID_OPTS;
			return NULL;
		}
	}

	if(force_radix2) {

		if(	opt_params.other_wg_size_q2_11 % opt_params.work_block_size != 0 ||
			opt_params.other_wg_size_q2_12 % opt_params.work_block_size != 0 ||
			opt_params.other_wg_size_q2_11 <= 0 ||
			opt_params.other_wg_size_q2_12 <= 0 ||
			opt_params.wg_per_2d_vect_q2_11 <= 0 ||
			opt_params.wg_per_2d_vect_q2_12 <= 0 ||
			opt_params.max_sum_size_q2_1 < 2 || opt_params.max_sum_size_q2_1 % 2 != 0) {

			printf(
				"(error) b4pdf3D / init solver: Invalid opt_params. \n");
			*error = B4PFM_INVALID_OPTS;
			return NULL;
		}
	}

	if(k1 & 1 || force_radix2) {

		if(	opt_params.other_wg_size_q2_21 % opt_params.work_block_size != 0 ||
			opt_params.other_wg_size_q2_22 % opt_params.work_block_size != 0 ||
			opt_params.other_wg_size_q2_21 <= 0 ||
			opt_params.other_wg_size_q2_22 <= 0 ||
			opt_params.wg_per_2d_vect_q2_21 <= 0 ||
			opt_params.wg_per_2d_vect_q2_22 <= 0 ||
			opt_params.max_sum_size_q2_2 < 2 || opt_params.max_sum_size_q2_2 % 2 != 0 ) {

			printf(
				"(error) b4pdf3D / init solver: Invalid opt_params. \n");
			*error = B4PFM_INVALID_OPTS;
			return NULL;
		}
	}

	/* Malloc new solver if possible */
	r_b4pfm3D *solver = malloc(sizeof(r_b4pfm3D));

	if(solver == NULL) {
		printf(
			"(error) b4pdf3D / init solver: Can't malloc solver.\n");
		*error = B4PFM_UNKNOWN_ERROR;
		return NULL;
	}

	/* Pre-initialize solver */
	solver->context = 0;

	solver->force_radix2 = force_radix2;

	solver->solver_2d = 0;

	solver->k1 = k1;
	solver->k2 = k2;
	solver->k3 = k3;
	solver->ldf = ldf;

	solver->double_prec = prec;
        
	solver->kernel_3D11 = 0;
	solver->kernel_3D12 = 0;
	solver->kernel_3D21 = 0;	
	solver->kernel_3D22A = 0;
	solver->kernel_3D22B = 0;

	solver->kernel_Q2_3D11 = 0;
	solver->kernel_Q2_3D12 = 0;
	solver->kernel_Q2_3D21 = 0;	
	solver->kernel_Q2_3D22 = 0;

	solver->wg_per_2d_vect_11 = opt_params.wg_per_2d_vect_11;
	solver->wg_per_2d_vect_12 = opt_params.wg_per_2d_vect_12;
	solver->wg_per_2d_vect_21 = opt_params.wg_per_2d_vect_21;
	solver->wg_per_2d_vect_22a = opt_params.wg_per_2d_vect_22a;
	solver->wg_per_2d_vect_22b = opt_params.wg_per_2d_vect_22b;

	solver->max_sum_size1 = opt_params.max_sum_size1;
	solver->max_sum_size2a = opt_params.max_sum_size2a;
	solver->max_sum_size2b = opt_params.max_sum_size2b;

	solver->wg_per_2d_vect_q2_11 = opt_params.wg_per_2d_vect_q2_11;
	solver->wg_per_2d_vect_q2_12 = opt_params.wg_per_2d_vect_q2_12;
	solver->wg_per_2d_vect_q2_21 = opt_params.wg_per_2d_vect_q2_21;
	solver->wg_per_2d_vect_q2_22 = opt_params.wg_per_2d_vect_q2_22;

	solver->max_sum_size_q2_1 = opt_params.max_sum_size_q2_1;
	solver->max_sum_size_q2_2 = opt_params.max_sum_size_q2_2;

	/* Sets opencl context for solver */
	err = clRetainContext(context);

	if(err != CL_SUCCESS) {
		printf(
			"(error) b4pdf3D / init solver: Invalid context. " \
			"OpenCL errorcode: %d\n", err);
		*error = B4PFM_INVALID_ARGS;
		return NULL;
	}
	
	solver->context = context;

	/* TODO: Move this! */
	cl_program program = 0;

#define free_all() 												\
	if(program)					clReleaseProgram(program);		\
	free_3d_solver(solver)

	/* Find the OpenCL device corresponding to given context */
	cl_device_id device;

	err = clGetContextInfo(solver->context, CL_CONTEXT_DEVICES, 
		sizeof(cl_device_id), &device, 0);

	if(err != CL_SUCCESS) {
		printf(
			"(error) b4pdf3D / init solver: Can't query device id from context. " \
			"OpenCL errorcode: %d\n", err);
		free_all();
		*error = B4PFM_OPENCL_ERROR;
		return NULL;
	}

	/* Check device specific opt_params */

	/* Find out the amount of local memory in device */
	cl_ulong max_local_mem_size;
	clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, 
		sizeof(cl_ulong), &max_local_mem_size, 0);

	/* Determine maximum work-group size */
	size_t max_work_item_sizes[3];
	clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, 
		sizeof(max_work_item_sizes), max_work_item_sizes, 0);

	/* Check */
	
	if(!force_radix2) {
		if(	opt_params.other_wg_size_11 > max_work_item_sizes[0] ||
			opt_params.other_wg_size_12 > max_work_item_sizes[0] ||
			opt_params.other_wg_size_21 > max_work_item_sizes[0] ||
			opt_params.other_wg_size_22a > max_work_item_sizes[0] ||
			opt_params.other_wg_size_22b > max_work_item_sizes[0]) {

			printf(
				"(error) b4pdf3D / init solver: Invalid opt_params. \n");
			free_all();
			*error = B4PFM_INVALID_OPTS;
			return NULL;
		}
	}

	if(force_radix2) {
		if(	opt_params.other_wg_size_q2_11 > max_work_item_sizes[0] ||
			opt_params.other_wg_size_q2_12 > max_work_item_sizes[0]) {

			printf(
				"(error) b4pdf3D / init solver: Invalid opt_params. \n");
			free_all();
			*error = B4PFM_INVALID_OPTS;
			return NULL;
		}
	}

	if(k1 & 1 || force_radix2) {
		if(	opt_params.other_wg_size_q2_21 > max_work_item_sizes[0] ||
			opt_params.other_wg_size_q2_22 > max_work_item_sizes[0]) {

			printf(
				"(error) b4pdf3D / init solver: Invalid opt_params. \n");
			free_all();
			*error = B4PFM_INVALID_OPTS;
			return NULL;
		}
	}

	/* TODO: Same checks for other_wg_size */

	/* Checks double precision support */
	char *extensions = malloc(1000*sizeof(char));
	clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 1000, extensions, 0);
	int double_support = strstr(extensions, "cl_khr_fp64") != 0;
	int ati_double_support = strstr(extensions, "cl_amd_fp64") != 0;
	free(extensions);

	if(solver->double_prec && double_support == 0 && ati_double_support == 0) {
		printf(
			"(error) b4pdf3D / init solver: Device doesn't support double precision. \n");
		free_all();
		*error = B4PFM_INVALID_ARGS;
		return NULL;
	}


	/* Creates a new opencl program-object */
	program = clCreateProgramWithSource(solver->context, 1, &kernels3D, 0, &err);
	if(err != CL_SUCCESS) {
		printf(
			"(error) b4pdf3D / init solver: Can't create program. " \
			"OpenCL errorcode: %d\n.", err);
		free_all();
		*error = B4PFM_OPENCL_ERROR;
		return NULL;
	}

	/* Build opencl program */
	char *format = 
		" -D Q2SOLVER=%d"		\
		" -D Q4SOLVER=%d"		\
		" -D DOUBLE=%d"			\
		" -D AMD_FP64=%d"		\
		" -D D=%d"				\
		" -D OTHER_WG_SIZE_3D11=%d"	\
		" -D OTHER_WG_SIZE_3D12=%d"	\
		" -D OTHER_WG_SIZE_3D21=%d"	\
		" -D OTHER_WG_SIZE_3D22A=%d"	\
		" -D OTHER_WG_SIZE_3D22B=%d"	\
		" -D WG_PER_2D_VECT_11=%d"	\
		" -D WG_PER_2D_VECT_12=%d"	\
		" -D WG_PER_2D_VECT_21=%d"	\
		" -D WG_PER_2D_VECT_22A=%d"	\
		" -D WG_PER_2D_VECT_22B=%d"	\
		" -D MAX_SUM_SIZE1=%d"	\
		" -D MAX_SUM_SIZE2A=%d"	\
		" -D MAX_SUM_SIZE2B=%d"	\
		" -D OTHER_WG_SIZE_Q2_3D11=%d"	\
		" -D OTHER_WG_SIZE_Q2_3D12=%d"	\
		" -D OTHER_WG_SIZE_Q2_3D21=%d"	\
		" -D OTHER_WG_SIZE_Q2_3D22=%d"	\
		" -D WG_PER_2D_VECT_Q2_11=%d"	\
		" -D WG_PER_2D_VECT_Q2_12=%d"	\
		" -D WG_PER_2D_VECT_Q2_21=%d"	\
		" -D WG_PER_2D_VECT_Q2_22=%d"	\
		" -D MAX_SUM_SIZE_Q2_1=%d"	\
		" -D MAX_SUM_SIZE_Q2_2=%d"	\
		" -D K1=%d"				\
		" -D K2=%d"				\
		" -D K3=%d"				\
		" -D LDF=%d";

	char opt[strlen(format)+100];

	sprintf(opt, format, 
		k1 & 1 || force_radix2,
		!force_radix2,
		solver->double_prec,    
		ati_double_support,
		opt_params.vector_width, 
		opt_params.other_wg_size_11,
		opt_params.other_wg_size_12,
		opt_params.other_wg_size_21,
		opt_params.other_wg_size_22a,
		opt_params.other_wg_size_22b,
		opt_params.wg_per_2d_vect_11,
		opt_params.wg_per_2d_vect_12,
		opt_params.wg_per_2d_vect_21,
		opt_params.wg_per_2d_vect_22a,
		opt_params.wg_per_2d_vect_22b,
		opt_params.max_sum_size1,
		opt_params.max_sum_size2a,
		opt_params.max_sum_size2b,
		opt_params.other_wg_size_q2_11,
		opt_params.other_wg_size_q2_12,
		opt_params.other_wg_size_q2_21,
		opt_params.other_wg_size_q2_22,
		opt_params.wg_per_2d_vect_q2_11,
		opt_params.wg_per_2d_vect_q2_12,
		opt_params.wg_per_2d_vect_q2_21,
		opt_params.wg_per_2d_vect_q2_22,
		opt_params.max_sum_size_q2_1,
		opt_params.max_sum_size_q2_2,
		k1,
		k2,
		k3,
		ldf);

	if(debug == B4PFM_DEBUG_FULL || debug == B4PFM_DEBUG_TIMING) 
		printf("(debug) b4pdf3D / init solver: Compiler args: %s\n", opt);

	err = clBuildProgram(program, 0, 0, opt, 0, 0);

	if (err || debug == B4PFM_DEBUG_FULL || debug == B4PFM_DEBUG_TIMING) {
		char *log = malloc(102400);
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 
			102400, log, NULL );
		printf("(debug) b4pdf3D / init solver: OpenCL compiler output:\n");
		printf("======================================================\n");
		printf("%s\n", log);
		printf("======================================================\n");
		free(log);
	}       

	if(err != CL_SUCCESS) {
		printf(
			"(error) b4pdf3D / init solver: Can't build program. " \
			"OpenCL errorcode: %d\n", err);
		free_all();
		*error = B4PFM_OPENCL_ERROR;
		return NULL;
	} 
/*
	size_t bin_size;
	clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &bin_size, 0);
	char *log = malloc(bin_size);
	clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(char *), &log, 0);
	printf("======================================================\n");
	printf("%s\n", log);
	printf("======================================================\n");
	free(log);
*/

	clUnloadCompiler();

	/* Creates opencl kernel handels */

	if(!force_radix2) {
		solver->kernel_3D11 = clCreateKernel(program, "b4pfm3D_11", &err);
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf3D / init solver: Can't create kernel b4pfm3D_11. " \
				"OpenCL errorcode: %d\n.", err);	
			free_all();
			*error = B4PFM_OPENCL_ERROR;
			return NULL;
		}
        
		solver->kernel_3D12 = clCreateKernel(program, "b4pfm3D_12", &err);
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf3D / init solver: Can't create kernel b4pfm3D_12. " \
				"OpenCL errorcode: %d\n.", err);
			free_all();
			*error = B4PFM_OPENCL_ERROR;
			return NULL; 
		}


		solver->kernel_3D21 = clCreateKernel(program, "b4pfm3D_21", &err);
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf3D / init solver: Can't create kernel b4pfm3D_21. " \
				"OpenCL errorcode: %d\n.", err);
			free_all();
			*error = B4PFM_OPENCL_ERROR;
			return NULL;
		}

		solver->kernel_3D22A = clCreateKernel(program, "b4pfm3D_22a", &err);	
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf3D / init solver: Can't create kernel b4pfm3D_22a. " \
				"OpenCL errorcode: %d\n.", err);
			free_all();
			*error = B4PFM_OPENCL_ERROR;
			return NULL; 
		}

		solver->kernel_3D22B = clCreateKernel(program, "b4pfm3D_22b", &err);
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf3D / init solver: Can't create kernel b4pfm3D_22b. " \
				"OpenCL errorcode: %d\n.", err);
			free_all();
			*error = B4PFM_OPENCL_ERROR;
			return NULL; 
		}
	}

	if(force_radix2) {
		solver->kernel_Q2_3D11 = clCreateKernel(program, "b2pfm3D_11", &err);
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf3D / init solver: Can't create kernel b2pfm3D_11. " \
				"OpenCL errorcode: %d\n.", err);	
			free_all();
			*error = B4PFM_OPENCL_ERROR;
			return NULL;
		}
        
		solver->kernel_Q2_3D12 = clCreateKernel(program, "b2pfm3D_12", &err);
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf3D / init solver: Can't create kernel b2pfm3D_12. " \
				"OpenCL errorcode: %d\n.", err);
			free_all();
			*error = B4PFM_OPENCL_ERROR;
			return NULL; 
		}

	}

	if(k1 & 1 || force_radix2) {


		solver->kernel_Q2_3D21 = clCreateKernel(program, "b2pfm3D_21", &err);
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf3D / init solver: Can't create kernel b2pfm3D_21. " \
				"OpenCL errorcode: %d\n.", err);
			free_all();
			*error = B4PFM_OPENCL_ERROR;
			return NULL;
		}

		solver->kernel_Q2_3D22 = clCreateKernel(program, "b2pfm3D_22", &err);	
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf3D / init solver: Can't create kernel b2pfm3D_22. " \
				"OpenCL errorcode: %d\n.", err);
			free_all();
			*error = B4PFM_OPENCL_ERROR;
			return NULL; 
		}
	}

	/* Check work group sizes */

#define handle_wrong_wg_size(kern,param,max,err) \
	if(opt_params.param <= max) { \
		solver->kern = opt_params.param; \
	} else { \
		printf( \
			"(error) b4pdf2D / init solver: param is too big. " \
			"Max value: %d\n", (int) max); \
		free_all(); \
		*err_info = max; \
		*error = err; \
		return NULL; \
	}

	/* Finds out what is the optimal work group size for each kernel */

	if(!force_radix2) {
		size_t tc11, tc12, tc21, tc22a, tc22b;
		clGetKernelWorkGroupInfo(solver->kernel_3D11, device, 
			CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &tc11, 0);
		clGetKernelWorkGroupInfo(solver->kernel_3D12, device, 
			CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &tc12, 0);
		clGetKernelWorkGroupInfo(solver->kernel_3D21, device, 
			CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &tc21, 0);
		clGetKernelWorkGroupInfo(solver->kernel_3D22A, device, 
			CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &tc22a, 0);
		clGetKernelWorkGroupInfo(solver->kernel_3D22B, device, 
			CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &tc22b, 0);

		handle_wrong_wg_size(kernel_3D11_threads,  other_wg_size_11, 	tc11,	B4PFM_OTHER_WG_3D11_TOO_BIG);
		handle_wrong_wg_size(kernel_3D12_threads,  other_wg_size_12, 	tc12, 	B4PFM_OTHER_WG_3D12_TOO_BIG);
		handle_wrong_wg_size(kernel_3D21_threads,  other_wg_size_21, 	tc21, 	B4PFM_OTHER_WG_3D21_TOO_BIG);
		handle_wrong_wg_size(kernel_3D22A_threads, other_wg_size_22a, 	tc22a, 	B4PFM_OTHER_WG_3D22A_TOO_BIG);
		handle_wrong_wg_size(kernel_3D22B_threads, other_wg_size_22b, 	tc22b, 	B4PFM_OTHER_WG_3D22B_TOO_BIG);
	}

	if(force_radix2) {
		size_t tc11, tc12;
		clGetKernelWorkGroupInfo(solver->kernel_Q2_3D11, device, 
			CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &tc11, 0);
		clGetKernelWorkGroupInfo(solver->kernel_Q2_3D12, device, 
			CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &tc12, 0);

		handle_wrong_wg_size(kernel_Q2_3D11_threads,  other_wg_size_q2_11, tc11,	B4PFM_OTHER_WG_Q2_3D11_TOO_BIG);
		handle_wrong_wg_size(kernel_Q2_3D12_threads,  other_wg_size_q2_12, tc12, 	B4PFM_OTHER_WG_Q2_3D12_TOO_BIG);
	}

	if(k1 & 1 || force_radix2) {
		size_t tc21, tc22;
		clGetKernelWorkGroupInfo(solver->kernel_Q2_3D21, device, 
			CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &tc21, 0);
		clGetKernelWorkGroupInfo(solver->kernel_Q2_3D22, device, 
			CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &tc22, 0);

		handle_wrong_wg_size(kernel_Q2_3D21_threads,  other_wg_size_q2_21, tc21, 	B4PFM_OTHER_WG_Q2_3D21_TOO_BIG);
		handle_wrong_wg_size(kernel_Q2_3D22_threads,  other_wg_size_q2_22, tc22, 	B4PFM_OTHER_WG_Q2_3D22_TOO_BIG);
	}

	if(debug == B4PFM_DEBUG_FULL || debug == B4PFM_DEBUG_TIMING)
		printf("(debug) b4pdf3D / init solver: Initializing 2D-solver...\n");
	solver->solver_2d = init_2d_solver(context, k1, k2, k3, ldf, prec, opt_params.params_2d, force_radix2, debug, error, err_info);
	if(solver->solver_2d == NULL) {
		free_all();
		return NULL;
	}
	if(debug == B4PFM_DEBUG_FULL || debug == B4PFM_DEBUG_TIMING)
		printf("(debug) b4pdf3D / init solver: 2D-solver initialized.\n");

	if(debug != B4PFM_DEBUG_NONE) printf(
		"(debug) b4pdf3D / init solver: Solver is now initialized. \n");

#undef free_all

	/* Everything is ready. Return. */
	*error = B4PFM_OK;
	return solver;
	
}

int run_3d_solver(r_b4pfm3D *solver, cl_command_queue o_queue, cl_mem f, cl_mem o_tmp1, cl_mem o_tmp2, int debug, int *o_err_info) {

	int *err_info;
	int tmp_err_info;
	if(o_err_info == 0)
		err_info = &tmp_err_info;
	else
		err_info = o_err_info;

	if(solver == NULL)
		return B4PFM_INVALID_SOLVER;


	else if(debug != B4PFM_DEBUG_NONE) printf(
		"(debug) b4pdf3D / run solver: Begin...\n");

	int err;

	int var_size = solver->double_prec ? sizeof(cl_double) : sizeof(cl_float);

	int cmd_queue_prof_was_enabled = 0;
	int cmd_queue_prof_enabled = 0;

	/* Creates a new opencl command queue if necessary. */
	cl_command_queue queue;
	if(o_queue == 0) {
		if(debug != B4PFM_DEBUG_NONE) printf(
			"(debug) b4pdf3D / run solver: No OpenCL Commandqueue given. "\
			"Creating OpenCL Commandqueue... \n");
		cl_device_id device;
 		err = clGetContextInfo(solver->context, CL_CONTEXT_DEVICES, 
			sizeof(cl_device_id), &device, 0);
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf3D / run solver: Can't query device id from context. " \
				"OpenCL errorcode: %d\n", err);
			return B4PFM_OPENCL_ERROR;
		}

		if(debug == B4PFM_DEBUG_TIMING) {
			queue = clCreateCommandQueue(solver->context, device, CL_QUEUE_PROFILING_ENABLE, &err);
			printf(
				"(debug) b4pdf3D / run solver: OpenCL command queue profiling enabled. \n");
		} else
			queue = clCreateCommandQueue(solver->context, device, 0, &err);
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf3D / run solver: Can't create command queue. " \
				"OpenCL errorcode: %d\n.", err);
			return B4PFM_OPENCL_ERROR;
		}
	} else {

		queue = o_queue;
		clRetainCommandQueue(queue);

		if(debug == B4PFM_DEBUG_TIMING) {
			cl_command_queue_properties old_cmd_queue_pro;
			err = clSetCommandQueueProperty(queue, CL_QUEUE_PROFILING_ENABLE, CL_TRUE, &old_cmd_queue_pro);

			cmd_queue_prof_was_enabled = old_cmd_queue_pro & CL_QUEUE_PROFILING_ENABLE;

			if(err == CL_SUCCESS)
				cmd_queue_prof_enabled = 1;
			else
				printf(
					"(error) b4pdf3D / run solver: Can't enable OpenCL command queue profiling. " 
					"OpenCL errorcode: %d.\n", err);

			if(cmd_queue_prof_enabled && !cmd_queue_prof_was_enabled)
				printf(
					"(debug) b4pdf3D / run solver: OpenCL command queue profiling enabled. \n");
		}

	}

	/* Creates a new opencl memory buffer for temporar data */
	cl_mem tmp1;
	if(o_tmp1 == 0) {
		if(debug != B4PFM_DEBUG_NONE) printf(
			"(debug) b4pdf3D / run solver: No global workplace no. 1 given. " \
			"Creating global workplace buffer... \n");
		int tmp_mem_size = 3*POW2(solver->k1-2)*(POW2(solver->k2)-1)*POW2(solver->k3)*var_size;
		if(debug == B4PFM_DEBUG_FULL || debug == B4PFM_DEBUG_TIMING) printf(
			"(debug) b4pdf3D / run solver: Global workplace no. 1 size = %d bytes. \n",
			tmp_mem_size);
		tmp1 = clCreateBuffer(solver->context, CL_MEM_READ_WRITE, tmp_mem_size, 0, &err);
        
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf3D / run solver: Can't allocate device memory for " \
				"workplace-buffer no. 1. OpenCL errorcode: %d\n.", err);
			clReleaseCommandQueue(queue);   
			return B4PFM_OPENCL_ERROR; 
		}
	} else {
		clRetainMemObject(o_tmp1);
		tmp1 = o_tmp1;
	}

	/* Creates a new opencl memory buffer for temporar data */
	cl_mem tmp2;
	if(o_tmp2 == 0) {
		if(debug != B4PFM_DEBUG_NONE) printf(
			"(debug) b4pdf3D / run solver: No global workplace no. 2 given. " \
			"Creating global workplace buffer... \n");
		int tmp_mem_size = 3*POW2(solver->k1-2)*3*POW2(solver->k2-2)*POW2(solver->k3)*var_size;
		if(debug == B4PFM_DEBUG_FULL || debug == B4PFM_DEBUG_TIMING) printf(
			"(debug) b4pdf3D / run solver: Global workplace no. 2 size = %d bytes. \n",
			tmp_mem_size);
		tmp2 = clCreateBuffer(solver->context, CL_MEM_READ_WRITE, tmp_mem_size, 0, &err);
        
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf3D / run solver: Can't allocate device memory for " \
				"workplace-buffer no. 2. OpenCL errorcode: %d\n.", err);
			clReleaseMemObject(tmp1);
			clReleaseCommandQueue(queue);   
			return B4PFM_OPENCL_ERROR; 
		}
	} else {
		clRetainMemObject(o_tmp2);
		tmp2 = o_tmp2;
	}

	/* Set initial kernel arguments */
	err  = clSetKernelArg(solver->kernel_3D11, KARGS_3D11_O_F, 
		sizeof(cl_mem), (void *)&f);
	err |= clSetKernelArg(solver->kernel_3D12, KARGS_3D12_O_F, 
		sizeof(cl_mem), (void *)&f);
	err |= clSetKernelArg(solver->kernel_3D21, KARGS_3D21_O_F, 
		sizeof(cl_mem), (void *)&f);
	err |= clSetKernelArg(solver->kernel_3D22B, KARGS_3D22B_O_F, 
		sizeof(cl_mem), (void *)&f);

	err  = clSetKernelArg(solver->kernel_3D11, KARGS_3D11_O_G, 
		sizeof(cl_mem), (void *)&tmp1);
	err |= clSetKernelArg(solver->kernel_3D12, KARGS_3D12_O_G, 
		sizeof(cl_mem), (void *)&tmp1);
	err |= clSetKernelArg(solver->kernel_3D21, KARGS_3D21_O_G, 
		sizeof(cl_mem), (void *)&tmp1);
	err |= clSetKernelArg(solver->kernel_3D22A, KARGS_3D22A_O_G, 
		sizeof(cl_mem), (void *)&tmp1);
	err |= clSetKernelArg(solver->kernel_3D22B, KARGS_3D22B_O_G, 
		sizeof(cl_mem), (void *)&tmp1);

	int k1 = solver->k1;

	if(k1 & 1) {
		/* Set initial kernel arguments */
		err |= clSetKernelArg(solver->kernel_Q2_3D21, KARGS_Q2_3D21_O_F, 
			sizeof(cl_mem), (void *)&f);
		err |= clSetKernelArg(solver->kernel_Q2_3D22, KARGS_Q2_3D22_O_F, 
			sizeof(cl_mem), (void *)&f);

		err |= clSetKernelArg(solver->kernel_Q2_3D21, KARGS_Q2_3D21_O_G, 
			sizeof(cl_mem), (void *)&tmp1);
		err |= clSetKernelArg(solver->kernel_Q2_3D22, KARGS_Q2_3D22_O_G, 
			sizeof(cl_mem), (void *)&tmp1);	
	}

	if(err != CL_SUCCESS) {
		printf(
			"(error) b4pdf3D / run solver: Can't set initial kernel arguments. " \
			"OpenCL errorcode: %d\n", err);
		clReleaseMemObject(tmp2);
		clReleaseMemObject(tmp1);
		clReleaseCommandQueue(queue);
		return B4PFM_OPENCL_ERROR;
	}

	cl_event begin, end;

	if(cmd_queue_prof_enabled) {
		clFinish(queue);
		clEnqueueMarker(queue, &begin);
	}

	int r1;
	for(r1 = 1; r1 <= k1/2 - (k1 & 1 ? 0 : 1); r1++) {

		err  = clSetKernelArg(solver->kernel_3D11, KARGS_3D11_R, 
			sizeof(cl_int), (void *)&r1);
		err |= clSetKernelArg(solver->kernel_3D12, KARGS_3D12_R,
			sizeof(cl_int), (void *)&r1);
                                
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf3D / run solver: Can't set kernel arguments: stage 1 / r = %d. " \
				"OpenCL errorcode: %d\n", r1, err);
			if(cmd_queue_prof_enabled) clReleaseEvent(begin);
			clReleaseMemObject(tmp2);
			clReleaseMemObject(tmp1);
			clReleaseCommandQueue(queue);
			return B4PFM_OPENCL_ERROR;
		}

		size_t global_size[2], local_size[2];

		/* Lauches kernel  */
		global_size[0] = (POW2(k1-2*r1)-1) * 3*POW2(2*r1-2) * solver->kernel_3D11_threads;
		global_size[1] = solver->wg_per_2d_vect_11;
		local_size[0] = solver->kernel_3D11_threads;
		local_size[1] = 1;
		err = clEnqueueNDRangeKernel(queue, solver->kernel_3D11, 2, 0, 
			global_size, local_size, 0, 0, 0);
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf3D / run solver: Can't launch kernel: stage 1 / r = %d / step 1." \
				"OpenCL errorcode: %d\n", r1, err);
			if(cmd_queue_prof_enabled) clReleaseEvent(begin);
			clReleaseMemObject(tmp2);
			clReleaseMemObject(tmp1);
			clReleaseCommandQueue(queue);
			return B4PFM_OPENCL_ERROR;
		}

		err = run_2d_solver(solver->solver_2d, queue, tmp1, tmp2, (POW2(k1-2*r1)-1) * 3*POW2(2*r1-2), r1, debug == B4PFM_DEBUG_TIMING ? B4PFM_DEBUG_FULL : debug, err_info);
		if(err != B4PFM_OK) {
			printf(
				"(error) b4pdf3D / run solver: Can't launch 2D-solver: stage 1 / r = %d. " \
				"OpenCL errorcode: %d\n", r1, err);
			if(cmd_queue_prof_enabled) clReleaseEvent(begin);
			clReleaseMemObject(tmp2);
			clReleaseMemObject(tmp1);
			clReleaseCommandQueue(queue);
			return err;
		}

		int part_count = 3*POW2(2*r1-2);
		int prev_count;

		do {

			prev_count = part_count;
			part_count /= MIN(part_count, solver->max_sum_size1);

			err |= clSetKernelArg(solver->kernel_3D12, KARGS_3D12_PREV_COUNT,
				sizeof(cl_int), (void *)&prev_count);
			if(err != CL_SUCCESS) {
				printf(
					"(error) b4pdf3D / run solver: Can't set kernel arguments: stage 1 / r = %d / step 2 / kernel B. " \
					"OpenCL errorcode: %d\n", r1, err);
				if(cmd_queue_prof_enabled) clReleaseEvent(begin);
				clReleaseMemObject(tmp2);
				clReleaseMemObject(tmp1);
				clReleaseCommandQueue(queue);
				return B4PFM_OPENCL_ERROR;
			}

			global_size[0] = (POW2(k1-2*r1)-1) * part_count * solver->kernel_3D12_threads;
			global_size[1] = solver->wg_per_2d_vect_12;
			local_size[0] = solver->kernel_3D12_threads;
			local_size[1] = 1;
			err = clEnqueueNDRangeKernel(queue, solver->kernel_3D12, 2, 0, 
				global_size, local_size, 0, 0, 0);
			if(err != CL_SUCCESS) {
				printf(
					"(error) b4pdf3D / run solver: Can't launch kernel: stage 1 / r = %d / step 2 / kernel B." \
					"OpenCL errorcode: %d\n", r1, err);
				if(cmd_queue_prof_enabled) clReleaseEvent(begin);
				clReleaseMemObject(tmp2);
				clReleaseMemObject(tmp1);
				clReleaseCommandQueue(queue);
				return B4PFM_OPENCL_ERROR;
			}

		} while(part_count > 1);
	}

	if(k1 & 1) {

		cl_int k1_minus = k1 - 1;

		err  = clSetKernelArg(solver->kernel_Q2_3D21, KARGS_Q2_3D21_R, 
			sizeof(cl_int), (void *)&k1_minus);
		err |= clSetKernelArg(solver->kernel_Q2_3D22, KARGS_Q2_3D22_R,
			sizeof(cl_int), (void *)&k1_minus);
                                
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf3D / run q4-solver / q2-step: Can't set set kernel arguments: stage 2." \
				"OpenCL errorcode: %d\n", err);
			if(cmd_queue_prof_enabled) clReleaseEvent(begin);
			clReleaseMemObject(tmp2);
			clReleaseMemObject(tmp1);
			clReleaseCommandQueue(queue);
			return B4PFM_OPENCL_ERROR;
		}

		size_t global_size[2], local_size[2];

		/* Lauches kernel 1 */
		global_size[0] = POW2(k1_minus) * solver->kernel_Q2_3D21_threads;
		global_size[1] = solver->wg_per_2d_vect_q2_21;
		local_size[0] = solver->kernel_Q2_3D21_threads;
		local_size[1] = 1;
		err = clEnqueueNDRangeKernel(queue, solver->kernel_Q2_3D21, 2, 0, 
			global_size, local_size, 0, 0, 0);
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf3D / run q4-solver / q2-step: Can't launch kernel: stage 2 / step 1. " \
				"OpenCL errorcode: %d\n", err);
			if(cmd_queue_prof_enabled) clReleaseEvent(begin);
			clReleaseMemObject(tmp2);
			clReleaseMemObject(tmp1);
			clReleaseCommandQueue(queue);
			return B4PFM_OPENCL_ERROR;
		}

		err = run_2d_solver(solver->solver_2d, queue, tmp1, tmp2, POW2(k1_minus), k1, debug == B4PFM_DEBUG_TIMING ? B4PFM_DEBUG_FULL : debug, err_info);

		if(err != B4PFM_OK) {
			printf(
				"(error) b4pdf3D / run q4-solver / q2-step: Can't launch 2D-solver: stage 2. " \
				"OpenCL errorcode: %d\n", err);
			if(cmd_queue_prof_enabled) clReleaseEvent(begin);
			clReleaseMemObject(tmp2);
			clReleaseMemObject(tmp1);
			clReleaseCommandQueue(queue);
			return err;
		}

		int part_count = POW2(k1_minus);
		int prev_count;

		do {

			prev_count = part_count;
			part_count /= MIN(part_count, solver->max_sum_size_q2_2);

			err |= clSetKernelArg(solver->kernel_Q2_3D22, KARGS_Q2_3D22_PREV_COUNT,
				sizeof(cl_int), (void *)&prev_count);
			if(err != CL_SUCCESS) {
				printf(
					"(error) b4pdf3D / run q4-solver / q2-step: Can't set kernel arguments: stage 2 / step 2. " \
					"OpenCL errorcode: %d\n", err);
				if(cmd_queue_prof_enabled) clReleaseEvent(begin);
				clReleaseMemObject(tmp2);
				clReleaseMemObject(tmp1);
				clReleaseCommandQueue(queue);
				return B4PFM_OPENCL_ERROR;
			}

			global_size[0] = part_count * solver->kernel_Q2_3D22_threads;
			global_size[1] = solver->wg_per_2d_vect_q2_22;
			local_size[0] = solver->kernel_Q2_3D22_threads;
			local_size[1] = 1;
			err = clEnqueueNDRangeKernel(queue, solver->kernel_Q2_3D22, 2, 0, 
				global_size, local_size, 0, 0, 0);
			if(err != CL_SUCCESS) {
				printf(
					"(error) b4pdf3D / run q4-solver / q2-step: Can't launch kernel: stage 2 / step 2." \
					"OpenCL errorcode: %d\n", err);
				if(cmd_queue_prof_enabled) clReleaseEvent(begin);
				clReleaseMemObject(tmp2);
				clReleaseMemObject(tmp1);
				clReleaseCommandQueue(queue);
				return B4PFM_OPENCL_ERROR;
			}

		} while(part_count > 1);

	
	}


	for(r1 = k1/2-1; r1 >= 0; r1--) {

		err  = clSetKernelArg(solver->kernel_3D21, KARGS_3D21_R, 
			sizeof(cl_int), (void *)&r1);
		err |= clSetKernelArg(solver->kernel_3D22A, KARGS_3D22A_R,
			sizeof(cl_int), (void *)&r1);
		err |= clSetKernelArg(solver->kernel_3D22B, KARGS_3D22B_R,
			sizeof(cl_int), (void *)&r1);
                                
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf3D / run solver: Can't set set kernel arguments: stage 2 / r = %d." \
				"OpenCL errorcode: %d\n", r1, err);
			if(cmd_queue_prof_enabled) clReleaseEvent(begin);
			clReleaseMemObject(tmp2);
			clReleaseMemObject(tmp1);
			clReleaseCommandQueue(queue);
			return B4PFM_OPENCL_ERROR;
		}

		size_t global_size[2], local_size[2];

		/* Lauches kernel 1 */
		global_size[0] = (POW2(k1-2*r1-2)) * 3*POW2(2*r1) * solver->kernel_3D21_threads;
		global_size[1] = solver->wg_per_2d_vect_21;
		local_size[0] = solver->kernel_3D21_threads;
		local_size[1] = 1;
		err = clEnqueueNDRangeKernel(queue, solver->kernel_3D21, 2, 0, 
			global_size, local_size, 0, 0, 0);
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf3D / run solver: Can't launch kernel: stage 2 / r = %d, step 1. " \
				"OpenCL errorcode: %d\n", r1, err);
			if(cmd_queue_prof_enabled) clReleaseEvent(begin);
			clReleaseMemObject(tmp2);
			clReleaseMemObject(tmp1);
			clReleaseCommandQueue(queue);
			return B4PFM_OPENCL_ERROR;
		}

		/* FIXME r1+2? */
		err = run_2d_solver(solver->solver_2d, queue, tmp1, tmp2, 3*POW2(k1-2), r1+1, debug == B4PFM_DEBUG_TIMING ? B4PFM_DEBUG_FULL : debug, err_info);
		if(err != B4PFM_OK) {
			printf(
				"(error) b4pdf3D / run solver: Can't launch 2D-solver: stage 2 / r = %d. " \
				"OpenCL errorcode: %d\n", r1, err);
			if(cmd_queue_prof_enabled) clReleaseEvent(begin);
			clReleaseMemObject(tmp2);
			clReleaseMemObject(tmp1);
			clReleaseCommandQueue(queue);
			return err;
		}

		int part_count = 3*POW2(2*r1) / MIN(3*POW2(2*r1), solver->max_sum_size2a);

		global_size[0] = POW2(k1-2*r1-2) * part_count * solver->kernel_3D22A_threads;
		global_size[1] = solver->wg_per_2d_vect_22a;
		local_size[0] = solver->kernel_3D22A_threads;
		local_size[1] = 1;
		err = clEnqueueNDRangeKernel(queue, solver->kernel_3D22A, 2, 0, 
			global_size, local_size, 0, 0, 0);
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf3D / run solver: Can't launch kernel: stage 2 / r = %d / step 2 / kernel A." \
				"OpenCL errorcode: %d\n", r1, err);
			if(cmd_queue_prof_enabled) clReleaseEvent(begin);
			clReleaseMemObject(tmp2);
			clReleaseMemObject(tmp1);
			clReleaseCommandQueue(queue);
			return B4PFM_OPENCL_ERROR;
		}
		
		int prev_count;

		do {

			prev_count = part_count;
			part_count /= MIN(part_count, solver->max_sum_size2b);

			err |= clSetKernelArg(solver->kernel_3D22B, KARGS_3D22B_PREV_COUNT,
				sizeof(cl_int), (void *)&prev_count);
			if(err != CL_SUCCESS) {
				printf(
					"(error) b4pdf3D / run solver: Can't set kernel arguments: stage 2 / r = %d / step 2 / kernel B. " \
					"OpenCL errorcode: %d\n", r1, err);
				if(cmd_queue_prof_enabled) clReleaseEvent(begin);
				clReleaseMemObject(tmp2);
				clReleaseMemObject(tmp1);
				clReleaseCommandQueue(queue);
				return B4PFM_OPENCL_ERROR;
			}

			global_size[0] = 3*POW2(k1-2*r1-2) * part_count * solver->kernel_3D22B_threads;
			global_size[1] = solver->wg_per_2d_vect_22b;
			local_size[0] = solver->kernel_3D22B_threads;
			local_size[1] = 1;
			err = clEnqueueNDRangeKernel(queue, solver->kernel_3D22B, 2, 0, 
				global_size, local_size, 0, 0, 0);
			if(err != CL_SUCCESS) {
				printf(
					"(error) b4pdf3D / run solver: Can't launch kernel: stage 2 / r = %d / step 2 / kernel B." \
					"OpenCL errorcode: %d\n", r1, err);
				if(cmd_queue_prof_enabled) clReleaseEvent(begin);
				clReleaseMemObject(tmp2);
				clReleaseMemObject(tmp1);
				clReleaseCommandQueue(queue);
				return B4PFM_OPENCL_ERROR;
			}

		} while(part_count > 1);

	
	}

	double time;

	if(cmd_queue_prof_enabled) {
		clEnqueueMarker(queue, &end);
		err = clFinish(queue);

		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf3D / run solver: Can't finish command queue." \
				"OpenCL errorcode: %d\n",  err);
			clReleaseEvent(end);
			clReleaseEvent(begin);
			clReleaseMemObject(tmp2);
			clReleaseMemObject(tmp1);
			clReleaseCommandQueue(queue);
			return B4PFM_OPENCL_ERROR;
		}

		cl_ulong start, stop;

		err  = clGetEventProfilingInfo(begin, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &start, NULL);
		err |= clGetEventProfilingInfo(end, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &stop, NULL);

		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf3D / run solver: Can't get event profiling info. " \
				"OpenCL errorcode: %d\n",  err);
			time = -1;
		} else
			time = (stop-start)*1.0e-9;

		clReleaseEvent(end);
		clReleaseEvent(begin);

		if(!cmd_queue_prof_was_enabled) {
			clSetCommandQueueProperty(queue, CL_QUEUE_PROFILING_ENABLE, CL_FALSE, 0);

			printf(
				"(debug) b4pdf3D / run solver: OpenCL command queue profiling disabled. \n");
		}
	}

	clReleaseMemObject(tmp2);
	clReleaseMemObject(tmp1);
	clReleaseCommandQueue(queue);

	if(debug != B4PFM_DEBUG_NONE && !cmd_queue_prof_enabled) printf(
		"(debug) b4pdf3D / run solver: Ready.\n");

	if(cmd_queue_prof_enabled) printf(
		"(debug) b4pdf3D / run solver: Ready. Computing time: %fs\n", time);

	return B4PFM_OK;

}

int run_3d_q2_solver(r_b4pfm3D *solver, cl_command_queue o_queue, cl_mem f, cl_mem o_tmp1, cl_mem o_tmp2, int o_r1, int debug, int *o_err_info) {

	int *err_info;
	int tmp_err_info;
	if(o_err_info == 0)
		err_info = &tmp_err_info;
	else
		err_info = o_err_info;

	if(solver == NULL)
		return B4PFM_INVALID_SOLVER;


	else if(debug != B4PFM_DEBUG_NONE) printf(
		"(debug) b4pdf3D / run q2-solver: Begin... \n");

	int err;

	int var_size = solver->double_prec ? sizeof(cl_double) : sizeof(cl_float);

	int cmd_queue_prof_was_enabled = 0;
	int cmd_queue_prof_enabled = 0;

	/* Creates a new opencl command queue if necessary. */
	cl_command_queue queue;
	if(o_queue == 0) {
		if(debug != B4PFM_DEBUG_NONE) printf(
			"(debug) b4pdf3D / run q2-solver: No OpenCL Commandqueue given. "\
			"Creating OpenCL Commandqueue... \n");
		cl_device_id device;
 		err = clGetContextInfo(solver->context, CL_CONTEXT_DEVICES, 
			sizeof(cl_device_id), &device, 0);
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf3D / run q2-solver: Can't query device id from context. " \
				"OpenCL errorcode: %d\n", err);
			return B4PFM_OPENCL_ERROR;
		}

		if(debug == B4PFM_DEBUG_TIMING) {
			queue = clCreateCommandQueue(solver->context, device, CL_QUEUE_PROFILING_ENABLE, &err);
			printf(
				"(debug) b4pdf3D / run q2-solver: OpenCL command queue profiling enabled. \n");
		} else
			queue = clCreateCommandQueue(solver->context, device, 0, &err);
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf3D / run q2-solver: Can't create command queue. " \
				"OpenCL errorcode: %d\n.", err);
			return B4PFM_OPENCL_ERROR;
		}
	} else {

		queue = o_queue;
		clRetainCommandQueue(queue);

		if(debug == B4PFM_DEBUG_TIMING) {
			cl_command_queue_properties old_cmd_queue_pro;
			err = clSetCommandQueueProperty(queue, CL_QUEUE_PROFILING_ENABLE, CL_TRUE, &old_cmd_queue_pro);

			cmd_queue_prof_was_enabled = old_cmd_queue_pro & CL_QUEUE_PROFILING_ENABLE;

			if(err == CL_SUCCESS)
				cmd_queue_prof_enabled = 1;
			else
				printf(
					"(error) b4pdf3D / run q2-solver: Can't enable OpenCL command queue profiling. " 
					"OpenCL errorcode: %d.\n", err);

			if(cmd_queue_prof_enabled && !cmd_queue_prof_was_enabled)
				printf(
					"(debug) b4pdf3D / run q2-solver: OpenCL command queue profiling enabled. \n");
		}

	}

	/* Creates a new opencl memory buffer for temporar data */
	cl_mem tmp1;
	if(o_tmp1 == 0) {
		if(debug != B4PFM_DEBUG_NONE) printf(
			"(debug) b4pdf3D / run q2-solver: No global workplace no. 1 given. " \
			"Creating global workplace buffer... \n");
		int tmp_mem_size = 3*POW2(solver->k1-2)*(POW2(solver->k2)-1)*POW2(solver->k3)*var_size;
		if(debug == B4PFM_DEBUG_FULL || debug == B4PFM_DEBUG_TIMING) printf(
			"(debug) b4pdf3D / run q2-solver: Global workplace no. 1 size = %d bytes. \n",
			tmp_mem_size);
		tmp1 = clCreateBuffer(solver->context, CL_MEM_READ_WRITE, tmp_mem_size, 0, &err);
        
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf3D / run q2-solver: Can't allocate device memory for " \
				"workplace-buffer no. 1. OpenCL errorcode: %d\n.", err);
			clReleaseCommandQueue(queue);   
			return B4PFM_OPENCL_ERROR; 
		}
	} else {
		clRetainMemObject(o_tmp1);
		tmp1 = o_tmp1;
	}

	/* Creates a new opencl memory buffer for temporar data */
	cl_mem tmp2;
	if(o_tmp2 == 0) {
		if(debug != B4PFM_DEBUG_NONE) printf(
			"(debug) b4pdf3D / run q2-solver: No global workplace no. 2 given. " \
			"Creating global workplace buffer... \n");
		int tmp_mem_size = 3*POW2(solver->k1-2)*3*POW2(solver->k2-2)*POW2(solver->k3)*var_size;
		if(debug == B4PFM_DEBUG_FULL || debug == B4PFM_DEBUG_TIMING) printf(
			"(debug) b4pdf3D / run q2-solver: Global workplace no. 2 size = %d bytes. \n",
			tmp_mem_size);
		tmp2 = clCreateBuffer(solver->context, CL_MEM_READ_WRITE, tmp_mem_size, 0, &err);
        
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf3D / run q2-solver: Can't allocate device memory for " \
				"workplace-buffer no. 2. OpenCL errorcode: %d\n.", err);
			clReleaseMemObject(tmp1);
			clReleaseCommandQueue(queue);   
			return B4PFM_OPENCL_ERROR; 
		}
	} else {
		clRetainMemObject(o_tmp2);
		tmp2 = o_tmp2;
	}

	/* Set initial kernel arguments */
	err  = clSetKernelArg(solver->kernel_Q2_3D11, KARGS_Q2_3D11_O_F, 
		sizeof(cl_mem), (void *)&f);
	err |= clSetKernelArg(solver->kernel_Q2_3D12, KARGS_Q2_3D12_O_F, 
		sizeof(cl_mem), (void *)&f);
	err |= clSetKernelArg(solver->kernel_Q2_3D21, KARGS_Q2_3D21_O_F, 
		sizeof(cl_mem), (void *)&f);
	err |= clSetKernelArg(solver->kernel_Q2_3D22, KARGS_Q2_3D22_O_F, 
		sizeof(cl_mem), (void *)&f);

	err  = clSetKernelArg(solver->kernel_Q2_3D11, KARGS_Q2_3D11_O_G, 
		sizeof(cl_mem), (void *)&tmp1);
	err |= clSetKernelArg(solver->kernel_Q2_3D12, KARGS_Q2_3D12_O_G, 
		sizeof(cl_mem), (void *)&tmp1);
	err |= clSetKernelArg(solver->kernel_Q2_3D21, KARGS_Q2_3D21_O_G, 
		sizeof(cl_mem), (void *)&tmp1);
	err |= clSetKernelArg(solver->kernel_Q2_3D22, KARGS_Q2_3D22_O_G, 
		sizeof(cl_mem), (void *)&tmp1);

	if(err != CL_SUCCESS) {
		printf(
			"(error) b4pdf3D / run q2-solver: Can't set initial kernel arguments. " \
			"OpenCL errorcode: %d\n", err);
		clReleaseMemObject(tmp2);
		clReleaseMemObject(tmp1);
		clReleaseCommandQueue(queue);
		return B4PFM_OPENCL_ERROR;
	}

	int k1 = solver->k1;

	cl_event begin, end;

	if(cmd_queue_prof_enabled) {
		clFinish(queue);
		clEnqueueMarker(queue, &begin);
	}

	int r1;
	for(r1 = o_r1; r1 <= k1 - 1; r1++) {

		err  = clSetKernelArg(solver->kernel_Q2_3D11, KARGS_Q2_3D11_R, 
			sizeof(cl_int), (void *)&r1);
		err |= clSetKernelArg(solver->kernel_Q2_3D12, KARGS_Q2_3D12_R,
			sizeof(cl_int), (void *)&r1);
                                
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf3D / run q2-solver: Can't set kernel arguments: stage 1 / r = %d. " \
				"OpenCL errorcode: %d\n", r1, err);
			if(cmd_queue_prof_enabled) clReleaseEvent(begin);
			clReleaseMemObject(tmp2);
			clReleaseMemObject(tmp1);
			clReleaseCommandQueue(queue);
			return B4PFM_OPENCL_ERROR;
		}

		size_t global_size[2], local_size[2];

		/* Lauches kernel  */
		global_size[0] = (POW2(k1-r1)-1) * POW2(r1-1) * solver->kernel_Q2_3D11_threads;
		global_size[1] = solver->wg_per_2d_vect_q2_11;
		local_size[0] = solver->kernel_Q2_3D11_threads;
		local_size[1] = 1;
		err = clEnqueueNDRangeKernel(queue, solver->kernel_Q2_3D11, 2, 0, 
			global_size, local_size, 0, 0, 0);
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf3D / run q2-solver: Can't launch kernel: stage 1 / r = %d / step 1." \
				"OpenCL errorcode: %d\n", r1, err);
			if(cmd_queue_prof_enabled) clReleaseEvent(begin);
			clReleaseMemObject(tmp2);
			clReleaseMemObject(tmp1);
			clReleaseCommandQueue(queue);
			return B4PFM_OPENCL_ERROR;
		}

		err = run_2d_q2_solver(solver->solver_2d, queue, tmp1, tmp2, (POW2(k1-r1)-1) * POW2(r1-1), r1, 1, debug == B4PFM_DEBUG_TIMING ? B4PFM_DEBUG_FULL : debug, err_info);
		if(err != B4PFM_OK) {
			printf(
				"(error) b4pdf3D / run q2-solver: Can't launch 2D-solver: stage 1 / r = %d. " \
				"OpenCL errorcode: %d\n", r1, err);
			if(cmd_queue_prof_enabled) clReleaseEvent(begin);
			clReleaseMemObject(tmp2);
			clReleaseMemObject(tmp1);
			clReleaseCommandQueue(queue);
			return err;
		}

		int part_count = POW2(r1-1);
		int prev_count;

		do {

			prev_count = part_count;
			part_count /= MIN(part_count, solver->max_sum_size_q2_1);

			err |= clSetKernelArg(solver->kernel_Q2_3D12, KARGS_Q2_3D12_PREV_COUNT,
				sizeof(cl_int), (void *)&prev_count);
			if(err != CL_SUCCESS) {
				printf(
					"(error) b4pdf3D / run q2-solver: Can't set kernel arguments: stage 1 / r = %d / step 2. " \
					"OpenCL errorcode: %d\n", r1, err);
				if(cmd_queue_prof_enabled) clReleaseEvent(begin);
				clReleaseMemObject(tmp2);
				clReleaseMemObject(tmp1);
				clReleaseCommandQueue(queue);
				return B4PFM_OPENCL_ERROR;
			}

			global_size[0] = (POW2(k1-r1)-1) * part_count * solver->kernel_Q2_3D12_threads;
			global_size[1] = solver->wg_per_2d_vect_q2_12;
			local_size[0] = solver->kernel_Q2_3D12_threads;
			local_size[1] = 1;
			err = clEnqueueNDRangeKernel(queue, solver->kernel_Q2_3D12, 2, 0, 
				global_size, local_size, 0, 0, 0);
			if(err != CL_SUCCESS) {
				printf(
					"(error) b4pdf3D / run q2-solver: Can't launch kernel: stage 1 / r = %d / step 2." \
					"OpenCL errorcode: %d\n", r1, err);
				if(cmd_queue_prof_enabled) clReleaseEvent(begin);
				clReleaseMemObject(tmp2);
				clReleaseMemObject(tmp1);
				clReleaseCommandQueue(queue);
				return B4PFM_OPENCL_ERROR;
			}

		} while(part_count > 1);
	}

	for(r1 = k1-1; r1 >= o_r1-1; r1--) {

		err  = clSetKernelArg(solver->kernel_Q2_3D21, KARGS_Q2_3D21_R, 
			sizeof(cl_int), (void *)&r1);
		err |= clSetKernelArg(solver->kernel_Q2_3D22, KARGS_Q2_3D22_R,
			sizeof(cl_int), (void *)&r1);
                                
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf3D / run q2-solver: Can't set set kernel arguments: stage 2 / r = %d." \
				"OpenCL errorcode: %d\n", r1, err);
			if(cmd_queue_prof_enabled) clReleaseEvent(begin);
			clReleaseMemObject(tmp2);
			clReleaseMemObject(tmp1);
			clReleaseCommandQueue(queue);
			return B4PFM_OPENCL_ERROR;
		}

		size_t global_size[2], local_size[2];

		/* Lauches kernel 1 */
		global_size[0] = (POW2(k1-r1-1)) * POW2(r1) * solver->kernel_Q2_3D21_threads;
		global_size[1] = solver->wg_per_2d_vect_q2_21;
		local_size[0] = solver->kernel_Q2_3D21_threads;
		local_size[1] = 1;
		err = clEnqueueNDRangeKernel(queue, solver->kernel_Q2_3D21, 2, 0, 
			global_size, local_size, 0, 0, 0);
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf3D / run q2-solver: Can't launch kernel: stage 2 / r = %d, step 1. " \
				"OpenCL errorcode: %d\n", r1, err);
			if(cmd_queue_prof_enabled) clReleaseEvent(begin);
			clReleaseMemObject(tmp2);
			clReleaseMemObject(tmp1);
			clReleaseCommandQueue(queue);
			return B4PFM_OPENCL_ERROR;
		}

		err = run_2d_q2_solver(solver->solver_2d, queue, tmp1, tmp2, POW2(k1-1), r1+1, 1, debug == B4PFM_DEBUG_TIMING ? B4PFM_DEBUG_FULL : debug, err_info);

		if(err != B4PFM_OK) {
			printf(
				"(error) b4pdf3D / run q2-solver: Can't launch 2D-solver: stage 2 / r = %d. " \
				"OpenCL errorcode: %d\n", r1, err);
			if(cmd_queue_prof_enabled) clReleaseEvent(begin);
			clReleaseMemObject(tmp2);
			clReleaseMemObject(tmp1);
			clReleaseCommandQueue(queue);
			return err;
		}

		int part_count = POW2(r1);
		int prev_count;

		do {

			prev_count = part_count;
			part_count /= MIN(part_count, solver->max_sum_size_q2_2);

			err |= clSetKernelArg(solver->kernel_Q2_3D22, KARGS_Q2_3D22_PREV_COUNT,
				sizeof(cl_int), (void *)&prev_count);
			if(err != CL_SUCCESS) {
				printf(
					"(error) b4pdf3D / run q2-solver: Can't set kernel arguments: stage 2 / r = %d / step 2. " \
					"OpenCL errorcode: %d\n", r1, err);
				if(cmd_queue_prof_enabled) clReleaseEvent(begin);
				clReleaseMemObject(tmp2);
				clReleaseMemObject(tmp1);
				clReleaseCommandQueue(queue);
				return B4PFM_OPENCL_ERROR;
			}

			global_size[0] = POW2(k1-r1-1) * part_count * solver->kernel_Q2_3D22_threads;
			global_size[1] = solver->wg_per_2d_vect_q2_22;
			local_size[0] = solver->kernel_Q2_3D22_threads;
			local_size[1] = 1;
			err = clEnqueueNDRangeKernel(queue, solver->kernel_Q2_3D22, 2, 0, 
				global_size, local_size, 0, 0, 0);
			if(err != CL_SUCCESS) {
				printf(
					"(error) b4pdf3D / run q2-solver: Can't launch kernel: stage 2 / r = %d / step 2." \
					"OpenCL errorcode: %d\n", r1, err);
				if(cmd_queue_prof_enabled) clReleaseEvent(begin);
				clReleaseMemObject(tmp2);
				clReleaseMemObject(tmp1);
				clReleaseCommandQueue(queue);
				return B4PFM_OPENCL_ERROR;
			}

		} while(part_count > 1);

	
	}

	double time;

	if(cmd_queue_prof_enabled) {
		clEnqueueMarker(queue, &end);
		err = clFinish(queue);

		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf3D / run q2-solver: Can't finish command queue." \
				"OpenCL errorcode: %d\n",  err);
			clReleaseEvent(end);
			clReleaseEvent(begin);
			clReleaseMemObject(tmp2);
			clReleaseMemObject(tmp1);
			clReleaseCommandQueue(queue);
			return B4PFM_OPENCL_ERROR;
		}

		cl_ulong start, stop;

		err  = clGetEventProfilingInfo(begin, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &start, NULL);
		err |= clGetEventProfilingInfo(end, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &stop, NULL);

		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf3D / run q2-solver: Can't get event profiling info. " \
				"OpenCL errorcode: %d\n",  err);
			time = -1;
		} else
			time = (stop-start)*1.0e-9;

		clReleaseEvent(end);
		clReleaseEvent(begin);

		if(!cmd_queue_prof_was_enabled) {
			clSetCommandQueueProperty(queue, CL_QUEUE_PROFILING_ENABLE, CL_FALSE, 0);

			printf(
				"(debug) b4pdf3D / run q2-solver: OpenCL command queue profiling disabled. \n");
		}
	}

	clReleaseMemObject(tmp2);
	clReleaseMemObject(tmp1);
	clReleaseCommandQueue(queue);

	if(debug != B4PFM_DEBUG_NONE && !cmd_queue_prof_enabled) printf(
		"(debug) b4pdf3D / run q2-solver: Ready.\n");

	if(cmd_queue_prof_enabled) printf(
		"(debug) b4pdf3D / run q2-solver: Ready. Computing time: %fs\n", time);

	return B4PFM_OK;

}


int load_and_run_3d_solver(r_b4pfm3D *solver, void *f, int debug, int *o_err_info) {

	int *err_info;
	int tmp_err_info;
	if(o_err_info == 0)
		err_info = &tmp_err_info;
	else
		err_info = o_err_info;

	int err;

	int n1 = POW2(solver->k1)-1;
	int n2 = POW2(solver->k2)-1;
	int ldf = solver->ldf;

	if(solver == NULL)
		return B4PFM_INVALID_SOLVER;

	cl_device_id device;
	err = clGetContextInfo(solver->context, CL_CONTEXT_DEVICES, 
		sizeof(cl_device_id), &device, 0);
                
	if(err != CL_SUCCESS) {
		printf(
			"(error) b4pdf3D / pre-run solver: Can't query device id from context. " \
			"OpenCL errorcode: %d\n", err);
		return B4PFM_OPENCL_ERROR;
	}

	int cmd_queue_prof_enabled = 0;

	cl_command_queue queue;
                
	if(debug == B4PFM_DEBUG_TIMING) {
		queue = clCreateCommandQueue(solver->context, device, CL_QUEUE_PROFILING_ENABLE, &err);
		printf(
			"(debug) b4pdf3D / pre-run solver: OpenCL command queue profiling enabled. \n");
		cmd_queue_prof_enabled = 1;
	} else
		queue = clCreateCommandQueue(solver->context, device, 0, &err);

	if(err != CL_SUCCESS) {
		printf(
			"(error) b4pdf3D / pre-run solver: Can't create command queue. " \
			"OpenCL errorcode: %d\n.", err);
		return B4PFM_OPENCL_ERROR;
	}

	cl_event begin, end;

	if(cmd_queue_prof_enabled) {
		clFinish(queue);
		clEnqueueMarker(queue, &begin);
	}

	int var_size = solver->double_prec ? sizeof(cl_double) : sizeof(cl_float);

	cl_mem f_buffer = clCreateBuffer(solver->context, CL_MEM_READ_WRITE, n1*n2*ldf*var_size, 0, &err);
        
	if(err != CL_SUCCESS) {
		printf(
			"(error) b4pdf3D / pre-run solver: Can't allocate device memory for " \
			"f-buffer. OpenCL errorcode: %d\n.", err);
		if(cmd_queue_prof_enabled) clReleaseEvent(begin);
		clReleaseCommandQueue(queue);   
		return B4PFM_OPENCL_ERROR; 
	}

	err = clEnqueueWriteBuffer(queue, f_buffer, CL_TRUE, 0, n1*n2*ldf*var_size, f, 0, 0, 0);

	if(err != CL_SUCCESS) {
		printf(
			"(error) b4pdf3D / pre-run solver: Can't write into f-buffer. " \
			"OpenCL errorcode: %d\n.", err);
		if(cmd_queue_prof_enabled) clReleaseEvent(begin);
		clReleaseCommandQueue(queue);
		clReleaseMemObject(f_buffer);
		return B4PFM_OPENCL_ERROR;
	}

	int err_info2;
	if(solver->force_radix2)
		err = run_3d_q2_solver(solver, queue, f_buffer, 0, 0, 1, debug, &err_info2);
	else
		err = run_3d_solver(solver, queue, f_buffer, 0, 0, debug, &err_info2);

	if(err != B4PFM_OK) {
		if(cmd_queue_prof_enabled) clReleaseEvent(begin);
		clReleaseCommandQueue(queue);
		clReleaseMemObject(f_buffer);
		*err_info = err_info2;
		return err;
	}

	err = clFinish(queue);

	if(err != CL_SUCCESS) {
		printf(
			"(error) b4pdf3D / pre-run solver: Can't finish workqueue. " \
			"OpenCL errorcode: %d\n", err);
		if(cmd_queue_prof_enabled) clReleaseEvent(begin);
		clReleaseCommandQueue(queue);
		clReleaseMemObject(f_buffer);
		return B4PFM_OPENCL_ERROR;
	}

	err = clEnqueueReadBuffer(queue, f_buffer, CL_TRUE, 0, n1*n2*ldf*var_size, f, 0, 0, 0);
        
	if(err != CL_SUCCESS) {
		printf(
			"(error) b4pdf3D / pre-run solver: Can't read from f-buffer. " \
			"OpenCL errorcode: %d\n.", err);
		if(cmd_queue_prof_enabled) clReleaseEvent(begin);
		clReleaseCommandQueue(queue);
		clReleaseMemObject(f_buffer);
		return B4PFM_OPENCL_ERROR;
	}

	double time;

	if(cmd_queue_prof_enabled) {
		clEnqueueMarker(queue, &end);
		err = clFinish(queue);

		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf3D / pre-run solver: Can't finish command queue." \
				"OpenCL errorcode: %d\n",  err);
			clReleaseEvent(end);
			clReleaseEvent(begin);
			clReleaseMemObject(f_buffer);
			clReleaseCommandQueue(queue);
			return B4PFM_OPENCL_ERROR;
		}

		cl_ulong start, stop;

		err  = clGetEventProfilingInfo(begin, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &start, NULL);
		err |= clGetEventProfilingInfo(end, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &stop, NULL);

		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf3D / pre-run solver: Can't get event profiling info. " \
				"OpenCL errorcode: %d\n",  err);
			time = -1;
		} else
			time = (stop-start)*1.0e-9;

		clReleaseEvent(end);
		clReleaseEvent(begin);
	}

	clReleaseCommandQueue(queue);
	clReleaseMemObject(f_buffer);

	if(cmd_queue_prof_enabled) printf(
		"(debug) b4pdf3D / pre-run solver: Total time: %fs\n", time);

	return B4PFM_OK;
}

int default_opt_3d_solver(cl_context context, b4pfm3D_params* opt_params, int k1, int k2, int k3, int ldf, int prec, int debug, int *o_err_info) {

	int *err_info;
	int tmp_err_info;
	if(o_err_info == 0)
		err_info = &tmp_err_info;
	else
		err_info = o_err_info;

	int err;

	

	/* Find the OpenCL device corresponding to given context */
	cl_device_id device;

	err = clGetContextInfo(context, CL_CONTEXT_DEVICES, 
		sizeof(cl_device_id), &device, 0);

	if(err != CL_SUCCESS) {
		printf(
			"(error) b4pdf3D / default opt.: Can't query device id from context. " \
			"OpenCL errorcode: %d\n", err);
		return B4PFM_OPENCL_ERROR;
	}

	/* Check whether the device is nvidia */
	char *vendor = malloc(100*sizeof(char));
	clGetDeviceInfo(device, CL_DEVICE_VENDOR, 100, vendor, 0);
	int nvidia = strstr(vendor, "NVIDIA") != 0;
	free(vendor);

	/* Find optimal data alignment */
	if(nvidia) {
		if(debug != B4PFM_DEBUG_NONE) printf(
			"(debug) b4pdf3D / default opt.: nVidia videocard found.\n");
		opt_params->work_block_size = 32;
	} else {
		opt_params->work_block_size = 64;
	}

	/* Find out the amount of local memory in device */
	cl_ulong max_local_mem_size;
	clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, 
		sizeof(cl_ulong), &max_local_mem_size, 0);

	if(debug == B4PFM_DEBUG_FULL || debug == B4PFM_DEBUG_TIMING) printf(
		"(debug) b4pdf2D / default opt.: OpenCL device's local memory size = %d\n",
		(int) max_local_mem_size);

	/* Determine maximum work-group size for cyclic solver */
	size_t max_work_item_sizes[3];
	clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, 
		sizeof(max_work_item_sizes), max_work_item_sizes, 0);

	if(debug == B4PFM_DEBUG_FULL || debug == B4PFM_DEBUG_TIMING) printf(
		"(debug) b4pdf2D / default opt.: OpenCL device's max workgroup size = %d\n",
		(int) max_work_item_sizes[0]);

	/* Find optimal data alignment */
	if(nvidia) {
		if(prec) {
			if(POW2(k3) >= 64)	opt_params->vector_width = 2;
			else 				opt_params->vector_width = 1;
		} else {
			if(POW2(k3) >= 128)	opt_params->vector_width = 4;
			else if(ldf >= 64)	opt_params->vector_width = 2;
			else				opt_params->vector_width = 1;
		}
	} else  {
		if(prec) {
			if(POW2(k3) >= 128)		opt_params->vector_width = 2;
			else 					opt_params->vector_width = 1;
		} else {
			if(POW2(k3) >= 256) 		opt_params->vector_width = 4;
			else if(POW2(k3) >= 128)	opt_params->vector_width = 2;
			else 						opt_params->vector_width = 1;
		}
	}

	while(opt_params->vector_width > 1 && POW2(k3) % (opt_params->vector_width*opt_params->work_block_size) != 0)
		opt_params->vector_width /= 2;

	opt_params->other_wg_size_11 = POW2(k3)/opt_params->vector_width;
	while(opt_params->other_wg_size_11 > max_work_item_sizes[0])
		opt_params->other_wg_size_11 /= 2*opt_params->vector_width;
	if(opt_params->other_wg_size_11 < opt_params->work_block_size)
		opt_params->other_wg_size_11 = opt_params->work_block_size;
	opt_params->other_wg_size_12 = opt_params->other_wg_size_11;
	opt_params->other_wg_size_21 = opt_params->other_wg_size_11;
	opt_params->other_wg_size_22a = opt_params->other_wg_size_11;
	opt_params->other_wg_size_22b = opt_params->other_wg_size_11;

	opt_params->other_wg_size_q2_11 = opt_params->other_wg_size_11;
	opt_params->other_wg_size_q2_12 = opt_params->other_wg_size_11;
	opt_params->other_wg_size_q2_21 = opt_params->other_wg_size_11;
	opt_params->other_wg_size_q2_22 = opt_params->other_wg_size_11;

	opt_params->wg_per_2d_vect_11  = 1;
	opt_params->wg_per_2d_vect_12 = 1;
	opt_params->wg_per_2d_vect_21  = 1;
	opt_params->wg_per_2d_vect_22a = 1;
	opt_params->wg_per_2d_vect_22b = 1;

	opt_params->wg_per_2d_vect_q2_11  = 1;
	opt_params->wg_per_2d_vect_q2_12 = 1;
	opt_params->wg_per_2d_vect_q2_21  = 1;
	opt_params->wg_per_2d_vect_q2_22 = 1;

	opt_params->max_sum_size1 = 4;
	opt_params->max_sum_size2a = 4;
	opt_params->max_sum_size2b = 4;	

	opt_params->max_sum_size_q2_1 = 2;
	opt_params->max_sum_size_q2_2 = 2;

	err = default_opt_2d_solver(context, &opt_params->params_2d, k2, k3, ldf, prec, debug, err_info);

	if(err != B4PFM_OK)
		return err;

	if(debug == B4PFM_DEBUG_FULL || debug == B4PFM_DEBUG_TIMING) { 
		printf("(debug) b4pdf3D / default opt.: Final opt_params = \n");
		print_opt_params_3d(opt_params);
	}


	return B4PFM_OK;
}

double run_speed_test_3d(cl_context context, cl_command_queue queue, cl_mem tmp1, cl_mem tmp2, cl_mem tmp3, b4pfm3D_params opt_params, int force_radix2, int k1, int k2, int k3, int ldf, int prec, int debug) {

	if(debug == B4PFM_DEBUG_FULL || debug == B4PFM_DEBUG_TIMING) {
		printf("(debug) b4pdf3D / run_speed_test_3d:Current opt_params = \n");
		print_opt_params_3d(&opt_params);
		print_opt_params_2d(&opt_params.params_2d);
	}

	int err;
	r_b4pfm3D* solver = init_3d_solver(context, k1, k2, k3, ldf, prec, opt_params, force_radix2, B4PFM_DEBUG_NONE, &err, 0);
	if(err != B4PFM_OK)
		return 1.0 / 0.0;

	cl_device_id device;
 	err = clGetContextInfo(context, CL_CONTEXT_DEVICES, 
		sizeof(cl_device_id), &device, 0);
	if(err != CL_SUCCESS) {
		printf(
			"(error) b4pdf3D / run_speed_test_3d: Can't query device id from context. " \
			"OpenCL errorcode: %d\n", err);
		return 1.0 / 0.0;
	}
	
	/* Pre-init kernels... */
	if(force_radix2)
		err = run_3d_q2_solver(solver, queue, tmp1, tmp2, tmp3, 1, B4PFM_DEBUG_NONE, 0);
	else
		err = run_3d_solver(solver, queue, tmp1, tmp2, tmp3, B4PFM_DEBUG_NONE, 0);
	if(err != B4PFM_OK) {
		free_3d_solver(solver);
		return 1.0 / 0.0;
	}

	clFinish(queue);

	cl_ulong start, stop;

	cl_event begin, end;

	clEnqueueMarker(queue, &begin);

	if(force_radix2)
		err = run_3d_q2_solver(solver, queue, tmp1, tmp2, tmp3, 1, B4PFM_DEBUG_NONE, 0);
	else
		err = run_3d_solver(solver, queue, tmp1, tmp2, tmp3, B4PFM_DEBUG_NONE, 0);
	if(err != B4PFM_OK) {
		clReleaseEvent(begin);
		free_3d_solver(solver);
		return 1.0 / 0.0;
	}

	clEnqueueMarker(queue, &end);

	err = clFinish(queue);
	if(err != CL_SUCCESS) {
		printf("(error) b4pdf3D / run_speed_test_3d: Can't finish command queue. OpenCL errorcode: %d\n", err);
		return 1.0 / 0.0;
	}

	err  = clGetEventProfilingInfo(begin, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &start, NULL);
	err |= clGetEventProfilingInfo(end, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &stop, NULL);

	double final;

	if(err != CL_SUCCESS) {
		printf(
			"(error) b4pdf3D / run solver: Can't get event profiling info. " \
			"OpenCL errorcode: %d\n",  err);
		clReleaseEvent(end);	
		clReleaseEvent(begin);
		free_3d_solver(solver);
		return 1.0 / 0.0;
	} else
		final = (stop-start)*1.0e-9;

	clReleaseEvent(end);	
	clReleaseEvent(begin);
	free_3d_solver(solver);

	if(debug == B4PFM_DEBUG_FULL || debug == B4PFM_DEBUG_TIMING) 
		printf("Current time: %f\n", final);

	return final;
}

int auto_optimize_3d_solver(cl_context o_context, cl_mem o_tmp1, cl_mem o_tmp2, cl_mem o_tmp3, b4pfm3D_params* opt_params, int force_radix2, int k1, int k2, int k3, int ldf, int prec, int debug, int *o_err_info) {

	int *err_info;
	int tmp_err_info;
	if(o_err_info == 0)
		err_info = &tmp_err_info;
	else
		err_info = o_err_info;

	int err;

	int n1 = POW2(k1)-1;
	int n2 = POW2(k2)-1;

	cl_context context;

	/* Create OpenCL context  */
	if(o_context == 0) {
		if(debug != B4PFM_DEBUG_NONE) printf(
			"(debug) b4pdf3D / auto opt.: No OpenCL context given. " \
			"Creating an OpenCL context... \n");

		cl_platform_id platform;
		cl_uint num;
		err = clGetPlatformIDs(1, &platform, &num);
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf3D / auto opt.: Can't query OpenCL platform. " \
				"OpenCL errorcode: %d\n", err);
			return B4PFM_OPENCL_ERROR;
		}

		if(num < 1) {
			printf(
				"(error) b4pdf3D / auto opt.: No OpenCL platforms. \n");
			return B4PFM_OPENCL_ERROR;
		}
                       
		cl_device_id device;
        
		err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, 0);
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf3D / auto opt.: Can't query OpenCL devices. " \
				"OpenCL errorcode: %d\n", err);
			return B4PFM_OPENCL_ERROR;
		}

		context = clCreateContext(0, 1, &device, 0, 0, &err);
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf3D / auto opt.: Can't create OpenCL context. " \
				"OpenCL errorcode: %d\n", err);
			return B4PFM_OPENCL_ERROR;
		}
	} else {
		context = o_context;
	}

	cl_device_id device;
	err = clGetContextInfo(context, CL_CONTEXT_DEVICES, 
		sizeof(cl_device_id), &device, 0);
                
	if(err != CL_SUCCESS) {
		printf(
			"(error) b4pdf3D / auto opt.: Can't query device id from context. " \
			"OpenCL errorcode: %d\n", err);
		clReleaseContext(context);
		return B4PFM_OPENCL_ERROR;
	}

	int var_size = prec ? sizeof(cl_double) : sizeof(cl_float);

	cl_mem tmp1, tmp2, tmp3;

	if(o_tmp1 == 0) {
		if(debug != B4PFM_DEBUG_NONE) printf(
			"(debug) b4pdf3D / auto opt.: No tmp1-buffer given. " \
			"Creating tmp1-buffer... \n");

		tmp1 = clCreateBuffer(context, CL_MEM_READ_WRITE, n1*n2*ldf*var_size, 0, &err);
        
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf3D / auto opt.: Can't allocate device memory for " \
				"tmp1-buffer. OpenCL errorcode: %d\n.", err);
			clReleaseContext(context); 
			return B4PFM_OPENCL_ERROR; 
		}

		if(debug == B4PFM_DEBUG_FULL || debug == B4PFM_DEBUG_TIMING) printf(
			"(debug) b4pdf2D / auto opt.: tmp1-buffer size = %d bytes. \n",
			n2*ldf*var_size);

	} else {
		clRetainMemObject(o_tmp1);
		tmp1 = o_tmp1;
	}


	if(o_tmp2 == 0) {
		if(debug != B4PFM_DEBUG_NONE) printf(
			"(debug) b4pdf3D / auto opt.: No tmp2-buffer given. " \
			"Creating tmp2-buffer... \n");
		int tmp_mem_size = 3*POW2(k1-2)*POW2(k2)*POW2(k3)*var_size;

		tmp2 = clCreateBuffer(context, CL_MEM_READ_WRITE, tmp_mem_size, 0, &err);
        
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf3D / auto opt.: Can't allocate device memory for " \
				"tmp2-buffer. OpenCL errorcode: %d\n", err);
			clReleaseMemObject(tmp1); 
			clReleaseContext(context);  
			return B4PFM_OPENCL_ERROR; 
		}		

		if(debug == B4PFM_DEBUG_FULL || debug == B4PFM_DEBUG_TIMING) printf(
			"(debug) b4pdf3D / auto opt.: tmp2-buffer size = %d bytes. \n",
			tmp_mem_size);

	} else {
		clRetainMemObject(o_tmp2);
		tmp2 = o_tmp2;
	}

	if(o_tmp3 == 0) {
		if(debug != B4PFM_DEBUG_NONE) printf(
			"(debug) b4pdf3D / auto opt.: No tmp3-buffer given. " \
			"Creating tmp2-buffer... \n");
		int tmp_mem_size = 3*POW2(k1-2)*3*POW2(k2-2)*POW2(k3)*var_size;

		tmp3 = clCreateBuffer(context, CL_MEM_READ_WRITE, tmp_mem_size, 0, &err);
        
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf3D / auto opt.: Can't allocate device memory for " \
				"tmp3-buffer. OpenCL errorcode: %d\n", err);
			clReleaseMemObject(tmp1); 
			clReleaseContext(context);  
			return B4PFM_OPENCL_ERROR; 
		}		

		if(debug == B4PFM_DEBUG_FULL || debug == B4PFM_DEBUG_TIMING) printf(
			"(debug) b4pdf3D / auto opt.: tmp3-buffer size = %d bytes. \n",
			tmp_mem_size);

	} else {
		clRetainMemObject(o_tmp3);
		tmp3 = o_tmp3;
	}

	cl_command_queue queue;
	queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
	if(err != CL_SUCCESS) {
		printf(
			"(error) b4pdf3D / auto opt.: Can't create command queue. " \
			"OpenCL errorcode: %d\n", err);
		return B4PFM_OPENCL_ERROR;
	}


/*
	Check whether the device is nvidia
	char *vendor = malloc(100*sizeof(char));
	clGetDeviceInfo(device, CL_DEVICE_VENDOR, 100, vendor, 0);
	int nvidia = strstr(vendor, "NVIDIA") != 0;
	free(vendor);
*/
	b4pfm3D_params params, last_best;
	default_opt_3d_solver(context, &params, k1, k2, k3, ldf, prec, B4PFM_DEBUG_NONE, 0);

	b4pfm3D_params best_params = params;
	double best_time = 1.0/0.0;

#define handle_result() { \
	double time = run_speed_test_3d(context, queue, tmp1, tmp2, tmp3, params, force_radix2, k1, k2, k3, ldf, prec, debug); \
	if(time < 0) { \
		clReleaseCommandQueue(queue); \
		queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err); \
		if(err != CL_SUCCESS) { \
			printf("(error) b4pdf3D / auto opt.: Can't create new command queue. OpenCL errorcode: %d \n", err); \
			clReleaseMemObject(tmp3); \
			clReleaseMemObject(tmp2); \
			clReleaseMemObject(tmp1); \
			clReleaseContext(context); \
			return B4PFM_OPENCL_ERROR; \
		} \
		printf("(warning) b4pdf3D / auto opt.: New command queue created. \n"); \
	} else if(time < best_time) { \
		best_time = time; \
		best_params = params; \
	} }

#define ppp_mul(p,mul,vec_p) \
	params = best_params; \
	params.vec_p = vw; \
	params.p /= mul;  \
	handle_result(); \
	params = best_params; \
	params.vec_p = vw; \
	params.p *= mul;  \
	handle_result(); 

#define ppp_add_helper(p,add,vec_p) \
	params = best_params; \
	params.vec_p = vw; \
	params.p += add;  \
	handle_result(); 

#define ppp_add(p,add,vec_p) \
	ppp_add_helper(p,-add,vec_p) \
	ppp_add_helper(p,add,vec_p) 

#define local_opt2(p1,p2,vec_p)\
	do { \
		last_best = best_params; \
		ppp_mul(p1,2,vec_p); \
		ppp_add(p2,1,vec_p); \
		ppp_mul(p2,2,vec_p); \
	} while(last_best.p1 != best_params.p1 || last_best.p2 != best_params.p2);

#define local_opt4(p1,p2,p3,p4,vec_p)\
	do { \
		last_best = best_params; \
		ppp_mul(p1,2,vec_p); \
		ppp_mul(p2,2,vec_p); \
		ppp_mul(p3,4,vec_p); \
		ppp_mul(p4,4,vec_p); \
	} while(last_best.p1 != best_params.p1 || last_best.p2 != best_params.p2 || last_best.p3 != best_params.p3 || last_best.p4 != best_params.p4);

#define local_opt6(p1,p2,p3,p4,p5,p6,vec_p)\
	do { \
		last_best = best_params; \
		ppp_mul(p1,2,vec_p); \
		ppp_mul(p2,2,vec_p); \
		ppp_mul(p3,4,vec_p); \
		ppp_mul(p4,4,vec_p); \
		ppp_mul(p5,2,vec_p); \
		ppp_add(p5,1,vec_p); \
		ppp_mul(p6,2,vec_p); \
		ppp_add(p6,1,vec_p); \
	} while(last_best.p1 != best_params.p1 || last_best.p2 != best_params.p2 || last_best.p3 != best_params.p3 || last_best.p4 != best_params.p4 || last_best.p5 != best_params.p5 || last_best.p6 != best_params.p6);

	params = best_params;
	params.params_2d.use_local_coef = 0;
	for(params.params_2d.force_priv_coef = 0; params.params_2d.force_priv_coef <= 1; params.params_2d.force_priv_coef++) {
		for(params.params_2d.hilodouble = 0; params.params_2d.hilodouble <= 1; params.params_2d.hilodouble++) {
			for(params.params_2d.cr_wg_size = params.params_2d.work_block_size; params.params_2d.cr_wg_size == params.params_2d.work_block_size || params.params_2d.cr_wg_size <= POW2(k3-1); params.params_2d.cr_wg_size *= 2) {
				for(params.params_2d.cr_local_mem_size = MIN(POW2(k3),params.params_2d.cr_wg_size); params.params_2d.cr_local_mem_size == MIN(POW2(k3),params.params_2d.cr_wg_size) || params.params_2d.cr_local_mem_size <= POW2(k3); params.params_2d.cr_local_mem_size *= 2) {
					handle_result();
				}
			}
		}
	}

	params = best_params;
	params.params_2d.use_local_coef = 1;
	params.params_2d.force_priv_coef = 0;
	for(params.params_2d.cr_wg_size = params.params_2d.work_block_size; params.params_2d.cr_wg_size == params.params_2d.work_block_size || params.params_2d.cr_wg_size <= POW2(k3-1); params.params_2d.cr_wg_size *= 2) {
		for(params.params_2d.hilodouble = 0; params.params_2d.hilodouble <= 1; params.params_2d.hilodouble++) {
			for(params.params_2d.cr_local_mem_size = MIN(POW2(k3),params.params_2d.cr_wg_size); params.params_2d.cr_local_mem_size == MIN(POW2(k3),params.params_2d.cr_wg_size) || params.params_2d.cr_local_mem_size <= POW2(k3); params.params_2d.cr_local_mem_size *= 2) {
				handle_result();
			}
		}
	}

	int vw;
	for(vw = 1; vw <= (prec ? 2 : 4); vw *= 2) {
	
		if(!force_radix2) {	

			params = best_params;
			params.params_2d.vector_width = vw;
			for(params.params_2d.other_wg_size_11 = params.params_2d.work_block_size; params.params_2d.other_wg_size_11 == params.params_2d.work_block_size || params.params_2d.other_wg_size_11 <= POW2(k3)/params.params_2d.vector_width; params.params_2d.other_wg_size_11 *= 2) {
				handle_result();
			}

			params = best_params;
			params.params_2d.vector_width = vw;
			for(params.params_2d.other_wg_size_21 = params.params_2d.work_block_size; params.params_2d.other_wg_size_21 == params.params_2d.work_block_size || params.params_2d.other_wg_size_21 <= POW2(k3)/params.params_2d.vector_width; params.params_2d.other_wg_size_21 *= 2) {
				handle_result();
			}

			do { 
				last_best = best_params; 
				ppp_mul(params_2d.other_wg_size_12,	2, params_2d.vector_width); 
				ppp_mul(params_2d.max_sum_size1,	4, params_2d.vector_width); 
			} while(last_best.params_2d.other_wg_size_12 != best_params.params_2d.other_wg_size_12 || 
					last_best.params_2d.max_sum_size1 != best_params.params_2d.max_sum_size1);
	
			local_opt4(params_2d.other_wg_size_22a, params_2d.other_wg_size_22b, params_2d.max_sum_size2a, params_2d.max_sum_size2b, params_2d.vector_width);

		}

		if(k2 & 1 || force_radix2) {	

			params = best_params;
			params.params_2d.vector_width = vw;
			for(params.params_2d.other_wg_size_q2_21 = params.params_2d.work_block_size; params.params_2d.other_wg_size_q2_21 == params.params_2d.work_block_size || params.params_2d.other_wg_size_q2_21 <= POW2(k3)/params.params_2d.vector_width; params.params_2d.other_wg_size_q2_21 *= 2) {
				handle_result();
			}
	
			do { 
				last_best = best_params; 
				ppp_mul(params_2d.other_wg_size_q2_22,	2, params_2d.vector_width); 
				ppp_mul(params_2d.max_sum_size_q2_2,	4, params_2d.vector_width); 
			} while(last_best.params_2d.other_wg_size_q2_22 != best_params.params_2d.other_wg_size_q2_22 || 
					last_best.params_2d.max_sum_size_q2_2 != best_params.params_2d.max_sum_size_q2_2);
		}

		if(force_radix2) {	

			params = best_params;
			params.params_2d.vector_width = vw;
			for(params.params_2d.other_wg_size_q2_11 = params.params_2d.work_block_size; params.params_2d.other_wg_size_q2_11 == params.params_2d.work_block_size || params.params_2d.other_wg_size_q2_11 <= POW2(k3)/params.params_2d.vector_width; params.params_2d.other_wg_size_q2_11 *= 2) {
				handle_result();
			}

			do { 
				last_best = best_params; 
				ppp_mul(params_2d.other_wg_size_q2_12,	2, params_2d.vector_width); 
				ppp_mul(params_2d.max_sum_size_q2_1,	4, params_2d.vector_width); 
			} while(last_best.params_2d.other_wg_size_q2_12 != best_params.params_2d.other_wg_size_q2_12 || 
					last_best.params_2d.max_sum_size_q2_1 != best_params.params_2d.max_sum_size_q2_1);
	
		}
	}


	for(vw = 1; vw <= (prec ? 2 : 4); vw *= 2) {

		if(!force_radix2) {	

			local_opt2(other_wg_size_11, wg_per_2d_vect_11, vector_width);
			local_opt2(other_wg_size_21, wg_per_2d_vect_21, vector_width);

			do { 
				last_best = best_params; 
				ppp_mul(other_wg_size_12, 	2, vector_width); 
				ppp_mul(max_sum_size1,		4, vector_width); 
				ppp_mul(wg_per_2d_vect_12, 	2, vector_width); 
				ppp_add(wg_per_2d_vect_12, 	1, vector_width); 
			} while(last_best.other_wg_size_12 != best_params.other_wg_size_12 || 
					last_best.max_sum_size1 != best_params.max_sum_size1 || 
					last_best.wg_per_2d_vect_12 != best_params.wg_per_2d_vect_12);
	
			local_opt6(other_wg_size_22a, other_wg_size_22b, max_sum_size2a, max_sum_size2b, wg_per_2d_vect_22a, wg_per_2d_vect_22b, vector_width);
		}

		if(force_radix2) {
			local_opt2(other_wg_size_q2_11, wg_per_2d_vect_q2_11, vector_width);

			do { 
				last_best = best_params; 
				ppp_mul(other_wg_size_q2_12, 	2, vector_width); 
				ppp_mul(max_sum_size_q2_1,		2, vector_width); 
				ppp_mul(wg_per_2d_vect_q2_12, 	2, vector_width); 
				ppp_add(wg_per_2d_vect_q2_12, 	1, vector_width); 
			} while(last_best.other_wg_size_q2_12 != best_params.other_wg_size_q2_12 || 
					last_best.max_sum_size_q2_1 != best_params.max_sum_size_q2_1 || 
					last_best.wg_per_2d_vect_q2_12 != best_params.wg_per_2d_vect_q2_12);
		}

		if(force_radix2 || k1 & 1) {
			local_opt2(other_wg_size_q2_21, wg_per_2d_vect_q2_21, vector_width);

			do { 
				last_best = best_params; 
				ppp_mul(other_wg_size_q2_22,2,vector_width); 
				ppp_mul(max_sum_size_q2_2,4,vector_width); 
				ppp_mul(wg_per_2d_vect_q2_22,2,vector_width); 
				ppp_add(wg_per_2d_vect_q2_22,1,vector_width); 
			} while(last_best.other_wg_size_q2_22 != best_params.other_wg_size_q2_22 || 
					last_best.max_sum_size_q2_2 != best_params.max_sum_size_q2_2 || 
					last_best.wg_per_2d_vect_q2_22 != best_params.wg_per_2d_vect_q2_22);
		}

	}


	if(debug != B4PFM_DEBUG_NONE) {
		printf(
			"(debug) b4pdf3D / run solver: Best time: %f, params = \n", best_time);
		print_opt_params_3d(&best_params);
	}

	
	*opt_params = best_params;
 
	clReleaseCommandQueue(queue);
	clReleaseMemObject(tmp3);
	clReleaseMemObject(tmp2);
	clReleaseMemObject(tmp1);
	clReleaseContext(context);

	return B4PFM_OK;
}



b4pfm3D b4pfm3D_init_solver(cl_context o_context, b4pfm3D_params *o_opt_params, int k1, int k2, int k3, int ldf, int prec, int force_radix2, int debug, int *error) {
	cl_int err;
	cl_context context;

	int n2 = POW2(k2)-1;

	/* Create OpenCL context  */
	if(o_context == 0) {
		if(debug != B4PFM_DEBUG_NONE) printf(
			"(debug) b4pdf3D / pre-init solver: No OpenCL context given. " \
			"Creating an OpenCL context... \n");

		cl_platform_id platform;
		cl_uint num;
		err = clGetPlatformIDs(1, &platform, &num);
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf3D / pre-init solver: Can't query OpenCL platform. " \
				"OpenCL errorcode: %d\n", err);
			*error = B4PFM_OPENCL_ERROR;
			return INV_SOLVER;
		}

		if(num < 1) {
			printf(
				"(error) b4pdf3D / pre-init solver: No OpenCL platforms. \n");
			*error = B4PFM_OPENCL_ERROR;
			return INV_SOLVER;
		}
                       
		cl_device_id device;
        
		err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, 0);
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf3D / pre-init solver: Can't query OpenCL devices. " \
				"OpenCL errorcode: %d\n", err);
			*error = B4PFM_OPENCL_ERROR;
			return INV_SOLVER;
		}

		context = clCreateContext(0, 1, &device, 0, 0, &err);
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf3D / pre-init solver: Can't create OpenCL context. " \
				"OpenCL errorcode: %d\n", err);
			*error = B4PFM_OPENCL_ERROR;
			return INV_SOLVER;
		}
	} else {
		context = o_context;
	}

	b4pfm3D_params n_opt_params;
	b4pfm3D_params *opt_params;
	int err_info;

	if(o_opt_params == NULL) {
		if(debug != B4PFM_DEBUG_NONE) printf(
			"(debug) b4pdf3D / pre-init solver: No opt_params given. Creating default opt_params...\n");		

		err = default_opt_3d_solver(context, &n_opt_params, k1, k2, k3, ldf, prec, debug, &err_info);

		if(err != B4PFM_OK) {
			clReleaseContext(context);
			*error = err;
			return INV_SOLVER;
		}

		opt_params = &n_opt_params;
	} else {
		opt_params = o_opt_params;
	}

	b4pfm3D solver_handle = INV_SOLVER;
	r_b4pfm3D *solver = NULL;

	/* Initialize data-structures */
	if(!initialized_3d) {
		int i;
		for(i = 0; i < MAX_SOLVERS; i++)
			solvers_3d[i] = NULL;	
		initialized_3d = 1;
	}

	int i;
	for(i = 0; i < MAX_SOLVERS; i++) {
		if(solvers_3d[i] == NULL) {
			solver_handle = i+1;
			break;
		}
	}

	if(solver_handle == INV_SOLVER) {
		printf(
			"(error) b4pdf3D / pre-init solver: Too many solvers.\n");
		*error = B4PFM_TOO_MANY_SOLVERS;
		clReleaseContext(context);
		return INV_SOLVER;
	}

	solver = init_3d_solver(context, k1, k2, k3, ldf, prec, *opt_params, force_radix2, debug, &err, &err_info);

#define handle_invalid_wg_size(param, err_code) \
	if(err == err_code && o_opt_params == NULL) { \
		opt_params->param = n2*POW2(k3)/opt_params->vector_width; \
		while(err_info < opt_params->param) \
			opt_params->param /= 2*opt_params->vector_width; \
		if(debug != B4PFM_DEBUG_NONE) printf( \
			"(debug) b4pdf2D / pre-init solver: Default param is too big. " \
			"New value = %d\n", opt_params->param);		 \
		solver = init_3d_solver(context, k1, k2, k3, ldf, prec, *opt_params, force_radix2, debug, &err, &err_info); \
	}

	handle_invalid_wg_size(other_wg_size_11, 	B4PFM_OTHER_WG_3D11_TOO_BIG);
	handle_invalid_wg_size(other_wg_size_12, 	B4PFM_OTHER_WG_3D12_TOO_BIG);
	handle_invalid_wg_size(other_wg_size_21, 	B4PFM_OTHER_WG_3D21_TOO_BIG);
	handle_invalid_wg_size(other_wg_size_22a, 	B4PFM_OTHER_WG_3D22A_TOO_BIG);
	handle_invalid_wg_size(other_wg_size_22b, 	B4PFM_OTHER_WG_3D22B_TOO_BIG);

	handle_invalid_wg_size(other_wg_size_q2_11, 	B4PFM_OTHER_WG_Q2_3D11_TOO_BIG);
	handle_invalid_wg_size(other_wg_size_q2_12, 	B4PFM_OTHER_WG_Q2_3D12_TOO_BIG);
	handle_invalid_wg_size(other_wg_size_q2_21, 	B4PFM_OTHER_WG_Q2_3D21_TOO_BIG);
	handle_invalid_wg_size(other_wg_size_q2_22, 	B4PFM_OTHER_WG_Q2_3D22_TOO_BIG);

	if(err == B4PFM_OK) {
		*error = B4PFM_OK;
		solvers_3d[solver_handle-1] = solver;
			return solver_handle;
	} else {
		*error = err;
		clReleaseContext(context);
		return INV_SOLVER; 
	}
}

int b4pfm3D_run_solver(b4pfm3D solver, cl_command_queue queue, cl_mem f, cl_mem tmp1, cl_mem tmp2, int debug) {
	int err_info;
	if(initialized_3d && solver >= 1 && solver <= MAX_SOLVERS && solvers_3d[solver-1] != NULL)
		return run_3d_solver(solvers_3d[solver-1], queue, f, tmp1, tmp2,  debug, &err_info);
	else
		return B4PFM_INVALID_SOLVER;
}

int b4pfm3D_load_and_run_solver_float(b4pfm3D solver, float *f, int debug) {
	int err_info;
	if(initialized_3d && solver >= 1 && solver <= MAX_SOLVERS && solvers_3d[solver-1] != NULL)
		return load_and_run_3d_solver(solvers_3d[solver-1], (void*) f, debug, &err_info);
	else
		return B4PFM_INVALID_SOLVER;
}

int b4pfm3D_load_and_run_solver_double(b4pfm3D solver, double *f, int debug) {
	int err_info;
	if(initialized_3d && solver >= 1 && solver <= MAX_SOLVERS && solvers_3d[solver-1] != NULL)
		return load_and_run_3d_solver(solvers_3d[solver-1], (void*) f, debug, &err_info);
	else
		return B4PFM_INVALID_SOLVER;
}

int b4pfm3D_free_solver(b4pfm3D solver) {
	if(initialized_3d && solver >= 1 && solver <= MAX_SOLVERS && solvers_3d[solver-1] != NULL) {
		int err = free_3d_solver(solvers_3d[solver-1]);
		if(err == B4PFM_OK)
			solvers_3d[solver-1] = NULL;
		return err;
	} else {
		return B4PFM_INVALID_SOLVER;
	}
}

int b4pfm3D_default_opts(cl_context context, b4pfm3D_params *opt_params, int k1, int k2, int k3, int ldf, int prec, int debug) {
	int err_info;
	return default_opt_3d_solver(context, opt_params, k1, k2, k3, ldf, prec, debug, &err_info);
}

int b4pfm3D_auto_opts(cl_context context, cl_mem tmp1, cl_mem tmp2, cl_mem tmp3, b4pfm3D_params *opt_params, int force_radix2, int k1, int k2, int k3, int ldf, int prec, int debug) {
	int err_info;
	return auto_optimize_3d_solver(context, tmp1, tmp2, tmp3, opt_params, force_radix2, k1, k2, k3, ldf, prec, debug, &err_info);
}

