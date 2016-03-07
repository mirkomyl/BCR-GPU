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

#include "kernels2D.dat"

/* Remember to update this when you change kernel args!!! */
#define CR_LOCAL_MEM_BIAS (3*4)

#define KARGS_2D11_O_F 	0
#define KARGS_2D11_O_G 	1
#define KARGS_2D11_R 	2

#define KARGS_2D12_O_F 	0
#define KARGS_2D12_O_G 	1
#define KARGS_2D12_R 	2
#define KARGS_2D12_PREV_COUNT 3

#define KARGS_2D21_O_F 	0
#define KARGS_2D21_O_G 	1
#define KARGS_2D21_R 	2

#define KARGS_2D22A_O_G 	0
#define KARGS_2D22A_R 	1

#define KARGS_2D22B_O_F 	0
#define KARGS_2D22B_O_G 	1
#define KARGS_2D22B_R 	2
#define KARGS_2D22B_PREV_COUNT 3

#define KARGS_2DCR_O_F 	0
#define KARGS_2DCR_R1	1
#define KARGS_2DCR_R2	2
#define KARGS_2DCR_RADIX2_2D 3
#define KARGS_2DCR_RADIX2_3D 4

#define KARGS_Q2_2D11_O_F 	0
#define KARGS_Q2_2D11_O_G 	1
#define KARGS_Q2_2D11_R 	2

#define KARGS_Q2_2D12_O_F 	0
#define KARGS_Q2_2D12_O_G 	1
#define KARGS_Q2_2D12_R 	2
#define KARGS_Q2_2D12_PREV_COUNT 3

#define KARGS_Q2_2D21_O_F 	0
#define KARGS_Q2_2D21_O_G 	1
#define KARGS_Q2_2D21_R 	2

#define KARGS_Q2_2D22_O_F 	0
#define KARGS_Q2_2D22_O_G 	1
#define KARGS_Q2_2D22_R 	2
#define KARGS_Q2_2D22_PREV_COUNT 3

typedef struct {
	b4pfm2D_params params;
	double time;
} opt_struct_2d;

int initialized_2d = 0;

r_b4pfm2D *solvers_2d[MAX_SOLVERS];

int free_2d_solver(r_b4pfm2D *solver) {

	if(solver == NULL)
		return B4PFM_INVALID_SOLVER;

#define free_kernel(name) \
	if(solver->name) clReleaseKernel(solver->name); solver->name = 0

	free_kernel(kernel_Q2_2D22);
	free_kernel(kernel_Q2_2D21);
	free_kernel(kernel_Q2_2D12);
	free_kernel(kernel_Q2_2D11);

	free_kernel(kernel_2DCR);
	free_kernel(kernel_2D22B);
	free_kernel(kernel_2D22A);
	free_kernel(kernel_2D21);
	free_kernel(kernel_2D12);
	free_kernel(kernel_2D11);

#undef free_kernel

	if(solver->context)
		clReleaseContext(solver->context);

	free(solver);

	return B4PFM_OK;
}

void print_opt_params_2d(b4pfm2D_params *opt_params) {
	printf(
		"{\n" \
		"     hilodouble           = %d\n" \
		"     use_local_coef       = %d\n" \
		"     force_priv_coef      = %d\n" \
		"     work_block_size      = %d\n" \
		"     cr_wg_size           = %d\n" \
		"     cr_local_mem_size    = %d\n" \
		"     other_wg_size_11     = %d\n" \
		"     other_wg_size_12     = %d\n" \
		"     other_wg_size_21     = %d\n" \
		"     other_wg_size_22a    = %d\n" \
		"     other_wg_size_22b    = %d\n" \
		"     vector_width         = %d\n" \
		"     max_sum_size1        = %d\n" \
		"     max_sum_size2a       = %d\n" \
		"     max_sum_size2b       = %d\n" \
		"     other_wg_size_q2_11  = %d\n" \
		"     other_wg_size_q2_12  = %d\n" \
		"     other_wg_size_q2_21  = %d\n" \
		"     other_wg_size_q2_22  = %d\n" \
		"     max_sum_size_q2_1    = %d\n" \
		"     max_sum_size_q2_2    = %d\n" \
		"}\n",
		opt_params->hilodouble,
		opt_params->use_local_coef,
		opt_params->force_priv_coef,
		opt_params->work_block_size,
		opt_params->cr_wg_size,
		opt_params->cr_local_mem_size,
		opt_params->other_wg_size_11,
		opt_params->other_wg_size_12,		
		opt_params->other_wg_size_21,
		opt_params->other_wg_size_22a,
		opt_params->other_wg_size_22b,	
		opt_params->vector_width,
		opt_params->max_sum_size1,
		opt_params->max_sum_size2a,
		opt_params->max_sum_size2b,
		opt_params->other_wg_size_q2_11,
		opt_params->other_wg_size_q2_12,
		opt_params->other_wg_size_q2_21,
		opt_params->other_wg_size_q2_22,
		opt_params->max_sum_size_q2_1,
		opt_params->max_sum_size_q2_2);
}

r_b4pfm2D* init_2d_solver(cl_context context, int k1, int k2, int k3, int ldf, int prec, b4pfm2D_params opt_params, int force_radix2, int debug, int *error, int *o_err_info) {

	int *err_info;
	int tmp_err_info;
	if(o_err_info == 0)
		err_info = &tmp_err_info;
	else
		err_info = o_err_info;

	if(debug == B4PFM_DEBUG_FULL || debug == B4PFM_DEBUG_TIMING) { 
		printf("(debug) b4pdf2D / init solver: Initializing B4PDF2D-solver. Args: k1=%d, k2=%d, k3=%d, ldf=%d, prec=%d, force_radix2=%d, opt_params = \n",
			k1, k2, k3, ldf, prec, force_radix2);
		print_opt_params_2d(&opt_params);
	} else if(debug != B4PFM_DEBUG_NONE) printf(
		"(debug) b4pdf2D / init solver: Initializing B4PDF2D-solver. \n");
	*error = B4PFM_UNKNOWN_ERROR;
	int err; 

	int n3 = POW2(k3)-1;

	/* Initial args checks */

	if(    1 > k2 || k2 > MAX_K2
		|| 1 > k3 || k3 > MAX_K3
		|| ldf < n3) {
		printf(
			"(error) b4pdf2D / init solver: Invalid arguments. \n");
		*error = B4PFM_INVALID_ARGS;
		return NULL;
	}

	int global_solver = POW2(k3) > opt_params.cr_local_mem_size;
	int middle_solver = opt_params.cr_local_mem_size > 2*opt_params.cr_wg_size;

	/* Initial opt_params checks */


	if(	(opt_params.use_local_coef && opt_params.force_priv_coef) ||
		!(opt_params.work_block_size == 32 || opt_params.work_block_size == 64) ||
		(prec && !(opt_params.vector_width == 1 || opt_params.vector_width == 2)) ||
		(!prec && !(opt_params.vector_width == 1 || opt_params.vector_width == 2 || opt_params.vector_width == 4)) ||
		opt_params.cr_wg_size % opt_params.work_block_size != 0 ||
		(global_solver && !middle_solver && opt_params.cr_local_mem_size < 2*opt_params.cr_wg_size) ||
		(!global_solver && !middle_solver && opt_params.cr_local_mem_size != POW2(k3))) {

		printf(
			"(error) b4pdf2D / init solver: Invalid opt_params. \n");
		*error = B4PFM_INVALID_OPTS;
		return NULL;
	}

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
			opt_params.max_sum_size1 <  4 ||  opt_params.max_sum_size1 % 4 != 0 ||
			opt_params.max_sum_size2a < 4 || opt_params.max_sum_size2a % 4 != 0 ||
			opt_params.max_sum_size2b < 4 || opt_params.max_sum_size2b % 4 != 0 ) {

			printf(
				"(error) b4pdf2D / init solver: Invalid opt_params. \n");
			*error = B4PFM_INVALID_OPTS;
			return NULL;
		}
	}

	if(force_radix2) {
		if( opt_params.other_wg_size_q2_11 % opt_params.work_block_size != 0 ||
			opt_params.other_wg_size_q2_12 % opt_params.work_block_size != 0 ||
			opt_params.other_wg_size_q2_11 <= 0 ||
			opt_params.other_wg_size_q2_12 <= 0 ||
			opt_params.max_sum_size_q2_1 < 2 ||  opt_params.max_sum_size_q2_1 % 2 != 0) {

			printf(
				"(error) b4pdf2D / init solver: Invalid opt_params. \n");
			*error = B4PFM_INVALID_OPTS;
			return NULL;
		}
	}

	if(k2 & 1 || force_radix2) {
		if( opt_params.other_wg_size_q2_21 % opt_params.work_block_size != 0 ||
			opt_params.other_wg_size_q2_22 % opt_params.work_block_size != 0 ||
			opt_params.other_wg_size_q2_21 <= 0 ||
			opt_params.other_wg_size_q2_22 <= 0 ||
			opt_params.max_sum_size_q2_2 < 2 || opt_params.max_sum_size_q2_2 % 2 != 0 ) {

			printf(
				"(error) b4pdf2D / init solver: Invalid opt_params. \n");
			*error = B4PFM_INVALID_OPTS;
			return NULL;
		}
	}

	if(!force_radix2) {

		if(	ldf < opt_params.other_wg_size_11 ||
			ldf < opt_params.other_wg_size_12 ||
			ldf < opt_params.other_wg_size_21 ||
			ldf < opt_params.other_wg_size_22a ||
			ldf < opt_params.other_wg_size_22b)
			printf(
				"(warning) b4pdf2D / init solver: ldf is smaller than given workgroup sizes. \n");
	}

	if(force_radix2) {
		if(	ldf < opt_params.other_wg_size_q2_11 ||
			ldf < opt_params.other_wg_size_q2_12)
			printf(
				"(warning) b4pdf2D / init solver: ldf is smaller than given workgroup sizes. \n");
	}

	if(k2 & 1 || force_radix2) {
		if(	ldf < opt_params.other_wg_size_q2_21 ||
			ldf < opt_params.other_wg_size_q2_22)
			printf(
				"(warning) b4pdf2D / init solver: ldf is smaller than given workgroup sizes. \n");
	}

	size_t var_size = prec ? sizeof(cl_double) : sizeof(cl_float);

	/* Malloc new solver if possible */
	r_b4pfm2D *solver = malloc(sizeof(r_b4pfm2D));

	if(solver == NULL) {
		printf(
			"(error) b4pdf2D / init solver: Can't malloc solver.\n");
		*error = B4PFM_UNKNOWN_ERROR;
		return NULL;
	}

	/* Pre-initialize solver */
	solver->context = 0;
	solver->force_radix2 = force_radix2;
	solver->k1 = k1;
	solver->k2 = k2;
	solver->k3 = k3;
	solver->ldf = ldf;

	solver->double_prec = prec;
        
	solver->kernel_2D11 = 0;
	solver->kernel_2D12 = 0;
	solver->kernel_2D21 = 0;	
	solver->kernel_2D22A = 0;
	solver->kernel_2D22B = 0;
	solver->kernel_2DCR = 0;

	solver->kernel_Q2_2D11 = 0;
	solver->kernel_Q2_2D12 = 0;
	solver->kernel_Q2_2D21 = 0;	
	solver->kernel_Q2_2D22 = 0;

	solver->max_sum_size1 = opt_params.max_sum_size1;
	solver->max_sum_size2a = opt_params.max_sum_size2a;
	solver->max_sum_size2b = opt_params.max_sum_size2b;

	solver->max_sum_size_q2_1 = opt_params.max_sum_size_q2_1;
	solver->max_sum_size_q2_2 = opt_params.max_sum_size_q2_2;

	/* Sets opencl context for solver */
	err = clRetainContext(context);

	if(err != CL_SUCCESS) {
		printf(
			"(error) b4pdf2D / init solver: Invalid context. " \
			"OpenCL errorcode: %d\n", err);
		*error = B4PFM_INVALID_ARGS;
		return NULL;
	}
	
	solver->context = context;

	/* TODO: Move this! */
	cl_program program = 0;

#define free_all() 												\
	if(program)					clReleaseProgram(program);		\
	free_2d_solver(solver)

	/* Find the OpenCL device corresponding to given context */
	cl_device_id device;

	err = clGetContextInfo(solver->context, CL_CONTEXT_DEVICES, 
		sizeof(cl_device_id), &device, 0);

	if(err != CL_SUCCESS) {
		printf(
			"(error) b4pdf2D / init solver: Can't query device id from context. " \
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

	int cr_local_mem_usage = (opt_params.cr_local_mem_size + (opt_params.use_local_coef ? 2*k3 : 0))*var_size + CR_LOCAL_MEM_BIAS;

	int max_cr_work_group_size = MIN(max_work_item_sizes[0], ((max_local_mem_size-((opt_params.use_local_coef ? 2*k3 : 0 )*var_size)-CR_LOCAL_MEM_BIAS))/var_size);

		if(	opt_params.cr_wg_size > max_cr_work_group_size ||
			cr_local_mem_usage > max_local_mem_size) {

			printf(
				"(error) b4pdf2D / init solver: Invalid opt_params. \n");
			free_all();
			*error = B4PFM_INVALID_OPTS;
			return NULL;
		}

	if(!force_radix2) {

		/* Check */
		if(	opt_params.other_wg_size_11 > max_work_item_sizes[0] ||
			opt_params.other_wg_size_12 > max_work_item_sizes[0] ||
			opt_params.other_wg_size_21 > max_work_item_sizes[0] ||
			opt_params.other_wg_size_22a > max_work_item_sizes[0] ||
			opt_params.other_wg_size_22b > max_work_item_sizes[0]) {

			printf(
				"(error) b4pdf2D / init solver: Invalid opt_params. \n");
			free_all();
			*error = B4PFM_INVALID_OPTS;
			return NULL;
		}
	}

	if(force_radix2) {
		if(	opt_params.other_wg_size_q2_11 > max_work_item_sizes[0] ||
			opt_params.other_wg_size_q2_12 > max_work_item_sizes[0]) {

			printf(
				"(error) b4pdf2D / init solver: Invalid opt_params. \n");
			free_all();
			*error = B4PFM_INVALID_OPTS;
			return NULL;
		}
	}

	if(k2 & 1 || force_radix2) {
		if(	opt_params.other_wg_size_q2_21 > max_work_item_sizes[0] ||
			opt_params.other_wg_size_q2_22 > max_work_item_sizes[0]) {

			printf(
				"(error) b4pdf2D / init solver: Invalid opt_params. \n");
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
			"(error) b4pdf2D / init solver: Device doesn't support double precision. \n");
		free_all();
		*error = B4PFM_INVALID_ARGS;
		return NULL;
	}


	/* Creates a new opencl program-object */
	program = clCreateProgramWithSource(solver->context, 1, &kernels2D, 0, &err);
	if(err != CL_SUCCESS) {
		printf(
			"(error) b4pdf2D / init solver: Can't create program. " \
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
		" -D HILODOUBLE=%d"		\
		" -D GLOBAL_SOLVER=%d"	\
		" -D MIDDLE_SOLVER=%d"	\
		" -D USE_LOCAL_COEF=%d"	\
		" -D FORCE_PRIV_COEF=%d"\
		" -D CR_LOCAL_MEM_SIZE=%d" 	\
		" -D CR_WG_SIZE=%d"		\
		" -D OTHER_WG_SIZE_2D11=%d"	\
		" -D OTHER_WG_SIZE_2D12=%d"	\
		" -D OTHER_WG_SIZE_2D21=%d"	\
		" -D OTHER_WG_SIZE_2D22A=%d"	\
		" -D OTHER_WG_SIZE_2D22B=%d"	\
		" -D OTHER_WG_SIZE_Q2_2D11=%d"	\
		" -D OTHER_WG_SIZE_Q2_2D12=%d"	\
		" -D OTHER_WG_SIZE_Q2_2D21=%d"	\
		" -D OTHER_WG_SIZE_Q2_2D22=%d"	\
		" -D MAX_SUM_SIZE1=%d"	\
		" -D MAX_SUM_SIZE2A=%d"	\
		" -D MAX_SUM_SIZE2B=%d"	\
		" -D MAX_SUM_SIZE_Q2_1=%d"	\
		" -D MAX_SUM_SIZE_Q2_2=%d"	\
		" -D K1=%d"				\
		" -D K2=%d"				\
		" -D K3=%d"				\
		" -D LDF=%d";

	char opt[strlen(format)+100];

	sprintf(opt, format, 
		k2 & 1 || force_radix2,
		!force_radix2,
		solver->double_prec,    
		ati_double_support,
		opt_params.vector_width,
		opt_params.hilodouble, 
		global_solver,
		middle_solver,
		opt_params.use_local_coef,
		opt_params.force_priv_coef,
		opt_params.cr_local_mem_size,
		opt_params.cr_wg_size,
		opt_params.other_wg_size_11,
		opt_params.other_wg_size_12,
		opt_params.other_wg_size_21,
		opt_params.other_wg_size_22a,
		opt_params.other_wg_size_22b,
		opt_params.other_wg_size_q2_11,
		opt_params.other_wg_size_q2_12,
		opt_params.other_wg_size_q2_21,
		opt_params.other_wg_size_q2_22,
		opt_params.max_sum_size1,
		opt_params.max_sum_size2a,
		opt_params.max_sum_size2b,
		opt_params.max_sum_size_q2_1,
		opt_params.max_sum_size_q2_2,
		k1,
		k2,
		k3,
		ldf);

	if(debug == B4PFM_DEBUG_FULL || debug == B4PFM_DEBUG_TIMING) 
		printf("(debug) b4pdf2D / init solver: Compiler args: %s\n", opt);

	err = clBuildProgram(program, 0, 0, opt, 0, 0);

	if (err || debug == B4PFM_DEBUG_FULL || debug == B4PFM_DEBUG_TIMING) {
		char *log = malloc(102400);
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 
			102400, log, NULL );
		printf("(debug) b4pdf2D / init solver: OpenCL compiler output:\n");
		printf("======================================================\n");
		printf("%s\n", log);
		printf("======================================================\n");
		free(log);
	}       

	if(err != CL_SUCCESS) {
		printf(
			"(error) b4pdf2D / init solver: Can't build program. " \
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

	if(!force_radix2) {

		/* Creates opencl kernel handels */
		solver->kernel_2D11 = clCreateKernel(program, "b4pfm_11", &err);
		if(err != CL_SUCCESS) {	
			printf(
				"(error) b4pdf2D / init solver: Can't create kernel b4pfm_11. " \
				"OpenCL errorcode: %d\n.", err);
			free_all();
			*error = B4PFM_OPENCL_ERROR;
			return NULL;
		}

		solver->kernel_2D12 = clCreateKernel(program, "b4pfm_12", &err);
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf2D / init solver: Can't create kernel b4pfm_12. " \
				"OpenCL errorcode: %d\n.", err);
			free_all();
			*error = B4PFM_OPENCL_ERROR;
			return NULL; 
		}

		solver->kernel_2D21 = clCreateKernel(program, "b4pfm_21", &err);
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf2D / init solver: Can't create kernel b4pfm_21. " \
				"OpenCL errorcode: %d\n.", err);
			free_all();
			*error = B4PFM_OPENCL_ERROR;
			return NULL;
		}

		solver->kernel_2D22A = clCreateKernel(program, "b4pfm_22a", &err);	
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf2D / init solver: Can't create kernel b4pfm_22a. " \
				"OpenCL errorcode: %d\n.", err);
			free_all();
			*error = B4PFM_OPENCL_ERROR;
			return NULL; 
		}

		solver->kernel_2D22B = clCreateKernel(program, "b4pfm_22b", &err);
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf2D / init solver: Can't create kernel b4pfm_22b. " \
				"OpenCL errorcode: %d\n.", err);
			free_all();
			*error = B4PFM_OPENCL_ERROR;
			return NULL; 
		}
	}

	solver->kernel_2DCR = clCreateKernel(program, "b4pfm_CR", &err);
	if(err != CL_SUCCESS) {
		printf(
			"(error) b4pdf2D / init solver: Can't create kernel b4pfm_CR. " \
			"OpenCL errorcode: %d\n.", err);
		free_all();
		*error = B4PFM_OPENCL_ERROR;
		return NULL;
	}

	if(force_radix2) {
		solver->kernel_Q2_2D11 = clCreateKernel(program, "b2pfm_11", &err);
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf2D / init solver: Can't create kernel b2pfm_CR11. " \
				"OpenCL errorcode: %d\n.", err);
			free_all();
			*error = B4PFM_OPENCL_ERROR;
			return NULL;
		}

		solver->kernel_Q2_2D12 = clCreateKernel(program, "b2pfm_12", &err);
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf2D / init solver: Can't create kernel b2pfm_12. " \
				"OpenCL errorcode: %d\n.", err);
			free_all();
			*error = B4PFM_OPENCL_ERROR;
			return NULL; 
		}
	}

	if(k2 & 1 || force_radix2) {
	
		solver->kernel_Q2_2D21 = clCreateKernel(program, "b2pfm_21", &err);
		if(err != CL_SUCCESS) {
			printf(	
				"(error) b4pdf2D / init solver: Can't create kernel b2pfm_21. " \
				"OpenCL errorcode: %d\n.", err);
			free_all();
			*error = B4PFM_OPENCL_ERROR;
			return NULL;
		}

		solver->kernel_Q2_2D22 = clCreateKernel(program, "b2pfm_22", &err);
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf2D / init solver: Can't create kernel b2pfm_22. " \
				"OpenCL errorcode: %d\n.", err);
			free_all();
			*error = B4PFM_OPENCL_ERROR;
			return NULL; 
		}
	}

	/* Check work group sizes */

	/* Finds out what is the optimal work group size for each kernel */


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

	size_t tc_cr;
	clGetKernelWorkGroupInfo(solver->kernel_2DCR, device, 
			CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &tc_cr, 0);

	handle_wrong_wg_size(kernel_2DCR_threads, cr_wg_size, tc_cr, B4PFM_CR_WG_TOO_BIG);

	if(!force_radix2) {

		size_t tc11, tc12, tc21, tc22a, tc22b;
		clGetKernelWorkGroupInfo(solver->kernel_2D11, device, 
			CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &tc11, 0);
		clGetKernelWorkGroupInfo(solver->kernel_2D12, device, 
			CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &tc12, 0);
		clGetKernelWorkGroupInfo(solver->kernel_2D21, device, 
			CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &tc21, 0);
		clGetKernelWorkGroupInfo(solver->kernel_2D22A, device, 
			CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &tc22a, 0);
		clGetKernelWorkGroupInfo(solver->kernel_2D22B, device, 
			CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &tc22b, 0);

		handle_wrong_wg_size(kernel_2D11_threads,  other_wg_size_11, 	tc11,	B4PFM_OTHER_WG_2D11_TOO_BIG);
		handle_wrong_wg_size(kernel_2D12_threads,  other_wg_size_12, 	tc12, 	B4PFM_OTHER_WG_2D12_TOO_BIG);
		handle_wrong_wg_size(kernel_2D21_threads,  other_wg_size_21, 	tc21, 	B4PFM_OTHER_WG_2D21_TOO_BIG);
		handle_wrong_wg_size(kernel_2D22A_threads, other_wg_size_22a, 	tc22a, 	B4PFM_OTHER_WG_2D22A_TOO_BIG);
		handle_wrong_wg_size(kernel_2D22B_threads, other_wg_size_22b, 	tc22b, 	B4PFM_OTHER_WG_2D22B_TOO_BIG);
	}


	if(force_radix2) {
		size_t q2_tc11, q2_tc12;
		clGetKernelWorkGroupInfo(solver->kernel_Q2_2D11, device, 
			CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &q2_tc11, 0);
		clGetKernelWorkGroupInfo(solver->kernel_Q2_2D12, device, 
			CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &q2_tc12, 0);

		handle_wrong_wg_size(kernel_Q2_2D11_threads,  other_wg_size_q2_11, 	q2_tc11,	B4PFM_OTHER_WG_Q2_2D11_TOO_BIG);
		handle_wrong_wg_size(kernel_Q2_2D12_threads,  other_wg_size_q2_12, 	q2_tc12, 	B4PFM_OTHER_WG_Q2_2D12_TOO_BIG);

	}

	if(k2 & 1 || force_radix2) {
		size_t q2_tc21, q2_tc22;
		clGetKernelWorkGroupInfo(solver->kernel_Q2_2D21, device, 
			CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &q2_tc21, 0);
		clGetKernelWorkGroupInfo(solver->kernel_Q2_2D22, device, 
			CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &q2_tc22, 0);

		handle_wrong_wg_size(kernel_Q2_2D21_threads,  other_wg_size_q2_21, 	q2_tc21, 	B4PFM_OTHER_WG_Q2_2D21_TOO_BIG);
		handle_wrong_wg_size(kernel_Q2_2D22_threads,  other_wg_size_q2_22, 	q2_tc22, 	B4PFM_OTHER_WG_Q2_2D22_TOO_BIG);

	}

	if(debug != B4PFM_DEBUG_NONE) printf(
		"(debug) b4pdf2D / init solver: Solver is now initialized. \n");

#undef free_all

	/* Everything is ready. Return. */
	*error = B4PFM_OK;
	return solver;
	
}

int run_2d_solver(r_b4pfm2D *solver, cl_command_queue o_queue, cl_mem f, cl_mem o_tmp, int count, int r1, int debug, int *o_err_info) {

	int *err_info;
	int tmp_err_info;
	if(o_err_info == 0)
		err_info = &tmp_err_info;
	else
		err_info = o_err_info;

	if(solver == NULL)
		return B4PFM_INVALID_SOLVER;


	if(debug == B4PFM_DEBUG_FULL || debug == B4PFM_DEBUG_TIMING) printf(
		"(debug) b4pdf2D / run solver: count=%d, r1=%d\n", count, r1);
	else if(debug != B4PFM_DEBUG_NONE) printf(
		"(debug) b4pdf2D / run solver\n");

	int err;

	int var_size = solver->double_prec ? sizeof(cl_double) : sizeof(cl_float);

	int cmd_queue_prof_was_enabled = 0;
	int cmd_queue_prof_enabled = 0;

	/* Creates a new opencl command queue if necessary. */
	cl_command_queue queue;
	if(o_queue == 0) {
		if(debug != B4PFM_DEBUG_NONE) printf(
			"(debug) b4pdf2D / run solver: No OpenCL Commandqueue given. "\
			"Creating OpenCL Commandqueue... \n");
		cl_device_id device;
 		err = clGetContextInfo(solver->context, CL_CONTEXT_DEVICES, 
			sizeof(cl_device_id), &device, 0);
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf2D / run solver: Can't query device id from context. " \
				"OpenCL errorcode: %d\n", err);
			return B4PFM_OPENCL_ERROR;
		}

		if(debug == B4PFM_DEBUG_TIMING) {
			queue = clCreateCommandQueue(solver->context, device, CL_QUEUE_PROFILING_ENABLE, &err);
			printf(
				"(debug) b4pdf2D / run solver: OpenCL command queue profiling enabled. \n");
		} else
			queue = clCreateCommandQueue(solver->context, device, 0, &err);
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf2D / run solver: Can't create command queue. " \
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
					"(error) b4pdf2D / run solver: Can't enable OpenCL command queue profiling. " 
					"OpenCL errorcode: %d.\n", err);

			if(cmd_queue_prof_enabled && !cmd_queue_prof_was_enabled)
				printf(
					"(debug) b4pdf2D / run solver: OpenCL command queue profiling enabled. \n");
		}

	}

	/* Creates a new opencl memory buffer for temporar data */
	cl_mem tmp;
	if(o_tmp == 0) {
		if(debug != B4PFM_DEBUG_NONE) printf(
			"(debug) b4pdf2D / run solver: No global workplace given. " \
			"Creating global workplace buffer... \n");
		int tmp_mem_size = count*3*POW2(solver->k2-2)*POW2(solver->k3)*var_size;
		if(debug == B4PFM_DEBUG_FULL || debug == B4PFM_DEBUG_TIMING) printf(
			"(debug) b4pdf2D / run solver: Global workplace size = %d bytes. \n",
			tmp_mem_size);
		tmp = clCreateBuffer(solver->context, CL_MEM_READ_WRITE, tmp_mem_size, 0, &err);
        
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf2D / run solver: Can't allocate device memory for " \
				"workplace-buffer. OpenCL errorcode: %d\n.", err);
			clReleaseCommandQueue(queue);   
			return B4PFM_OPENCL_ERROR; 
		}
	} else {
		clRetainMemObject(o_tmp);
		tmp = o_tmp;
	}

	/* Set initial kernel arguments */
	err  = clSetKernelArg(solver->kernel_2D11, KARGS_2D11_O_F, 
		sizeof(cl_mem), (void *)&f);
	err |= clSetKernelArg(solver->kernel_2D12, KARGS_2D12_O_F, 
		sizeof(cl_mem), (void *)&f);
	err |= clSetKernelArg(solver->kernel_2D21, KARGS_2D21_O_F, 
		sizeof(cl_mem), (void *)&f);
	err |= clSetKernelArg(solver->kernel_2D22B, KARGS_2D22B_O_F, 
		sizeof(cl_mem), (void *)&f);

	err  = clSetKernelArg(solver->kernel_2D11, KARGS_2D11_O_G, 
		sizeof(cl_mem), (void *)&tmp);
	err |= clSetKernelArg(solver->kernel_2D12, KARGS_2D12_O_G, 
		sizeof(cl_mem), (void *)&tmp);
	err |= clSetKernelArg(solver->kernel_2D12, KARGS_2D12_O_G, 
		sizeof(cl_mem), (void *)&tmp);
	err |= clSetKernelArg(solver->kernel_2D21, KARGS_2D21_O_G, 
		sizeof(cl_mem), (void *)&tmp);
	err |= clSetKernelArg(solver->kernel_2D22A, KARGS_2D22A_O_G, 
		sizeof(cl_mem), (void *)&tmp);
	err |= clSetKernelArg(solver->kernel_2D22B, KARGS_2D22B_O_G, 
		sizeof(cl_mem), (void *)&tmp);

	err |= clSetKernelArg(solver->kernel_2DCR, KARGS_2DCR_O_F, 
		sizeof(cl_mem), (void *)&tmp);
	err |= clSetKernelArg(solver->kernel_2DCR, KARGS_2DCR_R1, 
		sizeof(cl_int), (void *)&r1);

	int k1 = solver->k1;
	int k2 = solver->k2;

	cl_int zero = 0;
	cl_int one = 1;
	err |= clSetKernelArg(solver->kernel_2DCR, KARGS_2DCR_RADIX2_2D, 
		sizeof(cl_int), (void *)&zero);
	if(k1 & 1 && r1 == k1) 
		err |= clSetKernelArg(solver->kernel_2DCR, KARGS_2DCR_RADIX2_3D, 
			sizeof(cl_int), (void *)&one);
	else
		err |= clSetKernelArg(solver->kernel_2DCR, KARGS_2DCR_RADIX2_3D, 
			sizeof(cl_int), (void *)&zero);

	if(k2 & 1) {
		err |= clSetKernelArg(solver->kernel_Q2_2D21, KARGS_Q2_2D21_O_F, 
			sizeof(cl_mem), (void *)&f);
		err |= clSetKernelArg(solver->kernel_Q2_2D22, KARGS_Q2_2D22_O_F, 
			sizeof(cl_mem), (void *)&f);

		err |= clSetKernelArg(solver->kernel_Q2_2D21, KARGS_Q2_2D21_O_G, 
			sizeof(cl_mem), (void *)&tmp);
		err |= clSetKernelArg(solver->kernel_Q2_2D22, KARGS_Q2_2D22_O_G, 
			sizeof(cl_mem), (void *)&tmp);
}


	if(err != CL_SUCCESS) {
		printf(
			"(error) b4pdf2D / run solver: Can't set initial kernel arguments. " \
			"OpenCL errorcode: %d\n", err);
		clReleaseMemObject(tmp);
		clReleaseCommandQueue(queue);
		return B4PFM_OPENCL_ERROR;
	}

	cl_event begin, end;

	if(cmd_queue_prof_enabled) {
		clFinish(queue);
		clEnqueueMarker(queue, &begin);
	}

	int r2;
	for(r2 = 1; r2 <= k2/2 - (k2 & 1 ? 0 : 1); r2++) {

		err  = clSetKernelArg(solver->kernel_2D11, KARGS_2D11_R, 
			sizeof(cl_int), (void *)&r2);
		err |= clSetKernelArg(solver->kernel_2DCR, KARGS_2DCR_R2,
			sizeof(cl_int), (void *)&r2);
		err |= clSetKernelArg(solver->kernel_2D12, KARGS_2D12_R,
			sizeof(cl_int), (void *)&r2);
                                
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf2D / run solver: Can't set kernel arguments: stage 1 / r = %d. " \
				"OpenCL errorcode: %d\n", r2, err);
			if(cmd_queue_prof_enabled) clReleaseEvent(begin);
			clReleaseMemObject(tmp);
			clReleaseCommandQueue(queue);
			return B4PFM_OPENCL_ERROR;
		}

		size_t global_size[2], local_size[2];

		/* Lauches kernel  */
		global_size[0] = (POW2(k2-2*r2)-1) * 3*POW2(2*r2-2) * solver->kernel_2D11_threads;
		global_size[1] = count;
		local_size[0] = solver->kernel_2D11_threads;
		local_size[1] = 1;
		err = clEnqueueNDRangeKernel(queue, solver->kernel_2D11, 2, 0, 
			global_size, local_size, 0, 0, 0);
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf2D / run solver: Can't launch kernel: stage 1 / r = %d / step 1." \
				"OpenCL errorcode: %d\n", r2, err);
			if(cmd_queue_prof_enabled) clReleaseEvent(begin);
			clReleaseMemObject(tmp);
			clReleaseCommandQueue(queue);
			return B4PFM_OPENCL_ERROR;
		}

		global_size[0] = (POW2(k2-2*r2)-1) * 3*POW2(2*r2-2) * solver->kernel_2DCR_threads;
		global_size[1] = count;
		local_size[0] = solver->kernel_2DCR_threads;
		local_size[1] = 1;

		err = clEnqueueNDRangeKernel(queue, solver->kernel_2DCR, 2, 0, 
			global_size, local_size, 0, 0, 0);
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf2D / run solver: Can't launch kernel: stage 1 / r = %d / CR. " \
				"OpenCL errorcode: %d\n", r2, err);
			if(cmd_queue_prof_enabled) clReleaseEvent(begin);
			clReleaseMemObject(tmp);
			clReleaseCommandQueue(queue);
			return B4PFM_OPENCL_ERROR;
		}

		int part_count = 3*POW2(2*r2-2);
		int prev_count;

		do {

			prev_count = part_count;
			part_count /= MIN(part_count, solver->max_sum_size1);

			err |= clSetKernelArg(solver->kernel_2D12, KARGS_2D12_PREV_COUNT,
				sizeof(cl_int), (void *)&prev_count);
			if(err != CL_SUCCESS) {
				printf(
					"(error) b4pdf2D / run solver: Can't set kernel arguments: stage 1 / r = %d / step 2. " \
					"OpenCL errorcode: %d\n", r2, err);
				if(cmd_queue_prof_enabled) clReleaseEvent(begin);
				clReleaseMemObject(tmp);
				clReleaseCommandQueue(queue);
				return B4PFM_OPENCL_ERROR;
			}

			global_size[0] = (POW2(k2-2*r2)-1) * part_count * solver->kernel_2D12_threads;
			global_size[1] = count;
			local_size[0] = solver->kernel_2D12_threads;
			local_size[1] = 1;
			err = clEnqueueNDRangeKernel(queue, solver->kernel_2D12, 2, 0, 
				global_size, local_size, 0, 0, 0);
			if(err != CL_SUCCESS) {
				printf(
					"(error) b4pdf2D / run solver: Can't launch kernel: stage 1 / r = %d / step 2." \
					"OpenCL errorcode: %d\n", r2, err);
				if(cmd_queue_prof_enabled) clReleaseEvent(begin);
				clReleaseMemObject(tmp);
				clReleaseCommandQueue(queue);
				return B4PFM_OPENCL_ERROR;
			}

		} while(part_count > 1);
	}

	if(k2 & 1) {

		cl_int one = 1;
		err |= clSetKernelArg(solver->kernel_2DCR, KARGS_2DCR_RADIX2_2D, 
			sizeof(cl_int), (void *)&one);

		cl_int k2_minus = k2 - 1;

		err  = clSetKernelArg(solver->kernel_Q2_2D21, KARGS_Q2_2D21_R, 
			sizeof(cl_int), (void *)&k2_minus);

		cl_int cr_r2 = k2_minus + 1;
		err |= clSetKernelArg(solver->kernel_2DCR, KARGS_2DCR_R2,
			sizeof(cl_int), (void *)&cr_r2);
		err |= clSetKernelArg(solver->kernel_Q2_2D22, KARGS_Q2_2D22_R,
			sizeof(cl_int), (void *)&k2_minus);
                                
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf2D / run q4-solver / q2-step: Can't set set kernel arguments: stage 2" \
				"OpenCL errorcode: %d\n", err);
			if(cmd_queue_prof_enabled) clReleaseEvent(begin);
			clReleaseMemObject(tmp);
			clReleaseCommandQueue(queue);
			return B4PFM_OPENCL_ERROR;
		}

		size_t global_size[2], local_size[2];

		/* Lauches kernel 1 */
		global_size[0] =  POW2(k2_minus) * solver->kernel_Q2_2D21_threads;
		global_size[1] = count;
		local_size[0] = solver->kernel_Q2_2D21_threads;
		local_size[1] = 1;
		err = clEnqueueNDRangeKernel(queue, solver->kernel_Q2_2D21, 2, 0, 
			global_size, local_size, 0, 0, 0);
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf2D / run q4-solver / q2-step: Can't launch kernel: stage 2 / step 1. " \
				"OpenCL errorcode: %d\n", err);
			if(cmd_queue_prof_enabled) clReleaseEvent(begin);
			clReleaseMemObject(tmp);
			clReleaseCommandQueue(queue);
			return B4PFM_OPENCL_ERROR;
		}

		global_size[0] = POW2(k2_minus) * solver->kernel_2DCR_threads;
		global_size[1] = count;
		local_size[0] = solver->kernel_2DCR_threads;
		local_size[1] = 1;
		err = clEnqueueNDRangeKernel(queue, solver->kernel_2DCR, 2, 0, 
			global_size, local_size, 0, 0, 0);
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf2D / run q4-solver / q2-step: Can't launch kernel: stage 2 / CR." \
				"OpenCL errorcode: %d\n", err);
			if(cmd_queue_prof_enabled) clReleaseEvent(begin);
			clReleaseMemObject(tmp);
			clReleaseCommandQueue(queue);
			return B4PFM_OPENCL_ERROR;
		}

		int part_count = POW2(k2_minus);
		int prev_count;

		do {

			prev_count = part_count;
			part_count /= MIN(part_count, solver->max_sum_size_q2_2);

			err |= clSetKernelArg(solver->kernel_Q2_2D22, KARGS_Q2_2D22_PREV_COUNT,
				sizeof(cl_int), (void *)&prev_count);
			if(err != CL_SUCCESS) {
				printf(
					"(error) b4pdf2D / run q4-solver / q2-step: Can't set kernel arguments: stage 2 / step 2 " \
					"OpenCL errorcode: %d\n", err);
				if(cmd_queue_prof_enabled) clReleaseEvent(begin);
				clReleaseMemObject(tmp);
				clReleaseCommandQueue(queue);
				return B4PFM_OPENCL_ERROR;
			}

			global_size[0] = part_count * solver->kernel_Q2_2D22_threads;
			global_size[1] = count;
			local_size[0] = solver->kernel_Q2_2D22_threads;
			local_size[1] = 1;
			err = clEnqueueNDRangeKernel(queue, solver->kernel_Q2_2D22, 2, 0, 
				global_size, local_size, 0, 0, 0);
			if(err != CL_SUCCESS) {
				printf(
					"(error) b4pdf2D / run q4-solver / q2-step: Can't launch kernel: stage 2 / step 2" \
					"OpenCL errorcode: %d\n", err);
				if(cmd_queue_prof_enabled) clReleaseEvent(begin);
				clReleaseMemObject(tmp);
				clReleaseCommandQueue(queue);
				return B4PFM_OPENCL_ERROR;
			}

		} while(part_count > 1);

		err |= clSetKernelArg(solver->kernel_2DCR, KARGS_2DCR_RADIX2_2D, 
			sizeof(cl_int), (void *)&zero);	

	}


	for(r2 = k2/2-1; r2 >= 0; r2--) {

		err  = clSetKernelArg(solver->kernel_2D21, KARGS_2D21_R, 
			sizeof(cl_int), (void *)&r2);

		cl_int cr_r2 = r2 + 1;
		err |= clSetKernelArg(solver->kernel_2DCR, KARGS_2DCR_R2,
			sizeof(cl_int), (void *)&cr_r2);
		err |= clSetKernelArg(solver->kernel_2D22A, KARGS_2D22A_R,
			sizeof(cl_int), (void *)&r2);
		err |= clSetKernelArg(solver->kernel_2D22B, KARGS_2D22B_R,
			sizeof(cl_int), (void *)&r2);
                                
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf2D / run solver: Can't set set kernel arguments: stage 2 / r = %d." \
				"OpenCL errorcode: %d\n", r2, err);
			if(cmd_queue_prof_enabled) clReleaseEvent(begin);
			clReleaseMemObject(tmp);
			clReleaseCommandQueue(queue);
			return B4PFM_OPENCL_ERROR;
		}

		size_t global_size[2], local_size[2];

		/* Lauches kernel 1 */
		global_size[0] = (POW2(k2-2*r2-2)) * 3*POW2(2*r2) * solver->kernel_2D21_threads;
		global_size[1] = count;
		local_size[0] = solver->kernel_2D21_threads;
		local_size[1] = 1;
		err = clEnqueueNDRangeKernel(queue, solver->kernel_2D21, 2, 0, 
			global_size, local_size, 0, 0, 0);
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf2D / run solver: Can't launch kernel: stage 2 / r = %d, step 1. " \
				"OpenCL errorcode: %d\n", r2, err);
			if(cmd_queue_prof_enabled) clReleaseEvent(begin);
			clReleaseMemObject(tmp);
			clReleaseCommandQueue(queue);
			return B4PFM_OPENCL_ERROR;
		}

		global_size[0] = (POW2(k2-2*r2-2)) * 3*POW2(2*r2) * solver->kernel_2DCR_threads;
		global_size[1] = count;
		local_size[0] = solver->kernel_2DCR_threads;
		local_size[1] = 1;
		err = clEnqueueNDRangeKernel(queue, solver->kernel_2DCR, 2, 0, 
			global_size, local_size, 0, 0, 0);
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf2D / run solver: Can't launch kernel: stage 2 / r = %d, CR." \
				"OpenCL errorcode: %d\n", r2, err);
			if(cmd_queue_prof_enabled) clReleaseEvent(begin);
			clReleaseMemObject(tmp);
			clReleaseCommandQueue(queue);
			return B4PFM_OPENCL_ERROR;
		}

		int part_count = 3*POW2(2*r2) / MIN(3*POW2(2*r2), solver->max_sum_size2a);

		global_size[0] = POW2(k2-2*r2-2) * part_count * solver->kernel_2D22A_threads;
		global_size[1] = count;
		local_size[0] = solver->kernel_2D22A_threads;
		local_size[1] = 1;
		err = clEnqueueNDRangeKernel(queue, solver->kernel_2D22A, 2, 0, 
			global_size, local_size, 0, 0, 0);
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf2D / run solver: Can't launch kernel: stage 2 / r = %d / step 2 / kernel A." \
				"OpenCL errorcode: %d\n", r2, err);
			if(cmd_queue_prof_enabled) clReleaseEvent(begin);
			clReleaseMemObject(tmp);
			clReleaseCommandQueue(queue);
			return B4PFM_OPENCL_ERROR;
		}
		
		int prev_count;

		do {

			prev_count = part_count;
			part_count /= MIN(part_count, solver->max_sum_size2b);

			err |= clSetKernelArg(solver->kernel_2D22B, KARGS_2D22B_PREV_COUNT,
				sizeof(cl_int), (void *)&prev_count);
			if(err != CL_SUCCESS) {
				printf(
					"(error) b4pdf2D / run solver: Can't set kernel arguments: stage 2 / r = %d / step 2 / kernel B. " \
					"OpenCL errorcode: %d\n", r2, err);
				if(cmd_queue_prof_enabled) clReleaseEvent(begin);
				clReleaseMemObject(tmp);
				clReleaseCommandQueue(queue);
				return B4PFM_OPENCL_ERROR;
			}

			global_size[0] = 3*POW2(k2-2*r2-2) * part_count * solver->kernel_2D22B_threads;
			global_size[1] = count;
			local_size[0] = solver->kernel_2D22B_threads;
			local_size[1] = 1;
			err = clEnqueueNDRangeKernel(queue, solver->kernel_2D22B, 2, 0, 
				global_size, local_size, 0, 0, 0);
			if(err != CL_SUCCESS) {
				printf(
					"(error) b4pdf2D / run solver: Can't launch kernel: stage 2 / r = %d / step 2 / kernel B." \
					"OpenCL errorcode: %d\n", r2, err);
				if(cmd_queue_prof_enabled) clReleaseEvent(begin);
				clReleaseMemObject(tmp);
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
				"(error) b4pdf2D / run solver: Can't finish command queue." \
				"OpenCL errorcode: %d\n",  err);
			clReleaseEvent(end);
			clReleaseEvent(begin);
			clReleaseMemObject(tmp);
			clReleaseCommandQueue(queue);
			return B4PFM_OPENCL_ERROR;
		}

		cl_ulong start, stop;

		err  = clGetEventProfilingInfo(begin, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &start, NULL);
		err |= clGetEventProfilingInfo(end, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &stop, NULL);

		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf2D / run solver: Can't get event profiling info. " \
				"OpenCL errorcode: %d\n",  err);
			time = -1;
		} else
			time = (stop-start)*1.0e-9;

		clReleaseEvent(end);
		clReleaseEvent(begin);

		if(!cmd_queue_prof_was_enabled) {
			clSetCommandQueueProperty(queue, CL_QUEUE_PROFILING_ENABLE, CL_FALSE, 0);

			printf(
				"(debug) b4pdf2D / run solver: OpenCL command queue profiling disabled. \n");
		}
	}

	clReleaseMemObject(tmp);
	clReleaseCommandQueue(queue);

	if(debug != B4PFM_DEBUG_NONE && !cmd_queue_prof_enabled) printf(
		"(debug) b4pdf2D / run solver: Ready.\n");

	if(cmd_queue_prof_enabled) printf(
		"(debug) b4pdf2D / run solver: Ready. Computing time: %fs\n", time);

	return B4PFM_OK;

}

int run_2d_q2_solver(r_b4pfm2D *solver, cl_command_queue o_queue, cl_mem f, cl_mem o_tmp, int count, int r1, int o_r2, int debug, int *o_err_info) {

	int *err_info;
	int tmp_err_info;
	if(o_err_info == 0)
		err_info = &tmp_err_info;
	else
		err_info = o_err_info;

	if(solver == NULL)
		return B4PFM_INVALID_SOLVER;


	if(debug == B4PFM_DEBUG_FULL || debug == B4PFM_DEBUG_TIMING) printf(
		"(debug) b4pdf2D / run q2-solver: count=%d, r1=%d, o_r2=%d\n", count, r1, o_r2);
	else if(debug != B4PFM_DEBUG_NONE) printf(
		"(debug) b4pdf2D / run q2-solver\n");

	int err;

	int var_size = solver->double_prec ? sizeof(cl_double) : sizeof(cl_float);

	int cmd_queue_prof_was_enabled = 0;
	int cmd_queue_prof_enabled = 0;

	/* Creates a new opencl command queue if necessary. */
	cl_command_queue queue;
	if(o_queue == 0) {
		if(debug != B4PFM_DEBUG_NONE) printf(
			"(debug) b4pdf2D / run q2-solver: No OpenCL Commandqueue given. "\
			"Creating OpenCL Commandqueue... \n");
		cl_device_id device;
 		err = clGetContextInfo(solver->context, CL_CONTEXT_DEVICES, 
			sizeof(cl_device_id), &device, 0);
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf2D / run q2-solver: Can't query device id from context. " \
				"OpenCL errorcode: %d\n", err);
			return B4PFM_OPENCL_ERROR;
		}

		if(debug == B4PFM_DEBUG_TIMING) {
			queue = clCreateCommandQueue(solver->context, device, CL_QUEUE_PROFILING_ENABLE, &err);
			printf(
				"(debug) b4pdf2D / run q2-solver: OpenCL command queue profiling enabled. \n");
		} else
			queue = clCreateCommandQueue(solver->context, device, 0, &err);
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf2D / run q2-solver: Can't create command queue. " \
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
					"(error) b4pdf2D / run q2-solver: Can't enable OpenCL command queue profiling. " 
					"OpenCL errorcode: %d.\n", err);

			if(cmd_queue_prof_enabled && !cmd_queue_prof_was_enabled)
				printf(
					"(debug) b4pdf2D / run q2-solver: OpenCL command queue profiling enabled. \n");
		}

	}

	/* Creates a new opencl memory buffer for temporar data */
	cl_mem tmp;
	if(o_tmp == 0) {
		if(debug != B4PFM_DEBUG_NONE) printf(
			"(debug) b4pdf2D / run q2-solver: No global workplace given. " \
			"Creating global workplace buffer... \n");
		int tmp_mem_size = count*POW2(solver->k2-1)*POW2(solver->k3)*var_size;
		if(debug == B4PFM_DEBUG_FULL || debug == B4PFM_DEBUG_TIMING) printf(
			"(debug) b4pdf2D / run q2-solver: Global workplace size = %d bytes. \n",
			tmp_mem_size);
		tmp = clCreateBuffer(solver->context, CL_MEM_READ_WRITE, tmp_mem_size, 0, &err);
        
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf2D / run q2-solver: Can't allocate device memory for " \
				"workplace-buffer. OpenCL errorcode: %d\n.", err);
			clReleaseCommandQueue(queue);   
			return B4PFM_OPENCL_ERROR; 
		}
	} else {
		clRetainMemObject(o_tmp);
		tmp = o_tmp;
	}

	/* Set initial kernel arguments */
	err  = clSetKernelArg(solver->kernel_Q2_2D11, KARGS_Q2_2D11_O_F, 
		sizeof(cl_mem), (void *)&f);
	err |= clSetKernelArg(solver->kernel_Q2_2D12, KARGS_Q2_2D12_O_F, 
		sizeof(cl_mem), (void *)&f);
	err |= clSetKernelArg(solver->kernel_Q2_2D21, KARGS_Q2_2D21_O_F, 
		sizeof(cl_mem), (void *)&f);
	err |= clSetKernelArg(solver->kernel_Q2_2D22, KARGS_Q2_2D22_O_F, 
		sizeof(cl_mem), (void *)&f);

	err  = clSetKernelArg(solver->kernel_Q2_2D11, KARGS_Q2_2D11_O_G, 
		sizeof(cl_mem), (void *)&tmp);
	err |= clSetKernelArg(solver->kernel_Q2_2D12, KARGS_Q2_2D12_O_G, 
		sizeof(cl_mem), (void *)&tmp);
	err |= clSetKernelArg(solver->kernel_Q2_2D12, KARGS_Q2_2D12_O_G, 
		sizeof(cl_mem), (void *)&tmp);
	err |= clSetKernelArg(solver->kernel_Q2_2D21, KARGS_Q2_2D21_O_G, 
		sizeof(cl_mem), (void *)&tmp);
	err |= clSetKernelArg(solver->kernel_Q2_2D22, KARGS_Q2_2D22_O_G, 
		sizeof(cl_mem), (void *)&tmp);

	err |= clSetKernelArg(solver->kernel_2DCR, KARGS_2DCR_O_F, 
		sizeof(cl_mem), (void *)&tmp);
	err |= clSetKernelArg(solver->kernel_2DCR, KARGS_2DCR_R1, 
		sizeof(cl_int), (void *)&r1);

	cl_int one = 1;
	err |= clSetKernelArg(solver->kernel_2DCR, KARGS_2DCR_RADIX2_2D, 
		sizeof(cl_int), (void *)&one);
	err |= clSetKernelArg(solver->kernel_2DCR, KARGS_2DCR_RADIX2_3D, 
		sizeof(cl_int), (void *)&one);


	if(err != CL_SUCCESS) {
		printf(
			"(error) b4pdf2D / run q2-solver: Can't set initial kernel arguments. " \
			"OpenCL errorcode: %d\n", err);
		clReleaseMemObject(tmp);
		clReleaseCommandQueue(queue);
		return B4PFM_OPENCL_ERROR;
	}

	cl_event begin, end;

	if(cmd_queue_prof_enabled) {
		clFinish(queue);
		clEnqueueMarker(queue, &begin);
	}

	int k2 = solver->k2;

	int r2;
	for(r2 = o_r2; r2 <= k2 - 1; r2++) {

		err  = clSetKernelArg(solver->kernel_Q2_2D11, KARGS_Q2_2D11_R, 
			sizeof(cl_int), (void *)&r2);
		err |= clSetKernelArg(solver->kernel_2DCR, KARGS_2DCR_R2,
			sizeof(cl_int), (void *)&r2);
		err |= clSetKernelArg(solver->kernel_Q2_2D12, KARGS_Q2_2D12_R,
			sizeof(cl_int), (void *)&r2);
                                
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf2D / run q2-solver: Can't set kernel arguments: stage 1 / r = %d. " \
				"OpenCL errorcode: %d\n", r2, err);
			if(cmd_queue_prof_enabled) clReleaseEvent(begin);
			clReleaseMemObject(tmp);
			clReleaseCommandQueue(queue);
			return B4PFM_OPENCL_ERROR;
		}

		size_t global_size[2], local_size[2];

		/* Lauches kernel  */
		global_size[0] = (POW2(k2-r2)-1) * POW2(r2-1) * solver->kernel_Q2_2D11_threads;
		global_size[1] = count;
		local_size[0] = solver->kernel_Q2_2D11_threads;
		local_size[1] = 1;
		err = clEnqueueNDRangeKernel(queue, solver->kernel_Q2_2D11, 2, 0, 
			global_size, local_size, 0, 0, 0);
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf2D / run q2-solver: Can't launch kernel: stage 1 / r = %d / step 1." \
				"OpenCL errorcode: %d\n", r2, err);
			if(cmd_queue_prof_enabled) clReleaseEvent(begin);
			clReleaseMemObject(tmp);
			clReleaseCommandQueue(queue);
			return B4PFM_OPENCL_ERROR;
		}

		global_size[0] = (POW2(k2-r2)-1) * POW2(r2-1) * solver->kernel_2DCR_threads;
		global_size[1] = count;
		local_size[0] = solver->kernel_2DCR_threads;
		local_size[1] = 1;

		err = clEnqueueNDRangeKernel(queue, solver->kernel_2DCR, 2, 0, 
			global_size, local_size, 0, 0, 0);
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf2D / run q2-solver: Can't launch kernel: stage 1 / r = %d / CR. " \
				"OpenCL errorcode: %d\n", r2, err);
			if(cmd_queue_prof_enabled) clReleaseEvent(begin);
			clReleaseMemObject(tmp);
			clReleaseCommandQueue(queue);
			return B4PFM_OPENCL_ERROR;
		}

		int part_count = POW2(r2-1);
		int prev_count;

		do {

			prev_count = part_count;
			part_count /= MIN(part_count, solver->max_sum_size_q2_1);

			err |= clSetKernelArg(solver->kernel_Q2_2D12, KARGS_Q2_2D12_PREV_COUNT,
				sizeof(cl_int), (void *)&prev_count);
			if(err != CL_SUCCESS) {
				printf(
					"(error) b4pdf2D / run q2-solver: Can't set kernel arguments: stage 1 / r = %d / step 2. " \
					"OpenCL errorcode: %d\n", r2, err);
				if(cmd_queue_prof_enabled) clReleaseEvent(begin);
				clReleaseMemObject(tmp);
				clReleaseCommandQueue(queue);
				return B4PFM_OPENCL_ERROR;
			}

			global_size[0] = (POW2(k2-r2)-1) * part_count * solver->kernel_Q2_2D12_threads;
			global_size[1] = count;
			local_size[0] = solver->kernel_Q2_2D12_threads;
			local_size[1] = 1;
			err = clEnqueueNDRangeKernel(queue, solver->kernel_Q2_2D12, 2, 0, 
				global_size, local_size, 0, 0, 0);
			if(err != CL_SUCCESS) {
				printf(
					"(error) b4pdf2D / run q2-solver: Can't launch kernel: stage 1 / r = %d / step 2." \
					"OpenCL errorcode: %d\n", r2, err);
				if(cmd_queue_prof_enabled) clReleaseEvent(begin);
				clReleaseMemObject(tmp);
				clReleaseCommandQueue(queue);
				return B4PFM_OPENCL_ERROR;
			}

		} while(part_count > 1);
	}

	for(r2 = k2-1; r2 >= o_r2-1; r2--) {

		err  = clSetKernelArg(solver->kernel_Q2_2D21, KARGS_Q2_2D21_R, 
			sizeof(cl_int), (void *)&r2);

		cl_int cr_r2 = r2 + 1;
		err |= clSetKernelArg(solver->kernel_2DCR, KARGS_2DCR_R2,
			sizeof(cl_int), (void *)&cr_r2);
		err |= clSetKernelArg(solver->kernel_Q2_2D22, KARGS_Q2_2D22_R,
			sizeof(cl_int), (void *)&r2);
                                
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf2D / run q2-solver: Can't set set kernel arguments: stage 2 / r = %d." \
				"OpenCL errorcode: %d\n", r2, err);
			if(cmd_queue_prof_enabled) clReleaseEvent(begin);
			clReleaseMemObject(tmp);
			clReleaseCommandQueue(queue);
			return B4PFM_OPENCL_ERROR;
		}

		size_t global_size[2], local_size[2];

		/* Lauches kernel 1 */
		global_size[0] = (POW2(k2-r2-1)) * POW2(r2) * solver->kernel_Q2_2D21_threads;
		global_size[1] = count;
		local_size[0] = solver->kernel_Q2_2D21_threads;
		local_size[1] = 1;
		err = clEnqueueNDRangeKernel(queue, solver->kernel_Q2_2D21, 2, 0, 
			global_size, local_size, 0, 0, 0);
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf2D / run q2-solver: Can't launch kernel: stage 2 / r = %d, step 1. " \
				"OpenCL errorcode: %d\n", r2, err);
			if(cmd_queue_prof_enabled) clReleaseEvent(begin);
			clReleaseMemObject(tmp);
			clReleaseCommandQueue(queue);
			return B4PFM_OPENCL_ERROR;
		}

		global_size[0] = (POW2(k2-r2-1)) * POW2(r2) * solver->kernel_2DCR_threads;
		global_size[1] = count;
		local_size[0] = solver->kernel_2DCR_threads;
		local_size[1] = 1;
		err = clEnqueueNDRangeKernel(queue, solver->kernel_2DCR, 2, 0, 
			global_size, local_size, 0, 0, 0);
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf2D / run q2-solver: Can't launch kernel: stage 2 / r = %d, CR." \
				"OpenCL errorcode: %d\n", r2, err);
			if(cmd_queue_prof_enabled) clReleaseEvent(begin);
			clReleaseMemObject(tmp);
			clReleaseCommandQueue(queue);
			return B4PFM_OPENCL_ERROR;
		}

		int part_count = POW2(r2);
		int prev_count;

		do {

			prev_count = part_count;
			part_count /= MIN(part_count, solver->max_sum_size_q2_2);

			err |= clSetKernelArg(solver->kernel_Q2_2D22, KARGS_Q2_2D22_PREV_COUNT,
				sizeof(cl_int), (void *)&prev_count);
			if(err != CL_SUCCESS) {
				printf(
					"(error) b4pdf2D / run q2-solver: Can't set kernel arguments: stage 2 / r = %d / step 2 " \
					"OpenCL errorcode: %d\n", r2, err);
				if(cmd_queue_prof_enabled) clReleaseEvent(begin);
				clReleaseMemObject(tmp);
				clReleaseCommandQueue(queue);
				return B4PFM_OPENCL_ERROR;
			}

			global_size[0] = POW2(k2-r2-1) * part_count * solver->kernel_Q2_2D22_threads;
			global_size[1] = count;
			local_size[0] = solver->kernel_Q2_2D22_threads;
			local_size[1] = 1;
			err = clEnqueueNDRangeKernel(queue, solver->kernel_Q2_2D22, 2, 0, 
				global_size, local_size, 0, 0, 0);
			if(err != CL_SUCCESS) {
				printf(
					"(error) b4pdf2D / run q2-solver: Can't launch kernel: stage 2 / r = %d / step 2" \
					"OpenCL errorcode: %d\n", r2, err);
				if(cmd_queue_prof_enabled) clReleaseEvent(begin);
				clReleaseMemObject(tmp);
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
				"(error) b4pdf2D / run q2-solver: Can't finish command queue." \
				"OpenCL errorcode: %d\n",  err);
			clReleaseEvent(end);
			clReleaseEvent(begin);
			clReleaseMemObject(tmp);
			clReleaseCommandQueue(queue);
			return B4PFM_OPENCL_ERROR;
		}

		cl_ulong start, stop;

		err  = clGetEventProfilingInfo(begin, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &start, NULL);
		err |= clGetEventProfilingInfo(end, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &stop, NULL);

		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf2D / run q2-solver: Can't get event profiling info. " \
				"OpenCL errorcode: %d\n",  err);
			time = -1;
		} else
			time = (stop-start)*1.0e-9;

		clReleaseEvent(end);
		clReleaseEvent(begin);

		if(!cmd_queue_prof_was_enabled) {
			clSetCommandQueueProperty(queue, CL_QUEUE_PROFILING_ENABLE, CL_FALSE, 0);

			printf(
				"(debug) b4pdf2D / run q2-solver: OpenCL command queue profiling disabled. \n");
		}
	}

	clReleaseMemObject(tmp);
	clReleaseCommandQueue(queue);

	if(debug != B4PFM_DEBUG_NONE && !cmd_queue_prof_enabled) printf(
		"(debug) b4pdf2D / run q2-solver: Ready.\n");

	if(cmd_queue_prof_enabled) printf(
		"(debug) b4pdf2D / run q2-solver: Ready. Computing time: %fs\n", time);

	return B4PFM_OK;

}

int load_and_run_2d_solver(r_b4pfm2D *solver, void *f, int debug, int *o_err_info) {

	int *err_info;
	int tmp_err_info;
	if(o_err_info == 0)
		err_info = &tmp_err_info;
	else
		err_info = o_err_info;

	int err;

	int k2 = solver->k2;
	int ldf = solver->ldf;
	int n2 = POW2(k2)-1;

	if(solver == NULL)
		return B4PFM_INVALID_SOLVER;

	cl_device_id device;
	err = clGetContextInfo(solver->context, CL_CONTEXT_DEVICES, 
		sizeof(cl_device_id), &device, 0);
                
	if(err != CL_SUCCESS) {
		printf(
			"(error) b4pdf2D / pre-run solver: Can't query device id from context. " \
			"OpenCL errorcode: %d\n", err);
		return B4PFM_OPENCL_ERROR;
	}

	int cmd_queue_prof_enabled = 0;

	cl_command_queue queue;
                
	if(debug == B4PFM_DEBUG_TIMING) {
		queue = clCreateCommandQueue(solver->context, device, CL_QUEUE_PROFILING_ENABLE, &err);
		printf(
			"(debug) b4pdf2D / pre-run solver: OpenCL command queue profiling enabled. \n");
		cmd_queue_prof_enabled = 1;
	} else
		queue = clCreateCommandQueue(solver->context, device, 0, &err);

	if(err != CL_SUCCESS) {
		printf(
			"(error) b4pdf2D / pre-run solver: Can't create command queue. " \
			"OpenCL errorcode: %d\n.", err);
		return B4PFM_OPENCL_ERROR;
	}

	cl_event begin, end;

	if(cmd_queue_prof_enabled) {
		clFinish(queue);
		clEnqueueMarker(queue, &begin);
	}

	int var_size = solver->double_prec ? sizeof(cl_double) : sizeof(cl_float);

	cl_mem f_buffer = clCreateBuffer(solver->context, CL_MEM_READ_WRITE, n2*ldf*var_size, 0, &err);
        
	if(err != CL_SUCCESS) {
		printf(
			"(error) b4pdf2D / pre-run solver: Can't allocate device memory for " \
			"f-buffer. OpenCL errorcode: %d\n.", err);
		if(cmd_queue_prof_enabled) clReleaseEvent(begin);
		clReleaseCommandQueue(queue);   
		return B4PFM_OPENCL_ERROR; 
	}

	err = clEnqueueWriteBuffer(queue, f_buffer, CL_TRUE, 0, n2*ldf*var_size, f, 0, 0, 0);

	if(err != CL_SUCCESS) {
		printf(
			"(error) b4pdf2D / pre-run solver: Can't write into f-buffer. " \
			"OpenCL errorcode: %d\n.", err);
		if(cmd_queue_prof_enabled) clReleaseEvent(begin);
		clReleaseCommandQueue(queue);
		clReleaseMemObject(f_buffer);
		return B4PFM_OPENCL_ERROR;
	}

	int err_info2;
	if(solver->force_radix2)
		err = run_2d_q2_solver(solver, queue, f_buffer, 0, 1, 0, 1, debug, &err_info2);
	else
		err = run_2d_solver(solver, queue, f_buffer, 0, 1, 0, debug, &err_info2);

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
			"(error) b4pdf2D / pre-run solver: Can't finish workqueue. " \
			"OpenCL errorcode: %d\n", err);
		if(cmd_queue_prof_enabled) clReleaseEvent(begin);
		clReleaseCommandQueue(queue);
		clReleaseMemObject(f_buffer);
		return B4PFM_OPENCL_ERROR;
	}

	err = clEnqueueReadBuffer(queue, f_buffer, CL_TRUE, 0, n2*ldf*var_size, f, 0, 0, 0);
        
	if(err != CL_SUCCESS) {
		printf(
			"(error) b4pdf2D / pre-run solver: Can't read from f-buffer. " \
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
				"(error) b4pdf2D / pre-run solver: Can't finish command queue." \
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
				"(error) b4pdf2D / pre-run solver: Can't get event profiling info. " \
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
		"(debug) b4pdf2D / pre-run solver: Total time: %fs\n", time);

	return B4PFM_OK;
}

int default_opt_2d_solver(cl_context context, b4pfm2D_params* opt_params, int k2, int k3, int ldf, int prec, int debug, int *o_err_info) {

	int *err_info;
	int tmp_err_info;
	if(o_err_info == 0)
		err_info = &tmp_err_info;
	else
		err_info = o_err_info;

	int err;

	int var_size = prec ? sizeof(cl_double) : sizeof(cl_float);

	opt_params->use_local_coef = 0;
	opt_params->force_priv_coef = 0;
	opt_params->hilodouble = 0;

	/* Find the OpenCL device corresponding to given context */
	cl_device_id device;

	err = clGetContextInfo(context, CL_CONTEXT_DEVICES, 
		sizeof(cl_device_id), &device, 0);

	if(err != CL_SUCCESS) {
		printf(
			"(error) b4pdf2D / default opt.: Can't query device id from context. " \
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
			"(debug) b4pdf2D / default opt.: nVidia videocard found.\n");
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

	int cr_max_local_mem_size = max_local_mem_size - (opt_params->use_local_coef ? 2*k3 : 0)*var_size - CR_LOCAL_MEM_BIAS;

	/*cr_max_local_mem_size = MIN(2*512,cr_max_local_mem_size); */

	int max_cr_work_group_size = MIN(max_work_item_sizes[0], cr_max_local_mem_size/var_size);

	if(debug == B4PFM_DEBUG_FULL || debug == B4PFM_DEBUG_TIMING) printf(
		"(debug) b4pdf2D / default opt.: Max workgroup size for tridiagonal solver = %d.\n",
		max_cr_work_group_size);

	opt_params->cr_local_mem_size = POW2(k3);
	while(opt_params->cr_local_mem_size > cr_max_local_mem_size/var_size)
		opt_params->cr_local_mem_size /= 2;

	if(debug == B4PFM_DEBUG_FULL || debug == B4PFM_DEBUG_TIMING) printf(
		"(debug) b4pdf2D / default opt.: Local mem usage for tridiagonal solver = %d + %d bytes.\n",
		opt_params->cr_local_mem_size*var_size, (opt_params->use_local_coef ? 2*k3 : 0)*var_size);	

	opt_params->cr_wg_size = MAX(opt_params->work_block_size, POW2(k3-1));
	while(opt_params->cr_wg_size > max_cr_work_group_size)
		opt_params->cr_wg_size /= 2;

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

	opt_params->max_sum_size1 = 4;
	opt_params->max_sum_size2a = 4;
	opt_params->max_sum_size2b = 4;	

	opt_params->max_sum_size_q2_1 = 2;
	opt_params->max_sum_size_q2_2 = 2;	

	if(debug == B4PFM_DEBUG_FULL || debug == B4PFM_DEBUG_TIMING) { 
		printf("(debug) b4pdf2D / default opt.: Final opt_params = \n");
		print_opt_params_2d(opt_params);
	}


	return B4PFM_OK;
}

double run_speed_test_2d(cl_context context, cl_command_queue queue, cl_mem tmp1, cl_mem tmp2, b4pfm2D_params opt_params, int force_radix2, int k2, int k3, int ldf, int prec, int debug) {

	if(debug == B4PFM_DEBUG_FULL || debug == B4PFM_DEBUG_TIMING) {
		printf("(debug) b4pdf2D / run_speed_test_2d:Current opt_params = \n");
		print_opt_params_2d(&opt_params);
	}

	int err;
	r_b4pfm2D* solver = init_2d_solver(context, 0, k2, k3, ldf, prec, opt_params, force_radix2, B4PFM_DEBUG_NONE, &err, 0);
	if(err != B4PFM_OK)
		return 1.0 / 0.0;

	cl_device_id device;
 	err = clGetContextInfo(context, CL_CONTEXT_DEVICES, 
		sizeof(cl_device_id), &device, 0);
	if(err != CL_SUCCESS) {
		printf(
			"(error) b4pdf2D / run_speed_test_2d: Can't query device id from context. " \
			"OpenCL errorcode: %d\n", err);
		return 1.0 / 0.0;
	}
	
	/* Pre-init kernels... */
	if(force_radix2)
		err = run_2d_q2_solver(solver, queue, tmp1, tmp2, 1, 0, 1, B4PFM_DEBUG_NONE, 0);
	else
		err = run_2d_solver(solver, queue, tmp1, tmp2, 1, 0, B4PFM_DEBUG_NONE, 0);
	if(err != B4PFM_OK) {
		free_2d_solver(solver);
		return 1.0 / 0.0;
	}

	clFinish(queue);

	cl_ulong start, stop;

	cl_event begin, end;

	clEnqueueMarker(queue, &begin);

	if(force_radix2)
		err = run_2d_q2_solver(solver, queue, tmp1, tmp2, 1, 0, 1, B4PFM_DEBUG_NONE, 0);
	else
		err = run_2d_solver(solver, queue, tmp1, tmp2, 1, 0, B4PFM_DEBUG_NONE, 0);
	if(err != B4PFM_OK) {
		clReleaseEvent(begin);
		free_2d_solver(solver);
		return 1.0 / 0.0;
	}

	clEnqueueMarker(queue, &end);

	err = clFinish(queue);
	if(err != CL_SUCCESS) {
		printf("(error) b4pdf2D / run_speed_test_2d: Can't finish command queue. OpenCL errorcode: %d\n", err);
		return -1;
	}

	err  = clGetEventProfilingInfo(begin, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &start, NULL);
	err |= clGetEventProfilingInfo(end, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &stop, NULL);

	double final;

	if(err != CL_SUCCESS) {
		printf(
			"(error) b4pdf2D / run solver: Can't get event profiling info. " \
			"OpenCL errorcode: %d\n",  err);
		clReleaseEvent(end);	
		clReleaseEvent(begin);
		free_2d_solver(solver);
		return 1.0 / 0.0;
	} else
		final = (stop-start)*1.0e-9;

	clReleaseEvent(end);	
	clReleaseEvent(begin);
	free_2d_solver(solver);

	if(debug == B4PFM_DEBUG_FULL || debug == B4PFM_DEBUG_TIMING) 
		printf("Current time: %f\n", final);

	return final;
}

int auto_optimize_2d_solver(cl_context o_context, cl_mem o_tmp1, cl_mem o_tmp2, b4pfm2D_params* opt_params, int force_radix2, int k2, int k3, int ldf, int prec, int debug, int *o_err_info) {

	int *err_info;
	int tmp_err_info;
	if(o_err_info == 0)
		err_info = &tmp_err_info;
	else
		err_info = o_err_info;

	int err;

	int n2 = POW2(k2)-1;

	cl_context context;

	/* Create OpenCL context  */
	if(o_context == 0) {
		if(debug != B4PFM_DEBUG_NONE) printf(
			"(debug) b4pdf2D / auto opt.: No OpenCL context given. " \
			"Creating an OpenCL context... \n");

		cl_platform_id platform;
		cl_uint num;
		err = clGetPlatformIDs(1, &platform, &num);
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf2D / auto opt.: Can't query OpenCL platform. " \
				"OpenCL errorcode: %d\n", err);
			return B4PFM_OPENCL_ERROR;
		}

		if(num < 1) {
			printf(
				"(error) b4pdf2D / auto opt.: No OpenCL platforms. \n");
			return B4PFM_OPENCL_ERROR;
		}
                       
		cl_device_id device;
        
		err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, 0);
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf2D / auto opt.: Can't query OpenCL devices. " \
				"OpenCL errorcode: %d\n", err);
			return B4PFM_OPENCL_ERROR;
		}

		context = clCreateContext(0, 1, &device, 0, 0, &err);
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf2D / auto opt.: Can't create OpenCL context. " \
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
			"(error) b4pdf2D / auto opt.: Can't query device id from context. " \
			"OpenCL errorcode: %d\n", err);
		clReleaseContext(context);
		return B4PFM_OPENCL_ERROR;
	}

	int var_size = prec ? sizeof(cl_double) : sizeof(cl_float);

	cl_mem tmp1, tmp2;

	if(o_tmp1 == 0) {
		if(debug != B4PFM_DEBUG_NONE) printf(
			"(debug) b4pdf2D / auto opt.: No tmp1-buffer given. " \
			"Creating tmp1-buffer... \n");

		tmp1 = clCreateBuffer(context, CL_MEM_READ_WRITE, n2*ldf*var_size, 0, &err);
        
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf2D / auto opt.: Can't allocate device memory for " \
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
			"(debug) b4pdf2D / auto opt.: No tmp2-buffer given. " \
			"Creating tmp2-buffer... \n");
		int tmp_mem_size = 3*POW2(k2-2)*POW2(k3)*var_size;

		tmp2 = clCreateBuffer(context, CL_MEM_READ_WRITE, tmp_mem_size, 0, &err);
        
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf2D / auto opt.: Can't allocate device memory for " \
				"tmp2-buffer. OpenCL errorcode: %d\n", err);
			clReleaseMemObject(tmp1); 
			clReleaseContext(context);  
			return B4PFM_OPENCL_ERROR; 
		}		

		if(debug == B4PFM_DEBUG_FULL || debug == B4PFM_DEBUG_TIMING) printf(
			"(debug) b4pdf2D / auto opt.: tmp2-buffer size = %d bytes. \n",
			tmp_mem_size);

	} else {
		clRetainMemObject(o_tmp2);
		tmp2 = o_tmp2;
	}

	cl_command_queue queue;
	queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
	if(err != CL_SUCCESS) {
		printf(
			"(error) b4pdf2D / auto opt.: Can't create command queue. " \
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
	
	b4pfm2D_params params;
	default_opt_2d_solver(context, &params, k2, k3, ldf, prec, B4PFM_DEBUG_NONE, 0);

	b4pfm2D_params best_params = params;
	double best_time = 1.0/0.0;

#define handle_result() \
	double time = run_speed_test_2d(context, queue, tmp1, tmp2, params, force_radix2, k2, k3, ldf, prec, debug); \
	if(time < 0) { \
		clReleaseCommandQueue(queue); \
		queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err); \
		if(err != CL_SUCCESS) { \
			printf("(error) b4pdf2D / auto opt.: Can't create new command queue. OpenCL errorcode: %d \n", err); \
			clReleaseMemObject(tmp2); \
			clReleaseMemObject(tmp1); \
			clReleaseContext(context); \
			return B4PFM_OPENCL_ERROR; \
		} \
		printf("(warning) b4pdf2D / auto opt.: New command queue created. \n"); \
	} else if(time < best_time) { \
		best_time = time; \
		best_params = params; \
	}

	params.use_local_coef = 0;
	for(params.force_priv_coef = 0; params.force_priv_coef <= 1; params.force_priv_coef++) {
		for(params.hilodouble = 0; params.hilodouble <= 1; params.hilodouble++) {
			for(params.cr_wg_size = params.work_block_size; params.cr_wg_size == params.work_block_size || params.cr_wg_size <= POW2(k3-1); params.cr_wg_size *= 2) {
				for(params.cr_local_mem_size = MIN(POW2(k3),params.cr_wg_size); params.cr_local_mem_size == MIN(POW2(k3),params.cr_wg_size) || params.cr_local_mem_size <= POW2(k3); params.cr_local_mem_size *= 2) {
					handle_result();
				}
			}
		}
	}

	params = best_params;
	params.use_local_coef = 1;
	params.force_priv_coef = 0;
	for(params.hilodouble = 0; params.hilodouble <= 1; params.hilodouble++) {
		for(params.cr_wg_size = params.work_block_size; params.cr_wg_size == params.work_block_size || params.cr_wg_size <= POW2(k3-1); params.cr_wg_size *= 2) {
			for(params.cr_local_mem_size = MIN(POW2(k3),params.cr_wg_size); params.cr_local_mem_size == MIN(POW2(k3),params.cr_wg_size) || params.cr_local_mem_size <= POW2(k3); params.cr_local_mem_size *= 2) {
				handle_result();
			}
		}
	}

	int vw;
	for(vw = 1; vw <= (prec ? 2 : 4); vw *= 2) {

		if(!force_radix2) {	
	
			params = best_params;
			params.vector_width = vw;
			for(params.other_wg_size_11 = params.work_block_size; params.other_wg_size_11 == params.work_block_size || params.other_wg_size_11 <= POW2(k3)/params.vector_width; params.other_wg_size_11 *= 2) {
				handle_result();
			}

			params = best_params;
			params.vector_width = vw;
			for(params.other_wg_size_21 = params.work_block_size; params.other_wg_size_21 == params.work_block_size || params.other_wg_size_21 <= POW2(k3)/params.vector_width; params.other_wg_size_21 *= 2) {
				handle_result();
			}
	

			params = best_params;
			params.vector_width = vw;
			for(params.other_wg_size_12 = params.work_block_size; params.other_wg_size_12 == params.work_block_size || params.other_wg_size_12 <= POW2(k3)/params.vector_width; params.other_wg_size_12 *= 2) {
				for(params.max_sum_size1 = 4; params.max_sum_size1 <= POW2(k2); params.max_sum_size1 *= 4) {
					handle_result();
				}
			}

			params = best_params;
			params.vector_width = vw;
			for(params.other_wg_size_22a = params.work_block_size; params.other_wg_size_22a == params.work_block_size || params.other_wg_size_22a <= POW2(k3)/params.vector_width; params.other_wg_size_22a *= 2) {
				for(params.other_wg_size_22b = params.work_block_size; params.other_wg_size_22b == params.work_block_size || params.other_wg_size_22b <= POW2(k3)/params.vector_width; params.other_wg_size_22b *= 2) {
					for(params.max_sum_size2a = 4; params.max_sum_size2a <= POW2(k2); params.max_sum_size2a *= 4) {
						for(params.max_sum_size2b = 4; params.max_sum_size2b <= POW2(k2); params.max_sum_size2b *= 4) {
							handle_result();
						}
					}
				}
			}
		}

		if(k2 & 1 || force_radix2) {

			params = best_params;
			params.vector_width = vw;
			for(params.other_wg_size_q2_21 = params.work_block_size; params.other_wg_size_q2_21 == params.work_block_size || params.other_wg_size_q2_21 <= POW2(k3)/params.vector_width; params.other_wg_size_q2_21 *= 2) {
				handle_result();
			}

			params = best_params;
			params.vector_width = vw;
			for(params.other_wg_size_q2_22 = params.work_block_size; params.other_wg_size_q2_22 == params.work_block_size || params.other_wg_size_q2_22 <= POW2(k3)/params.vector_width; params.other_wg_size_q2_22 *= 2) {
				for(params.max_sum_size_q2_2 = 1; params.max_sum_size_q2_2 <= POW2(k2); params.max_sum_size_q2_2 *= 2) {
					handle_result();
				}
			}
		}

		if(force_radix2) {

			params = best_params;
			params.vector_width = vw;
			for(params.other_wg_size_q2_11 = params.work_block_size; params.other_wg_size_q2_11 == params.work_block_size || params.other_wg_size_q2_11 <= POW2(k3)/params.vector_width; params.other_wg_size_q2_11 *= 2) {
				handle_result();
			}

			params = best_params;
			params.vector_width = vw;
			for(params.other_wg_size_q2_12 = params.work_block_size; params.other_wg_size_q2_12 == params.work_block_size || params.other_wg_size_q2_12 <= POW2(k3)/params.vector_width; params.other_wg_size_q2_12 *= 2) {
				for(params.max_sum_size_q2_1 = 1; params.max_sum_size_q2_1 <= POW2(k2); params.max_sum_size_q2_1 *= 2) {
					handle_result();
				}
			}
		}
	}

	if(debug != B4PFM_DEBUG_NONE) {
		printf(
			"(debug) b4pdf2D / run solver: Best time: %f, params = \n", best_time);
		print_opt_params_2d(&best_params);
	}

	
	*opt_params = best_params;
 
	clReleaseCommandQueue(queue);
	clReleaseMemObject(tmp2);
	clReleaseMemObject(tmp1);
	clReleaseContext(context);

	return B4PFM_OK;
}



b4pfm2D b4pfm2D_init_solver(cl_context o_context, b4pfm2D_params *o_opt_params, int k1, int k2, int ldf, int prec, int force_radix2, int debug, int *error) {
	
	cl_int err;
	cl_context context;

	/* Create OpenCL context  */
	if(o_context == 0) {
		if(debug != B4PFM_DEBUG_NONE) printf(
			"(debug) b4pdf2D / pre-init solver: No OpenCL context given. " \
			"Creating an OpenCL context... \n");

		cl_platform_id platform;
		cl_uint num;
		err = clGetPlatformIDs(1, &platform, &num);
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf2D / pre-init solver: Can't query OpenCL platform. " \
				"OpenCL errorcode: %d\n", err);
			*error = B4PFM_OPENCL_ERROR;
			return INV_SOLVER;
		}

		if(num < 1) {
			printf(
				"(error) b4pdf2D / pre-init solver: No OpenCL platforms. \n");
			*error = B4PFM_OPENCL_ERROR;
			return INV_SOLVER;
		}
                       
		cl_device_id device;
        
		err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, 0);
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf2D / pre-init solver: Can't query OpenCL devices. " \
				"OpenCL errorcode: %d\n", err);
			*error = B4PFM_OPENCL_ERROR;
			return INV_SOLVER;
		}

		context = clCreateContext(0, 1, &device, 0, 0, &err);
		if(err != CL_SUCCESS) {
			printf(
				"(error) b4pdf2D / pre-init solver: Can't create OpenCL context. " \
				"OpenCL errorcode: %d\n", err);
			*error = B4PFM_OPENCL_ERROR;
			return INV_SOLVER;
		}
	} else {
		context = o_context;
	}

	b4pfm2D_params n_opt_params;
	b4pfm2D_params *opt_params;
	int err_info;

	if(o_opt_params == NULL) {
		if(debug != B4PFM_DEBUG_NONE) printf(
			"(debug) b4pdf2D / pre-init solver: No opt_params given. Creating default opt_params...\n");		

		err = default_opt_2d_solver(context, &n_opt_params, k1, k2, ldf, prec, debug, &err_info);

		if(err != B4PFM_OK) {
			clReleaseContext(context);
			*error = err;
			return INV_SOLVER;
		}

		opt_params = &n_opt_params;
	} else {
		opt_params = o_opt_params;
	}

	b4pfm2D solver_handle = INV_SOLVER;
	r_b4pfm2D *solver = NULL;

	/* Initialize data-structures */
	if(!initialized_2d) {
		int i;
		for(i = 0; i < MAX_SOLVERS; i++)
			solvers_2d[i] = NULL;	
		initialized_2d = 1;
	}

	int i;
	for(i = 0; i < MAX_SOLVERS; i++) {
		if(solvers_2d[i] == NULL) {
			solver_handle = i+1;
			break;
		}
	}

	if(solver_handle == INV_SOLVER) {
		printf(
			"(error) b4pdf2D / pre-init solver: Too many solvers.\n");
		*error = B4PFM_TOO_MANY_SOLVERS;
		clReleaseContext(context);
		return INV_SOLVER;
	}

	solver = init_2d_solver(context, 0, k1, k2, ldf, prec, *opt_params, force_radix2, debug, &err, &err_info);

#define handle_invalid_wg_size(param, err_code) \
	if(err == err_code && o_opt_params == NULL) { \
		opt_params->param = POW2(k2)/opt_params->vector_width; \
		while(err_info < opt_params->param) \
			opt_params->param /= 2*opt_params->vector_width; \
		if(debug != B4PFM_DEBUG_NONE) printf( \
			"(debug) b4pdf2D / pre-init solver: Default param is too big. " \
			"New value = %d\n", opt_params->param);		 \
		solver = init_2d_solver(context, 0, k1, k2, ldf, prec, *opt_params, force_radix2, debug, &err, &err_info); \
	}

	handle_invalid_wg_size(other_wg_size_11, 	B4PFM_OTHER_WG_2D11_TOO_BIG);
	handle_invalid_wg_size(other_wg_size_12, 	B4PFM_OTHER_WG_2D12_TOO_BIG);
	handle_invalid_wg_size(other_wg_size_21, 	B4PFM_OTHER_WG_2D21_TOO_BIG);
	handle_invalid_wg_size(other_wg_size_22a, 	B4PFM_OTHER_WG_2D22A_TOO_BIG);
	handle_invalid_wg_size(other_wg_size_22b, 	B4PFM_OTHER_WG_2D22B_TOO_BIG);

	handle_invalid_wg_size(other_wg_size_q2_11, 	B4PFM_OTHER_WG_Q2_2D11_TOO_BIG);
	handle_invalid_wg_size(other_wg_size_q2_12, 	B4PFM_OTHER_WG_Q2_2D12_TOO_BIG);
	handle_invalid_wg_size(other_wg_size_q2_21, 	B4PFM_OTHER_WG_Q2_2D21_TOO_BIG);
	handle_invalid_wg_size(other_wg_size_q2_22, 	B4PFM_OTHER_WG_Q2_2D22_TOO_BIG);

	if(err == B4PFM_CR_WG_TOO_BIG && o_opt_params == NULL) {
		opt_params->cr_wg_size = POW2(k2);
		while(err_info < opt_params->cr_wg_size)
			opt_params->cr_wg_size /= 2;

		if(debug != B4PFM_DEBUG_NONE) printf(
			"(debug) b4pdf2D / pre-init solver: Default cr_wg_size is too big. " \
			"New value = %d\n", opt_params->cr_wg_size);		

		solver = init_2d_solver(context, 0, k1, k2, ldf, prec, *opt_params, force_radix2, debug, &err, &err_info);
	}
		
	if(err == B4PFM_OK) {
		*error = B4PFM_OK;
		solvers_2d[solver_handle-1] = solver;
			return solver_handle;
	} else {
		*error = err;
		clReleaseContext(context);
		return INV_SOLVER; 
	}
}

int b4pfm2D_run_solver(b4pfm2D solver, cl_command_queue queue, cl_mem f, cl_mem tmp, int debug) {
	int err_info;
	if(initialized_2d && solver >= 1 && solver <= MAX_SOLVERS && solvers_2d[solver-1] != NULL)
		return run_2d_solver(solvers_2d[solver-1], queue, f, tmp, 1, 0, debug, &err_info);
	else
		return B4PFM_INVALID_SOLVER;
}

int b4pfm2D_load_and_run_solver_float(b4pfm2D solver, float *f, int debug) {
	int err_info;
	if(initialized_2d && solver >= 1 && solver <= MAX_SOLVERS && solvers_2d[solver-1] != NULL)
		return load_and_run_2d_solver(solvers_2d[solver-1], (void*) f, debug, &err_info);
	else
		return B4PFM_INVALID_SOLVER;
}

int b4pfm2D_load_and_run_solver_double(b4pfm2D solver, double *f, int debug) {
	int err_info;
	if(initialized_2d && solver >= 1 && solver <= MAX_SOLVERS && solvers_2d[solver-1] != NULL)
		return load_and_run_2d_solver(solvers_2d[solver-1], (void*) f, debug, &err_info);
	else
		return B4PFM_INVALID_SOLVER;
}

int b4pfm2D_free_solver(b4pfm2D solver) {
	if(initialized_2d && solver >= 1 && solver <= MAX_SOLVERS && solvers_2d[solver-1] != NULL) {
		int err = free_2d_solver(solvers_2d[solver-1]);
		if(err == B4PFM_OK)
			solvers_2d[solver-1] = NULL;
		return err;
	} else {
		return B4PFM_INVALID_SOLVER;
	}
}

int b4pfm2D_default_opts(cl_context context, b4pfm2D_params *opt_params, int k1, int k2, int ldf, int prec, int debug) {
	int err_info;
	return default_opt_2d_solver(context, opt_params, k1, k2, ldf, prec, debug, &err_info);
}

int b4pfm2D_auto_opts(cl_context context, cl_mem tmp1, cl_mem tmp2, b4pfm2D_params *opt_params, int force_radix2, int k1, int k2, int ldf, int prec, int debug) {
	int err_info;
	return auto_optimize_2d_solver(context, tmp1, tmp2, opt_params, force_radix2, k1, k2, ldf, prec, debug, &err_info);
}



