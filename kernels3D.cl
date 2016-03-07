/*
 *  Created on: May 27, 2013
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */



#define PI (acos(-1.0))

// 2^j
#define POW2(j) (1<<(j))

// 4^j
#define POW4(j) POW2(2*(j))

// (-1)^j
#define M1P(j) (1-((j)&0x1)*2)

// sin(pi*(2*j-1)/4)
#define SIN4(j) ((1-(((j)-1)&0x2))*sqrt(1.0/2.0))

#define TRD(i,r) ((i)*POW4(r)-1)

#define TRD2(i,r) ((i)*POW2(r)-1)

#define N2 (POW2(K2)-1)

//
// Double precision settings
//

#if DOUBLE

#if AMD_FP64
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

typedef double _var;
typedef double2 _var2;

#if D == 1
typedef double _varD;
#elif D == 2
typedef double2 _varD;
#elif D == 4
typedef double4 _varD;
#else 
#error "Invalid D."
#endif

#else

//
// Single precision settings
//

typedef float _var;
typedef float2 _var2;

#if D == 1
typedef float _varD;
#elif D == 2
typedef float2 _varD;
#elif D == 4
typedef float4 _varD;
#else
#error "Invalid D."
#endif

#endif


#if D == 1
#define VLOADD(a,b) (b)[a]
#define VSTORED(a,b,c) (c)[b] = (a)
#elif D == 2
#define VLOADD(a,b) vload2(a,b)
#define VSTORED(a,b,c) vstore2(a,b,c)
#elif D == 4
#define VLOADD(a,b) vload4(a,b)
#define VSTORED(a,b,c) vstore4(a,b,c)
#else
#error "Invalid D."
#endif

////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////

#if Q2SOLVER

/*
*/
__kernel 
__attribute__((reqd_work_group_size(OTHER_WG_SIZE_Q2_3D11,1,1))) 
void b2pfm3D_11(const __global _var *o_f, __global _var *o_g, int r) {
	const int local_id = get_local_id(0);
	
	const int g_id0 = get_group_id(0);
	const int g_id1 = get_group_id(1);

	const int line_id = g_id0 / POW2(r-1) + 1;
	const int trid_id = g_id0 % POW2(r-1) + 1;
	
	const int avg_v2d_size = POW2(K2)/get_num_groups(1);
	const int v2d_size = g_id1 == get_num_groups(1)-1 ? N2-g_id1*avg_v2d_size : avg_v2d_size;

	__global _var *g = o_g + (g_id0*N2 + g_id1*avg_v2d_size)*POW2(K3);
	const global _var *f = o_f + g_id1*avg_v2d_size*LDF;	

	int i;
	_var ff_mul = M1P(trid_id-1)*sin(PI*(2*trid_id-1)/(POW2(r)));
	for(i = local_id; i < v2d_size*(POW2(K3)/D); i += OTHER_WG_SIZE_Q2_3D11) {
		_varD ff = 
			+ VLOADD(i, f+TRD2(2*line_id-1,r-1)*N2*LDF)
			+ VLOADD(i, f+TRD2(2*line_id+1,r-1)*N2*LDF);

		VSTORED(ff_mul*ff, i, g);
	}
}

/*
*/
__kernel 
__attribute__((reqd_work_group_size(OTHER_WG_SIZE_Q2_3D12,1,1))) 
void b2pfm3D_12(__global _var *o_f, __global _var *o_g, int r, int prev_count) {
	const int local_id = get_local_id(0);
	
	const int g_id0 = get_group_id(0);
	const int g_id1 = get_group_id(1);

	const int line_count = POW2(K1-r)-1;
	const int part_count = get_num_groups(0) / line_count;
	const int line_id = g_id0 / part_count;
	const int part_id = g_id0 % part_count;
	const int part_size = POW2(r-1) / part_count;

	const int avg_v2d_size = POW2(K2)/get_num_groups(1);
	const int v2d_size = g_id1 == get_num_groups(1)-1 ? N2-g_id1*avg_v2d_size : avg_v2d_size;

	__global _var *g = o_g + ((line_id*POW2(r-1) + part_id*part_size)*N2 + g_id1*avg_v2d_size) * POW2(K3);

	const int prev_size = POW2(r-1) / prev_count;

	if(part_count == 1) {
		__global _var *f = o_f + (TRD2(line_id + 1,r)*N2 + g_id1*avg_v2d_size)*LDF;

		int i;
		for(i = local_id; i < v2d_size*POW2(K3)/D; i += OTHER_WG_SIZE_Q2_3D12) {
			_varD res = 0.0;

			int j;
			for(j = 0; j < min(prev_count,MAX_SUM_SIZE_Q2_1); j++) 
				res += VLOADD(i, g+j*prev_size*N2*POW2(K3));
		
			VSTORED(VLOADD(i, f)+(1.0/POW2(r-1))*res, i, f);
		} 
	} else {
		int i;
		for(i = local_id; i < v2d_size*POW2(K3)/D; i += OTHER_WG_SIZE_Q2_3D12) {
			_varD res = 0.0;

			int j;
			for(j = 0; j < min(prev_count,MAX_SUM_SIZE_Q2_1); j++) 
				res += VLOADD(i, g+j*prev_size*N2*POW2(K3));
		
			VSTORED(res, i, g);
		}
	}
}

__kernel 
__attribute__((reqd_work_group_size(OTHER_WG_SIZE_Q2_3D21,1,1))) 
void b2pfm3D_21(const __global _var *o_f, __global _var *o_g, int r) {
	const int local_id = get_local_id(0);
	
	const int g_id0 = get_group_id(0);
	const int g_id1 = get_group_id(1);

	const int d_id    = g_id0 / POW2(r);
	const int trid_id = g_id0 % POW2(r) + 1;

	const int avg_v2d_size = POW2(K2)/get_num_groups(1);
	const int v2d_size = g_id1 == get_num_groups(1)-1 ? N2-g_id1*avg_v2d_size : avg_v2d_size;

	const __global _var *f = o_f + g_id1*avg_v2d_size*POW2(K3);
	__global _var *g = o_g + (g_id0*N2 + g_id1*avg_v2d_size)*POW2(K3);
	
	int i;
	_var ff2_mul = M1P(trid_id-1)*sin(PI*(2*trid_id-1)/POW2(r+1));
	for(i = local_id; i < v2d_size*POW2(K3)/D; i += OTHER_WG_SIZE_Q2_3D21) {
		_varD ff1 = VLOADD(i, f+TRD2(2*d_id+1,r)*N2*LDF);

		_varD ff2 = 0.0;
		if(d_id > 0)
			ff2 += VLOADD(i, f+TRD2(d_id,r+1)*N2*LDF);
		if(d_id < POW2(K1-r-1)-1)
			ff2 += VLOADD(i, f+TRD2(d_id+1,r+1)*N2*LDF);

		VSTORED(ff1 + ff2_mul*ff2, i, g);
	}
}


/*
 3D solver, back substitution stage, right hand side update, kernel B
 o_f	Right hand side vector get_num_groups(1)*((4^K2-1)*LDF*sizeof(_var) bytes)
 o_g	Workspace (get_num_groups(1)*3*POW2(2*K2-2)*POW2(K3)*sizeof(_var) bytes)
 r		Recursion index, r = 1, 2, ..., K2-1
 get_num_groups(0) = 3*4^(K2-r-1)
 get_num_groups(1) = Number of 2D systems
*/
__kernel 
__attribute__((reqd_work_group_size(OTHER_WG_SIZE_Q2_3D22,1,1))) 
void b2pfm3D_22(__global _var *o_f, __global _var *o_g, int r, int prev_count) {
	const int local_id = get_local_id(0);
	
	const int g_id0 = get_group_id(0);
	const int g_id1 = get_group_id(1);

	const int d_count = POW2(K1-r-1);
	const int part_count = get_num_groups(0) / d_count;
	const int part_size = POW2(r) / part_count;

	const int d_id    = g_id0 / part_count;
	const int part_id = g_id0 % part_count;

	const int avg_v2d_size = POW2(K2)/get_num_groups(1);
	const int v2d_size = g_id1 == get_num_groups(1)-1 ? N2-g_id1*avg_v2d_size : avg_v2d_size;

	__global _var *g = o_g + ((d_id*POW2(r) + part_id*part_size)*N2 + g_id1*avg_v2d_size)* POW2(K3);

	const int prev_size = POW2(r) / prev_count;

	if(part_count == 1) {
		__global _var *f = o_f + (TRD2(2*d_id + 1,r)*N2 + g_id1*avg_v2d_size)*LDF;

		int i;
		for(i = local_id; i < v2d_size*POW2(K3)/D; i += OTHER_WG_SIZE_Q2_3D22) {
			_varD res = 0.0;
			int j;
			for(j = 0; j <  min(prev_count,MAX_SUM_SIZE_Q2_2); j++) 
				res += VLOADD(i, g+(j*prev_size)*N2*POW2(K3));	
		
			VSTORED((1.0/POW2(r))*res, i, f);
		}
	} else {
		int i;
		for(i = local_id; i < v2d_size*POW2(K3)/D; i += OTHER_WG_SIZE_Q2_3D22) {
			_varD res = 0.0;

			int j;
			for(j = 0; j <  min(prev_count,MAX_SUM_SIZE_Q2_2); j++) 
				res += VLOADD(i, g+j*prev_size*N2*POW2(K3));	
		
			VSTORED(res, i, g);
		}		
	}

}

#endif

////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////

#if Q4SOLVER

/*
 3D solver, reduction stage, tridiagonal system formation
 o_f	Right hand side vector get_num_groups(1)*((4^K2-1)*LDF*sizeof(_var) bytes)
 o_g	Workspace (get_num_groups(1)*3*POW2(2*K2-2)*POW2(K3)*sizeof(_var) bytes)
 r		Recursion index, r = 1, 2, ..., K2-1
 get_num_groups(0) = (4^(K2-r)-1) * 3*POW2(2*r-2)
 get_num_groups(1) = Number of 2D systems
*/
__kernel 
__attribute__((reqd_work_group_size(OTHER_WG_SIZE_3D11,1,1))) 
void b4pfm3D_11(const __global _var *o_f, __global _var *o_g, int r) {
	const int local_id = get_local_id(0);
	
	const int g_id0 = get_group_id(0);
	const int g_id1 = get_group_id(1);

	const int line_id = g_id0 / (3*POW2(2*r-2)) + 1;
	const int trid_id = g_id0 % (3*POW2(2*r-2)) + 1;
	
	const int avg_v2d_size = POW2(K2)/get_num_groups(1);
	const int v2d_size = g_id1 == get_num_groups(1)-1 ? N2-g_id1*avg_v2d_size : avg_v2d_size;

	__global _var *g = o_g + (g_id0*N2 + g_id1*avg_v2d_size)*POW2(K3);
	const global _var *f = o_f + g_id1*avg_v2d_size*LDF;
	
	if(trid_id <= POW2(2*r-2)) {
		int i;
		_var ff1_mul = M1P(trid_id-1)*sin(PI*(2*trid_id-1)/POW2(2*r-1));
		for(i = local_id; i < v2d_size*POW2(K3)/D; i += OTHER_WG_SIZE_3D11) {
			_varD ff1 = 
				- VLOADD(i, f+TRD(4*line_id-3,r-1)*N2*LDF)  
				+ VLOADD(i, f+TRD(4*line_id-1,r-1)*N2*LDF) 
				+ VLOADD(i, f+TRD(4*line_id+1,r-1)*N2*LDF) 
				- VLOADD(i, f+TRD(4*line_id+3,r-1)*N2*LDF);

			VSTORED(ff1_mul*ff1, i, g);
		}
	} else {
		int i;
		int j = trid_id - POW2(2*r-2);
		_var ff1_mul = M1P(j-1)*sin(PI*(2*j-1)/POW2(2*r));
		_var ff2_mul = SIN4(j)*sin(PI*(2*j-1)/POW2(2*r));
		for(i = local_id; i < v2d_size*POW2(K3)/D; i += OTHER_WG_SIZE_3D11) {
			_varD ff1 = 
				+ VLOADD(i, f+TRD(4*line_id-2,r-1)*N2*LDF)
				+ VLOADD(i, f+TRD(4*line_id+2,r-1)*N2*LDF);

			_varD ff2 = 
				+ VLOADD(i, f+TRD(4*line_id-3,r-1)*N2*LDF)  
				+ VLOADD(i, f+TRD(4*line_id-1,r-1)*N2*LDF) 
				+ VLOADD(i, f+TRD(4*line_id+1,r-1)*N2*LDF) 
				+ VLOADD(i, f+TRD(4*line_id+3,r-1)*N2*LDF);

			VSTORED(ff1_mul*ff1 + ff2_mul*ff2, i ,g);
		}
	}
}

/*
 3D solver, reduction stage, right hand side update, kernel B
 o_f	Right hand side vector get_num_groups(1)*((4^K2-1)*LDF*sizeof(_var) bytes)
 o_g	Workspace (get_num_groups(1)*3*POW2(2*K2-2)*POW2(K3)*sizeof(_var) bytes)
 r		Recursion index, r = 1, 2, ..., K2-1
 get_num_groups(0) = (4^(K2-r)-1)
 get_num_groups(1) = Number of 2D systems
*/
__kernel 
__attribute__((reqd_work_group_size(OTHER_WG_SIZE_3D12,1,1))) 
void b4pfm3D_12(__global _var *o_f, __global _var *o_g, int r, int prev_count) {
	const int local_id = get_local_id(0);
	
	const int g_id0 = get_group_id(0);
	const int g_id1 = get_group_id(1);

	const int line_count = (POW2(K1-2*r)-1);
	const int part_count = get_num_groups(0) / line_count;
	const int line_id = g_id0 / part_count;
	const int part_id = g_id0 % part_count;
	const int part_size = 3*POW2(2*r-2) / part_count;

	const int avg_v2d_size = POW2(K2)/get_num_groups(1);
	const int v2d_size = g_id1 == get_num_groups(1)-1 ? N2-g_id1*avg_v2d_size : avg_v2d_size;

	__global _var *g = o_g + ((line_id*3*POW2(2*r-2) + part_id*part_size)*N2 + g_id1*avg_v2d_size) * POW2(K3);

	const int prev_size = 3*POW2(2*r-2) / prev_count;

	if(part_count == 1) {
		__global _var *f = o_f + (TRD(line_id + 1,r)*N2 + g_id1*avg_v2d_size)*LDF;

		int i;
		for(i = local_id; i < v2d_size*POW2(K3)/D; i += OTHER_WG_SIZE_3D12) {
			_varD res = 0.0;

			int j;
			for(j = 0; j < min(prev_count,MAX_SUM_SIZE1); j++) 
				res += VLOADD(i, g+j*prev_size*N2*POW2(K3));
		
			VSTORED(VLOADD(i, f)+(1.0/POW2(2*r-1))*res, i, f);
		} 
	} else {
		int i;
		for(i = local_id; i < v2d_size*POW2(K3)/D; i += OTHER_WG_SIZE_3D12) {
			_varD res = 0.0;

			int j;
			for(j = 0; j < min(prev_count,MAX_SUM_SIZE1); j++) 
				res += VLOADD(i, g+j*prev_size*N2*POW2(K3));
		
			VSTORED(res, i, g);
		}
	}
}

/*
 3D solver, back substitution stage, tridiagonal system formation
 o_f	Right hand side vector ((4^K2-1)*LDF*sizeof(_var) bytes)
 o_g	Workspace (get_num_groups(1)*3*POW2(2*K2-2)*POW2(K3)*sizeof(_var) bytes)
 r		Recursion index, r = K2-1, ..., 1, 0
 get_num_groups(0) = (4^(K2-r-1)) * 3*POW2(2*r-2)
 get_num_groups(1) = Number of 2D systems
*/
__kernel 
__attribute__((reqd_work_group_size(OTHER_WG_SIZE_3D21,1,1))) 
void b4pfm3D_21(const __global _var *o_f, __global _var *o_g, int r) {
	const int local_id = get_local_id(0);
	
	const int g_id0 = get_group_id(0);
	const int g_id1 = get_group_id(1);

	const int d_id    = g_id0 / (3*POW2(2*r));
	const int trid_id = g_id0 % (3*POW2(2*r)) + 1;

	const int avg_v2d_size = POW2(K2)/get_num_groups(1);
	const int v2d_size = g_id1 == get_num_groups(1)-1 ? N2-g_id1*avg_v2d_size : avg_v2d_size;

	const __global _var *f = o_f + g_id1*avg_v2d_size*POW2(K3);
	__global _var *g = o_g + (g_id0*N2 + g_id1*avg_v2d_size)*POW2(K3);
	
	if(trid_id <= POW2(2*r)) {
		int i;
		_var ff1_mul = M1P(trid_id-1);
		_var ff2_mul = sin(PI*(2*trid_id-1)/POW2(2*r+1));
		for(i = local_id; i < v2d_size*POW2(K3)/D; i += OTHER_WG_SIZE_3D21) {
			_varD ff1 = 
				+ VLOADD(i, f+TRD(4*d_id+1,r)*N2*LDF) 
				- VLOADD(i, f+TRD(4*d_id+3,r)*N2*LDF);

			_varD ff2 = 0.0;
			if(d_id > 0)
				ff2 += VLOADD(i, f+TRD(d_id,r+1)*N2*LDF);
			if(d_id < POW2(K1-2*r-2)-1)
				ff2 -= VLOADD(i, f+TRD(d_id+1,r+1)*N2*LDF);

			VSTORED(ff1_mul*ff1 + ff2_mul*ff2, i, g);
		}
	} else {
		int i;
		int j = trid_id - POW2(2*r);
		_var ff1_mul = M1P(j-1);
		_var ff2_mul = SIN4(j);
		_var ff3_mul = sin(PI*(2*j-1)/POW2(2*r+2));
		for(i = local_id; i < v2d_size*POW2(K3)/D; i += OTHER_WG_SIZE_3D21) {
			_varD ff1 =  VLOADD(i, f+TRD(4*d_id+2,r)*N2*LDF);

			_varD ff2 = 
				+ VLOADD(i, f+TRD(4*d_id+1,r)*N2*LDF) 
				+ VLOADD(i, f+TRD(4*d_id+3,r)*N2*LDF);

			_varD ff3 = 0.0;
			if(d_id > 0)
				ff3 += VLOADD(i, f+TRD(d_id,r+1)*N2*LDF);
			if(d_id < POW2(K1-2*r-2)-1)
				ff3 += VLOADD(i, f+TRD(d_id+1,r+1)*N2*LDF);

			VSTORED(ff1_mul*ff1 + ff2_mul*ff2 + ff3_mul*ff3, i, g);
		}
	}
}

/*
 3D solver, back substitution stage, right hand side update, kernel A
 o_f	Right hand side vector get_num_groups(1)*((4^K2-1)*LDF*sizeof(_var) bytes)
 o_g	Workspace (get_num_groups(1)*3*POW2(2*K2-2)*POW2(K3)*sizeof(_var) bytes)
 r		Recursion index, r = 1, 2, ..., K2-1
 get_num_groups(0) = 3*4^(K2-r-1)
 get_num_groups(1) = Number of 2D systems
*/
__kernel 
__attribute__((reqd_work_group_size(OTHER_WG_SIZE_3D22A,1,1))) 
void b4pfm3D_22a(__global _var *o_g, int r) {
	const int local_id = get_local_id(0);
	
	const int g_id0 = get_group_id(0);
	const int g_id1 = get_group_id(1);

	const int d_count = POW2(K1-2*r-2);
	const int part_count = get_num_groups(0) / d_count;
	const int d_id = g_id0 / part_count;
	const int part_id = g_id0 % part_count;
	const int part_size = 3*POW2(2*r) / part_count;

	const int avg_v2d_size = POW2(K2)/get_num_groups(1);
	const int v2d_size = g_id1 == get_num_groups(1)-1 ? N2-g_id1*avg_v2d_size : avg_v2d_size;

	__global _var *g = o_g + ((d_id*3*POW2(2*r) + part_id*part_size)*N2 + g_id1*avg_v2d_size)* POW2(K3);

	const int j_start = part_id*part_size + 1;

	int i;
	for(i = local_id; i < v2d_size*POW2(K3)/D; i += OTHER_WG_SIZE_3D22A) {
		_varD res1 = 0.0;
		_varD res2 = 0.0;
		_varD res3 = 0.0;

		int j = j_start;
		int l = 0;
		for(; l < part_size && j <= POW2(2*r); l++, j++) {
			_varD tmp = M1P(j-1)*VLOADD(i, g+l*N2*POW2(K3));
			res1 += tmp;
			res3 -= tmp;
		}

		for(j -= POW2(2*r); l < part_size && j <= POW2(2*r+1); l++, j++) {
			_varD tmp1 = VLOADD(i, g+l*N2*POW2(K3));
			_varD tmp2 = SIN4(j)*tmp1;
			res1 += tmp2;
			res2 += M1P(j-1)*tmp1;
			res3 += tmp2;
		}
		
		VSTORED(res1, i, g);
		VSTORED(res2, i, g+N2*POW2(K3));
		VSTORED(res3, i, g+2*N2*POW2(K3));
	}
}

/*
 3D solver, back substitution stage, right hand side update, kernel B
 o_f	Right hand side vector get_num_groups(1)*((4^K2-1)*LDF*sizeof(_var) bytes)
 o_g	Workspace (get_num_groups(1)*3*POW2(2*K2-2)*POW2(K3)*sizeof(_var) bytes)
 r		Recursion index, r = 1, 2, ..., K2-1
 get_num_groups(0) = 3*4^(K2-r-1)
 get_num_groups(1) = Number of 2D systems
*/
__kernel 
__attribute__((reqd_work_group_size(OTHER_WG_SIZE_3D22B,1,1))) 
void b4pfm3D_22b(__global _var *o_f, __global _var *o_g, int r, int prev_count) {
	const int local_id = get_local_id(0);
	
	const int g_id0 = get_group_id(0);
	const int g_id1 = get_group_id(1);

	const int d_count = POW2(K1-2*r-2);
	const int part_count = get_num_groups(0) / (3*d_count);
	const int part_size = 3*POW2(2*r) / part_count;

	const int d_id    = (g_id0/3) / part_count;
	const int part_id = (g_id0/3) % part_count;
	const int line_id = g_id0 % 3;

	const int avg_v2d_size = POW2(K2)/get_num_groups(1);
	const int v2d_size = g_id1 == get_num_groups(1)-1 ? N2-g_id1*avg_v2d_size : avg_v2d_size;

	__global _var *g = o_g + ((d_id*3*POW2(2*r) + part_id*part_size)*N2 + g_id1*avg_v2d_size)* POW2(K3);

	const int prev_size = 3*POW2(2*r) / prev_count;

	if(part_count == 1) {
		__global _var *f = o_f + (TRD(4*d_id+line_id+1,r)*N2 + g_id1*avg_v2d_size)*LDF;

		int i;
		for(i = local_id; i < v2d_size*POW2(K3)/D; i += OTHER_WG_SIZE_3D22B) {
			_varD res = 0.0;
			int j;
			for(j = 0; j <  min(prev_count,MAX_SUM_SIZE2B); j++) 
				res += VLOADD(i, g+(j*prev_size+line_id)*N2*POW2(K3));	
		
			VSTORED((1.0/POW2(2*r+1))*res, i, f);
		}
	} else {
		int i;
		for(i = local_id; i < v2d_size*POW2(K3)/D; i += OTHER_WG_SIZE_3D22B) {
			_varD res = 0.0;

			int j;
			for(j = 0; j <  min(prev_count,MAX_SUM_SIZE2B); j++) 
				res += VLOADD(i, g+(j*prev_size+line_id)*N2*POW2(K3));	
		
			VSTORED(res, i, g+line_id*N2*POW2(K3));
		}		
	}

}

#endif

