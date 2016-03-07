/*
 *  Created on: May 27, 2013
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */


#define PI (acos(-1.0))

// 2^j
#define POW2(j) (1<<(j))

// (-1)^j
#define M1P(j) (1-((j)&0x1)*2)

// sin(pi*(2*j-1)/4)
#define SIN4(j) ((1-(((j)-1)&0x2))*sqrt(1.0/2.0))

#define TRD(i,r) ((i)*POW2(2*(r))-1)

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

#if HILODOUBLE
typedef struct {
	__local uint *lo;
	__local uint *hi;
} _var_local;
#define LLOAD(idx, buff) as_double((((ulong)(buff).hi[idx]) << 32) | ((ulong)(buff).lo[idx]))
void LSTORE(_var data, int idx, _var_local buff) { 
	(buff).lo[idx] = as_uint2(data).x; 
	(buff).hi[idx] = as_uint2(data).y; 
}
void LSTORE2(_var2 data, int idx, _var_local buff) { 
	vstore2((uint2)((uint)as_uint2(data.x).x, (uint)as_uint2(data.y).x), idx, (buff).lo);
	vstore2((uint2)((uint)as_uint2(data.x).y, (uint)as_uint2(data.y).y), idx, (buff).hi);
}
#if USE_LOCAL_COEF
#define CLOAD(idx, buff) LLOAD(idx, buff)
#define CSTORE(data, idx, buff) LSTORE(data, idx, buff)
#else // !USE_LOCAL_COEF
#define CSTORE(data, idx, buff) ((buff)[(idx)] = (data))
#define CLOAD(idx, buff) ((buff)[(idx)])
#endif // END !USE_LOCAL_COEF
#else // !HILODOUBLE
#define LLOAD(idx, buff) ((buff)[(idx)])
#define LSTORE(data, idx, buff) ((buff)[(idx)] = (data))
#define CLOAD(idx, buff) ((buff)[(idx)])
#define CSTORE(data, idx, buff) ((buff)[(idx)] = (data))
#define LSTORE2(data, idx, buff) vstore2(data, idx, buff)
#endif // END !HILODOUBLE



// r{1,2} = r_{1,2} during the reduction,
// r{1,2} = r_{1,2}+1 during back substitution
__kernel __attribute__((reqd_work_group_size(CR_WG_SIZE,1,1))) 
void b4pfm_CR(__global _var* o_f, int r1, int r2, int radix2_2d, int radix2_3d) { 

#if HILODOUBLE
	__local uint work_lo[CR_LOCAL_MEM_SIZE];
	__local uint work_hi[CR_LOCAL_MEM_SIZE];
	_var_local work;
	work.hi = work_hi;
	work.lo = work_lo;
#else
	__local _var work[CR_LOCAL_MEM_SIZE];
#endif

#if USE_LOCAL_COEF

#if HILODOUBLE
	__local uint dd_lo[K3];
	__local uint dd_hi[K3];
	__local uint tt_lo[K3];
	__local uint tt_hi[K3];
	_var_local dd, tt;
	dd.hi = dd_hi;
	dd.lo = dd_lo;
	tt.hi = tt_hi;
	tt.lo = tt_lo;
#else
	__local _var dd[K3];
	__local _var tt[K3];
#endif

#elif FORCE_PRIV_COEF
	__private _var dd[K3];
	__private _var tt[K3];
#else
	_var dd[K3];
	_var tt[K3];
#endif

	const size_t local_id = get_local_id(0);
	const size_t half_local_id = local_id - CR_WG_SIZE/2;

	const size_t g_id2d = get_group_id(0);
	const size_t trid_id_2d = radix2_2d ? g_id2d % POW2(r2-1) + 1 : g_id2d % (3*POW2(2*r2-2)) + 1;

#if K1 > 0
	const size_t g_id3d = get_group_id(1);
	const size_t trid_id_3d = radix2_3d ? g_id3d % POW2(r1-1) + 1 : g_id3d % (3*POW2(2*r1-2)) + 1;
#else
	const size_t g_id3d = 0;
#endif

	// Generate coefficient matrices


#if USE_LOCAL_COEF
	if(local_id == 0) {
#endif
#if K1 == 0

		if(radix2_2d) {
			CSTORE(4.0-2.0*cos(PI*(2*trid_id_2d-1)/POW2(r2)), 0, dd);
			CSTORE(-1.0, 0, tt);
		} else {
			if(trid_id_2d <= POW2(2*r2-2))
				CSTORE(4.0-2.0*cos(PI*(2*trid_id_2d-1)/POW2(2*r2-1)), 0, dd);
			else 
				CSTORE(4.0-2.0*cos(PI*(2*(trid_id_2d-POW2(2*r2-2))-1)/POW2(2*r2)), 0, dd);
		}
		CSTORE(-1.0, 0, tt);
#else
		if(radix2_3d) {
			CSTORE(6.0-2.0*cos(PI*(2*trid_id_3d-1)/POW2(r1)), 0, dd);
		} else {
			if(trid_id_3d <= POW2(2*r1-2)) 
				CSTORE(6.0-2.0*cos(PI*(2*trid_id_3d-1)/POW2(2*r1-1)), 0, dd);
			else
				CSTORE(6.0-2.0*cos(PI*(2*(trid_id_3d-POW2(2*r1-2))-1)/POW2(2*r1)), 0, dd);
		}

		if(radix2_2d) {
			CTORE(LLOAD(0, dd) - 2.0*cos(PI*(2*trid_id_2d-1)/POW2(r2)), 0, dd);
		} else {
			if(trid_id_2d <= POW2(2*r2-2))
				CTORE(LLOAD(0, dd) - 2.0*cos(PI*(2*trid_id_2d-1)/POW2(2*r2-1)), 0, dd);
			else
				CTORE(LLOAD(0, dd) - 2.0*cos(PI*(2*(trid_id_2d-POW2(2*r2-2))-1)/POW2(2*r2)), 0, dd);
		}

		CSTORE(-1.0, 0, tt);
#endif
#if USE_LOCAL_COEF
	}
	barrier(CLK_LOCAL_MEM_FENCE);
#endif

	__global _var *f;
	if(radix2_2d) 
		f = o_f + (g_id3d * POW2(K2-1) + g_id2d)*POW2(K3);
	else
		f = o_f + (g_id3d * 3*POW2(K2-2) + g_id2d)*POW2(K3);

	int r = 1; 
	int i;

	///////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////

#if GLOBAL_SOLVER

	// First part of the reduction stage. Used only if the 
	// problem is too big to be handled in local memory.
	for(; CR_LOCAL_MEM_SIZE < POW2(K3-r+1); r++) {
		_var d = CLOAD(r-1, dd);
		_var t = CLOAD(r-1, tt);

#if USE_LOCAL_COEF
		if(local_id == 0) {
			CSTORE(d-(2.0*t*t)/d, r, dd);
			CSTORE(-(t*t)/d, r, tt);
			// Syncronization occurs later
		}
#else
		CSTORE(d-(2.0*t*t)/d, r, dd);
		CSTORE(-(t*t)/d, r, tt);
#endif
		
		__global _var* first_part = f + POW2(K3) - (POW2(r-1)+1)*CR_WG_SIZE;
		__global _var* second_part = f + POW2(K3) - CR_WG_SIZE;

		_var mul = (t/d);

		// Spits up the remaining problem into smaller pieces and
		// iterates through them from bottom to up.
		const int part_count = POW2(K3-r)/CR_WG_SIZE;
		for(i = 0; i < part_count; i++) {

			_var2 data;
			_var lower;
			
			// Load
			if(local_id < CR_WG_SIZE/2)
				data = vload2(local_id, first_part);
			else
				data = vload2(half_local_id, second_part);
			
			// Reuse data from previous part
			if(local_id == CR_WG_SIZE - 1)
				lower = LLOAD(0, work);
			
			barrier(CLK_LOCAL_MEM_FENCE);
			LSTORE(data.x, local_id, work);
			barrier(CLK_LOCAL_MEM_FENCE);
			
			if(local_id < CR_WG_SIZE - 1)
				lower = LLOAD(local_id+1, work);

			data.y -= mul*(data.x+lower);

			// Save
			first_part[local_id] = data.x;
			second_part[local_id] = data.y;
			barrier(CLK_GLOBAL_MEM_FENCE);
			
			first_part -= POW2(r)*CR_WG_SIZE;
			second_part -= POW2(r)*CR_WG_SIZE;
		}
	}

#endif /* GLOBAL_SOLVER */

	///////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////

#if MIDDLE_SOLVER

	// Load right hand side vector into the local memory
	for(i = 0; i < POW2(K3-r+1)/CR_WG_SIZE; i += 2) {
		_var2 data;
		if(local_id < CR_WG_SIZE/2)
			data = vload2(local_id, f + ((i+1)*POW2(r-1)-1)*CR_WG_SIZE);
		else
			data = vload2(half_local_id, f + ((i+2)*POW2(r-1)-1)*CR_WG_SIZE);

		LSTORE(data.x, i*CR_WG_SIZE + local_id, work);
		LSTORE(data.y, (i+1)*CR_WG_SIZE + local_id, work);
	}

#elif GLOBAL_SOLVER

	_var2 data;
	if(local_id < CR_WG_SIZE/2)
		data = vload2(local_id, f+(POW2(r-1)-1)*CR_WG_SIZE);
#if POW2(K3) > 2*CR_WG_SIZE
	else
		data = vload2(half_local_id, f+(2*POW2(r-1)-1)*CR_WG_SIZE);
#endif

	if(local_id < POW2(K3-r)) {
		LSTORE(data.x, local_id, work);
		LSTORE(data.y, POW2(K3-r) + local_id, work);
	}

#else

	if(local_id < POW2(K3-r)) {
		_var2 data = vload2(local_id, f);
		LSTORE(data.x, local_id, work);
		LSTORE(data.y, POW2(K3-r) + local_id, work);
	}

#endif

	barrier(CLK_LOCAL_MEM_FENCE);

	///////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////

	const int loc_k = K3-r+1;
	int loc_r = 1;

#if MIDDLE_SOLVER

	// Middle part of the reduction stage. 
	for(; CR_WG_SIZE < POW2(K3-r); r++, loc_r++) {
		_var d = CLOAD(r-1, dd);
		_var t = CLOAD(r-1, tt);

#if USE_LOCAL_COEF
		if(local_id == 0) {
			CSTORE(d-(2.0*t*t)/d, r, dd);
			CSTORE(-(t*t)/d, r, tt);
			// Syncronization occurs later
		}
#else
		CSTORE(d-(2.0*t*t)/d, r, dd);
		CSTORE(-(t*t)/d, r, tt);
#endif
		
		int first_part =  (POW2(loc_r-1)-1)*CR_WG_SIZE;
		int second_part = (POW2(loc_r)-1)*CR_WG_SIZE;
		int third_part =  (3*POW2(loc_r-1)-1)*CR_WG_SIZE;
		int fourth_part = (POW2(loc_r+1)-1)*CR_WG_SIZE;	
		
		_var mul = (t/d);

		int i;
		// Spits up the remaining problem into smaller pieces and
		// iterates through them
		const int part_count = POW2(K3-r-1)/CR_WG_SIZE;
		for(i = 0; i < part_count; i++) {

			_var upper1, middle1, lower1;
			_var upper2, middle2, lower2;
			
			upper1 = LLOAD(first_part+local_id, work);
			middle1 = LLOAD(second_part+local_id, work);

			upper2 = LLOAD(third_part+local_id, work);
			middle2 = LLOAD(fourth_part+local_id, work);

			int lower1_addr = local_id < CR_WG_SIZE-1 ? first_part : third_part;

			lower1 = LLOAD(lower1_addr+(local_id+1)%CR_WG_SIZE, work);
			
			if(i != part_count-1) { 
				int lower2_addr = local_id < CR_WG_SIZE-1 ? third_part : first_part+POW2(loc_r+1)*CR_WG_SIZE;
				lower2 = LLOAD(lower2_addr+(local_id+1)%CR_WG_SIZE, work);
			} else if(local_id < CR_WG_SIZE-1) {
				lower2 = LLOAD(third_part+(local_id+1)%CR_WG_SIZE, work);
			}

			middle1 -= mul*(lower1+upper1);
			middle2 -= mul*(lower2+upper2);

			barrier(CLK_LOCAL_MEM_FENCE);

			int locat = local_id & 0x01 ? fourth_part : second_part;

			LSTORE(middle1, locat+(local_id>>1), work);
			LSTORE(middle2, locat+(CR_WG_SIZE/2) + (local_id>>1), work);

			barrier(CLK_LOCAL_MEM_FENCE);
			
			int add = POW2(loc_r+1)*CR_WG_SIZE;

			first_part  += add;
			second_part += add;
			third_part  += add;
			fourth_part += add;
		}
	}
	
#endif /* MIDDLE_SOLVER */

	///////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////

	// Last part of the reduction stage.

	int first_part = CR_LOCAL_MEM_SIZE - (POW2(loc_r-1)+1)*POW2(loc_k-loc_r);
	int second_part = CR_LOCAL_MEM_SIZE - POW2(loc_k-loc_r);

	for(; r <= K3-1; r++, loc_r++) {

		_var d = CLOAD(r-1, dd);
		_var t = CLOAD(r-1, tt);

#if USE_LOCAL_COEF
		if(local_id == 0) {
			CSTORE(d-(2.0*t*t)/d, r, dd);
			CSTORE(-(t*t)/d, r, tt);
			// Syncronization occurs later
		}
#else
		CSTORE(d-(2.0*t*t)/d, r, dd);
		CSTORE(-(t*t)/d, r, tt);
#endif

		_var upper, middle, lower;

		if(local_id < POW2(K3-r)-1) {		
			upper = LLOAD(first_part+local_id, work);
			middle = LLOAD(second_part+local_id, work);
			lower = LLOAD(first_part+local_id + 1, work);

			middle -= (t/d)*(lower+upper);
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		if(local_id < POW2(K3-r)-1)
			LSTORE(middle, second_part + (local_id & 0x01)*POW2(K3-r-1) + (local_id>>1), work);

		barrier(CLK_LOCAL_MEM_FENCE);
	
		first_part =  CR_LOCAL_MEM_SIZE - POW2(loc_k-loc_r);
		second_part = CR_LOCAL_MEM_SIZE - POW2(loc_k-loc_r-1);
	}

	///////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////

	// First part of the back substitution stage. 
	if(local_id == 0)
		LSTORE(LLOAD(CR_LOCAL_MEM_SIZE - 2 + local_id, work) / CLOAD(K3-1, dd), CR_LOCAL_MEM_SIZE - 2 + local_id, work);
	barrier(CLK_LOCAL_MEM_FENCE);

	for(r = K3-2, loc_r = loc_k-2; POW2(K3-r-1) < CR_WG_SIZE && loc_r >= 1; r--, loc_r--) {

		first_part = CR_LOCAL_MEM_SIZE - POW2(loc_k-loc_r);
		second_part = CR_LOCAL_MEM_SIZE - POW2(loc_k-loc_r-1);

		_var upper, middle, lower;

		if(local_id < POW2(K3-r-1)) {
			upper = local_id == 0 ? 0.0 : LLOAD(second_part+local_id-1, work);
			middle = LLOAD(first_part+local_id, work);
			lower = local_id == POW2(K3-r-1) - 1 ? 0.0 : LLOAD(second_part+local_id, work);

			middle = (middle - CLOAD(r, tt)*(upper+lower))/CLOAD(r, dd);
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		if(local_id < POW2(K3-r-1)) 
			LSTORE2((_var2)(middle,lower), first_part/2 + local_id, work);

		barrier(CLK_LOCAL_MEM_FENCE);
		
	}


	first_part = CR_LOCAL_MEM_SIZE - (POW2(loc_r)+1)*POW2(loc_k-loc_r-1);
	second_part = CR_LOCAL_MEM_SIZE - POW2(loc_k-loc_r-1);

	_var upper, middle, lower;

	if(local_id < POW2(K3-r-1)) {
		upper = local_id == 0 ? 0.0 : LLOAD(second_part+local_id-1, work);
		middle = LLOAD(first_part+local_id, work);
		lower = local_id == POW2(K3-r-1) - 1 ? 0.0 : LLOAD(second_part+local_id, work);

		middle = (middle - CLOAD(r, tt)*(upper+lower))/CLOAD(r, dd);
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if(local_id < POW2(K3-r-2)) 
		LSTORE2((_var2)(middle,lower), first_part/2 + local_id, work);
	else if(local_id < POW2(K3-r-1)) 
		LSTORE2((_var2)(middle,lower), second_part/2 + local_id-POW2(K3-r-2), work);

	barrier(CLK_LOCAL_MEM_FENCE);
	
	r--;
	loc_r--;

	///////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////

#if MIDDLE_SOLVER

	// Middle part of the back substitution stage. 
	for(; loc_r >= 0 ; r--, loc_r--) {
		_var d = CLOAD(r, dd);
		_var t = CLOAD(r, tt);

		int first_part  = CR_LOCAL_MEM_SIZE - (3*POW2(loc_r)+1)*CR_WG_SIZE;
		int second_part = CR_LOCAL_MEM_SIZE - (POW2(loc_r+1)+1)*CR_WG_SIZE;
		int third_part  = CR_LOCAL_MEM_SIZE - (POW2(loc_r)+1)*CR_WG_SIZE;
		int fourth_part = CR_LOCAL_MEM_SIZE - CR_WG_SIZE;
			
		
		// Spits up the remaining problem into smaller pieces and
		// iterates through them from up to bottom.
		for(i = POW2(K3-r-2)/CR_WG_SIZE-1; 0 <= i; i--) {

			_var upper1, lower1;
			_var upper2, lower2;

			int upper1_addr = local_id > 0 ? second_part : fourth_part - POW2(loc_r+2)*CR_WG_SIZE;
			int upper2_addr = local_id > 0 ? fourth_part : second_part;

			if(i == 0) 
				upper1 = local_id == 0 ? 0.0 : LLOAD(upper1_addr+local_id-1, work);
			else 
				upper1 = LLOAD(upper1_addr+(local_id-1)%CR_WG_SIZE, work);

			upper2 = LLOAD(upper2_addr+(local_id-1)%CR_WG_SIZE, work);

			lower1 = LLOAD(second_part+local_id, work);

			if(i*CR_WG_SIZE+local_id < POW2(K3-r-2))
				lower2 = local_id == POW2(K3-r-1) - 1 ? 0.0 : LLOAD(fourth_part+local_id, work);
			else 
				lower2 = LLOAD(fourth_part+local_id, work);

			_var middle1 = (LLOAD(first_part+local_id, work) - t*(upper1+lower1))/d;
			_var middle2 = (LLOAD(third_part+local_id, work) - t*(upper2+lower2))/d;

			barrier(CLK_LOCAL_MEM_FENCE);

			if(local_id < CR_WG_SIZE/2) {
				LSTORE2((_var2)(middle1,lower1), first_part/2 + local_id, work);
				LSTORE2((_var2)(middle2,lower2), third_part/2 + local_id, work);
			} else {
				LSTORE2((_var2)(middle1,lower1), second_part/2 + half_local_id, work);
				LSTORE2((_var2)(middle2,lower2), fourth_part/2 + half_local_id, work);
			}

			barrier(CLK_LOCAL_MEM_FENCE);

			int add = POW2(loc_r+2)*CR_WG_SIZE;
			first_part -= add;
			second_part -= add;
			third_part -= add;
			fourth_part -= add;
		}
	}
#endif /* MIDDLE_SOLVER */

	///////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////

#if MIDDLE_SOLVER

	// Load right hand side vector into the global memory
	for(i = 0; i < POW2(K3-r-1)/CR_WG_SIZE; i++) 
		f[((i+1)*POW2(r+1)-1)*CR_WG_SIZE+local_id] = LLOAD(i*CR_WG_SIZE+local_id, work);

#elif GLOBAL_SOLVER

	f[(POW2(r+1)-1)*CR_WG_SIZE+local_id] = LLOAD(local_id, work);
	if(POW2(K3-r-1) > CR_WG_SIZE)
		f[(2*POW2(r+1)-1)*CR_WG_SIZE+local_id] = LLOAD(CR_WG_SIZE+local_id, work);

#else

	if(local_id < POW2(K3-r-2)) {
		f[local_id] = LLOAD(local_id, work);
		f[POW2(K3-r-2) + local_id] = LLOAD(POW2(K3-r-2) + local_id, work);
	}

#endif

	barrier(CLK_LOCAL_MEM_FENCE);

	///////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////

#if GLOBAL_SOLVER

	// Last part of the back substitutio stage. Used only if the remaining 
	// problem is too big to be handled locally.
	for(; r >= 0; r--) {
		_var d = CLOAD(r, dd);
		_var t = CLOAD(r, tt);
		
		__global _var* first_part = f + (POW2(r)-1)*CR_WG_SIZE;
		__global _var* second_part = f + (POW2(r+1)-1)*CR_WG_SIZE;
		
		// Spits up the remaining problem into smaller pieces and
		// iterates through them from up to bottom.
		const int part_count = POW2(K3-r-1)/CR_WG_SIZE;
		for(i = 0; i < part_count; i++) {
			
			_var upper, lower;
			
			// Reuse data from last part
			if(local_id == 0)
				upper = LLOAD(CR_WG_SIZE-1, work);
			
			// Load
			barrier(CLK_LOCAL_MEM_FENCE);
			lower = second_part[local_id];
			LSTORE(lower, local_id, work); 
			barrier(CLK_LOCAL_MEM_FENCE);
			if(i == part_count-1 && local_id == CR_WG_SIZE-1)
				lower = 0.0;
			
			if(i == 0)
				upper = (local_id == 0 ? 0.0 : LLOAD(local_id-1, work));
			else if(local_id != 0)
				upper = LLOAD(local_id-1, work);
			
			// Calc
			_var middle = (first_part[local_id] - t*(upper+lower))/d;
			
			// Save
			if(local_id < CR_WG_SIZE/2)
				vstore2((_var2)(middle,lower), local_id, first_part);
			else
				vstore2((_var2)(middle,lower), half_local_id, second_part);
	
			barrier(CLK_GLOBAL_MEM_FENCE);
			
			first_part += POW2(r+1)*CR_WG_SIZE;
			second_part += POW2(r+1)*CR_WG_SIZE;
		}
	}
#endif /* GLOBAL_SOLVER */
	
}

//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////

#if Q2SOLVER

/*
 2D solver, reduction stage, tridiagonal system formation
*/
__kernel 
__attribute__((reqd_work_group_size(OTHER_WG_SIZE_Q2_2D11,1,1))) 
void b2pfm_11(const __global _var *o_f, __global _var *o_g, int r) {
	const int local_id = get_local_id(0);
	
	const int g_id2d = get_group_id(0);
	const int g_id3d = get_group_id(1);

	const int line_id = g_id2d / POW2(r-1) + 1;
	const int trid_id = g_id2d % POW2(r-1) + 1;

	__global _var *g = o_g + (g_id3d * POW2(K2-1) + g_id2d)*POW2(K3);
	const __global _var *f = o_f + g_id3d*N2*LDF;
	
	int i;
	_var ff_mul = M1P(trid_id-1)*sinpi(1.0*(2*trid_id-1)/(POW2(r)));
	for(i = local_id; i < (POW2(K3)/D); i += OTHER_WG_SIZE_Q2_2D11) {
		_varD ff = 
			+ VLOADD(i, f+TRD2(2*line_id-1,r-1)*LDF)
			+ VLOADD(i, f+TRD2(2*line_id+1,r-1)*LDF);

		VSTORED(ff_mul*ff, i, g);
	}
}

/*
*/
__kernel 
__attribute__((reqd_work_group_size(OTHER_WG_SIZE_Q2_2D12,1,1))) 
void b2pfm_12(__global _var *o_f, __global _var *o_g, int r, int prev_count) {
	const int local_id = get_local_id(0);
	
	const int g_id2d = get_group_id(0);
	const int g_id3d = get_group_id(1);

	const int line_count = POW2(K2-r)-1;
	const int part_count = get_num_groups(0) / line_count;
	const int line_id = g_id2d / part_count;
	const int part_id = g_id2d % part_count;
	const int part_size = POW2(r-1) / part_count;

	__global _var *g = o_g + (g_id3d * POW2(K2-1) + line_id*POW2(r-1) + part_id*part_size) * POW2(K3);

	const int prev_size = POW2(r-1) / prev_count;

	if(part_count == 1) {
		__global _var *f = o_f + (g_id3d*N2 + TRD2(line_id + 1,r))*LDF;

		int i;
		for(i = local_id; i < POW2(K3)/D; i += OTHER_WG_SIZE_Q2_2D12) {
			_varD res = 0.0;

			int j;
			for(j = 0; j < min(prev_count,MAX_SUM_SIZE_Q2_1); j++) 
				res += VLOADD(i, g+j*prev_size*POW2(K3));
		
			VSTORED(VLOADD(i, f)+(1.0/POW2(r-1))*res, i, f);
		} 
	} else {
		int i;
		for(i = local_id; i < POW2(K3)/D; i += OTHER_WG_SIZE_Q2_2D12) {
			_varD res = 0.0;

			int j;
			for(j = 0; j < min(prev_count,MAX_SUM_SIZE_Q2_1); j++) 
				res += VLOADD(i, g+j*prev_size*POW2(K3));
		
			VSTORED(res, i, g);
		}
	}
}

/*
*/
__kernel 
__attribute__((reqd_work_group_size(OTHER_WG_SIZE_Q2_2D21,1,1))) 
void b2pfm_21(const __global _var *o_f, __global _var *o_g, int r) {
	const int local_id = get_local_id(0);
	
	const int g_id2d = get_group_id(0);
	const int g_id3d = get_group_id(1);

	const int d_id    = g_id2d / POW2(r);
	const int trid_id = g_id2d % POW2(r) + 1;

	__global _var *g = o_g + (g_id3d * POW2(K2-1) + g_id2d)*POW2(K3);
	const __global _var *f = o_f + g_id3d*N2*LDF;
	
	int i;
	_var ff2_mul = M1P(trid_id-1)*sinpi(1.0*(2*trid_id-1)/POW2(r+1));
	for(i = local_id; i < (POW2(K3)/D); i += OTHER_WG_SIZE_Q2_2D21) {

		_varD ff1 = VLOADD(i, f+TRD2(2*d_id+1,r)*LDF);

		_varD ff2 = 0.0;
		if(d_id > 0)
			ff2 += VLOADD(i, f+TRD2(d_id,r+1)*LDF);
		if(d_id < POW2(K2-r-1)-1)
			ff2 += VLOADD(i, f+TRD2(d_id+1,r+1)*LDF);

		VSTORED(ff1 + ff2_mul*ff2, i, g);

	}
}

/*
*/
__kernel 
__attribute__((reqd_work_group_size(OTHER_WG_SIZE_Q2_2D22,1,1))) 
void b2pfm_22(__global _var *o_f, __global _var *o_g, int r, int prev_count) {
	const int local_id = get_local_id(0);
	
	const int g_id2d = get_group_id(0);
	const int g_id3d = get_group_id(1);

	const int d_count = POW2(K2-r-1);
	const int part_count = get_num_groups(0) / d_count;
	const int part_size = POW2(r) / part_count;

	const int d_id    = g_id2d / part_count;
	const int part_id = g_id2d % part_count;

	__global _var *g = o_g + (g_id3d * POW2(K2-1) + d_id*POW2(r) + part_id*part_size)* POW2(K3);

	const int prev_size = POW2(r) / prev_count;

	if(part_count == 1) {
		__global _var *f = o_f + (g_id3d*N2 + TRD2(2*d_id + 1,r))*LDF;

		int i;
		for(i = local_id; i < POW2(K3)/D; i += OTHER_WG_SIZE_Q2_2D22) {
			_varD res = 0.0;
			int j;
			for(j = 0; j <  min(prev_count,MAX_SUM_SIZE_Q2_2); j++) 
				res += VLOADD(i, g+(j*prev_size)*POW2(K3));	
		
			VSTORED((1.0/POW2(r))*res, i, f);
		}
	} else {
		int i;
		for(i = local_id; i < POW2(K3)/D; i += OTHER_WG_SIZE_Q2_2D22) {
			_varD res = 0.0;

			int j;
			for(j = 0; j <  min(prev_count,MAX_SUM_SIZE_Q2_2); j++) 
				res += VLOADD(i, g+(j*prev_size)*POW2(K3));	
		
			VSTORED(res, i, g);
		}		
	}

}

#endif

//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////

#if Q4SOLVER

/*
 2D solver, reduction stage, tridiagonal system formation
 o_f	Right hand side vector get_num_groups(1)*((4^K2-1)*LDF*sizeof(_var) bytes)
 o_g	Workspace (get_num_groups(1)*3*POW2(2*K2-2)*POW2(K3)*sizeof(_var) bytes)
 r		Recursion index, r = 1, 2, ..., K2-1
 get_num_groups(0) = (4^(K2-r)-1) * 3*POW2(2*r-2)
 get_num_groups(1) = Number of 2D systems
*/
__kernel 
__attribute__((reqd_work_group_size(OTHER_WG_SIZE_2D11,1,1))) 
void b4pfm_11(const __global _var *o_f, __global _var *o_g, int r) {
	const int local_id = get_local_id(0);
	
	const int g_id2d = get_group_id(0);
	const int g_id3d = get_group_id(1);

	const int line_id = g_id2d / (3*POW2(2*r-2)) + 1;
	const int trid_id = g_id2d % (3*POW2(2*r-2)) + 1;

	__global _var *g = o_g + (g_id3d * 3*POW2(K2-2) + g_id2d)*POW2(K3);
	const __global _var *f = o_f + g_id3d*N2*LDF;
	
	if(trid_id <= POW2(2*r-2)) {
		int i;
		_var ff1_mul = M1P(trid_id-1)*sinpi(1.0*(2*trid_id-1)/POW2(2*r-1));
		for(i = local_id; i < (POW2(K3)/D); i += OTHER_WG_SIZE_2D11) {
			_varD ff1 = 
				- VLOADD(i, f+TRD(4*line_id-3,r-1)*LDF)  
				+ VLOADD(i, f+TRD(4*line_id-1,r-1)*LDF) 
				+ VLOADD(i, f+TRD(4*line_id+1,r-1)*LDF) 
				- VLOADD(i, f+TRD(4*line_id+3,r-1)*LDF);

			VSTORED(ff1_mul*ff1, i, g);
		}
	} else {
		int i;
		int j = trid_id - POW2(2*r-2);
		_var ff1_mul = M1P(j-1)*sinpi(1.0*(2*j-1)/POW2(2*r));
		_var ff2_mul = SIN4(j)*sinpi(1.0*(2*j-1)/POW2(2*r));
		for(i = local_id; i < (POW2(K3)/D); i += OTHER_WG_SIZE_2D11) {
			_varD ff1 = 
				+ VLOADD(i, f+TRD(4*line_id-2,r-1)*LDF)
				+ VLOADD(i, f+TRD(4*line_id+2,r-1)*LDF);

			_varD ff2 = 
				+ VLOADD(i, f+TRD(4*line_id-3,r-1)*LDF)  
				+ VLOADD(i, f+TRD(4*line_id-1,r-1)*LDF) 
				+ VLOADD(i, f+TRD(4*line_id+1,r-1)*LDF) 
				+ VLOADD(i, f+TRD(4*line_id+3,r-1)*LDF);

			VSTORED(ff1_mul*ff1 + ff2_mul*ff2, i ,g);
		}
	}
}

/*
 2D solver, reduction stage, right hand side update, kernel B
 o_f	Right hand side vector get_num_groups(1)*((4^K2-1)*LDF*sizeof(_var) bytes)
 o_g	Workspace (get_num_groups(1)*3*POW2(2*K2-2)*POW2(K3)*sizeof(_var) bytes)
 r		Recursion index, r = 1, 2, ..., K2-1
 get_num_groups(0) = (4^(K2-r)-1)
 get_num_groups(1) = Number of 2D systems
*/
__kernel 
__attribute__((reqd_work_group_size(OTHER_WG_SIZE_2D12,1,1))) 
void b4pfm_12(__global _var *o_f, __global _var *o_g, int r, int prev_count) {
	const int local_id = get_local_id(0);
	
	const int g_id2d = get_group_id(0);
	const int g_id3d = get_group_id(1);

	const int line_count = (POW2(K2-2*r)-1);
	const int part_count = get_num_groups(0) / line_count;
	const int line_id = g_id2d / part_count;
	const int part_id = g_id2d % part_count;
	const int part_size = 3*POW2(2*r-2) / part_count;

	__global _var *g = o_g + (g_id3d * 3*POW2(K2-2) + line_id*3*POW2(2*r-2) + part_id*part_size) * POW2(K3);

	const int prev_size = 3*POW2(2*r-2) / prev_count;

	if(part_count == 1) {
		__global _var *f = o_f + (g_id3d*N2 + TRD(line_id + 1,r))*LDF;

		int i;
		for(i = local_id; i < POW2(K3)/D; i += OTHER_WG_SIZE_2D12) {
			_varD res = 0.0;

			int j;
			for(j = 0; j < min(prev_count,MAX_SUM_SIZE1); j++) 
				res += VLOADD(i, g+j*prev_size*POW2(K3));
		
			VSTORED(VLOADD(i, f)+(1.0/POW2(2*r-1))*res, i, f);
		} 
	} else {
		int i;
		for(i = local_id; i < POW2(K3)/D; i += OTHER_WG_SIZE_2D12) {
			_varD res = 0.0;

			int j;
			for(j = 0; j < min(prev_count,MAX_SUM_SIZE1); j++) 
				res += VLOADD(i, g+j*prev_size*POW2(K3));
		
			VSTORED(res, i, g);
		}
	}
}

/*
 2D solver, back substitution stage, tridiagonal system formation
 o_f	Right hand side vector ((4^K2-1)*LDF*sizeof(_var) bytes)
 o_g	Workspace (get_num_groups(1)*3*POW2(2*K2-2)*POW2(K3)*sizeof(_var) bytes)
 r		Recursion index, r = K2-1, ..., 1, 0
 get_num_groups(0) = (4^(K2-r-1)) * 3*POW2(2*r-2)
 get_num_groups(1) = Number of 2D systems
*/
__kernel 
__attribute__((reqd_work_group_size(OTHER_WG_SIZE_2D21,1,1))) 
void b4pfm_21(const __global _var *o_f, __global _var *o_g, int r) {
	const int local_id = get_local_id(0);
	
	const int g_id2d = get_group_id(0);
	const int g_id3d = get_group_id(1);

	const int d_id    = g_id2d / (3*POW2(2*r));
	const int trid_id = g_id2d % (3*POW2(2*r)) + 1;

	__global _var *g = o_g + (g_id3d * 3*POW2(K2-2) + g_id2d)*POW2(K3);
	const __global _var *f = o_f + g_id3d*N2*LDF;
	
	if(trid_id <= POW2(2*r)) {
		int i;
		_var ff1_mul = M1P(trid_id-1);
		_var ff2_mul = sinpi(1.0*(2*trid_id-1)/POW2(2*r+1));
		for(i = local_id; i < (POW2(K3)/D); i += OTHER_WG_SIZE_2D21) {
			_varD ff1 = 
				+ VLOADD(i, f+TRD(4*d_id+1,r)*LDF) 
				- VLOADD(i, f+TRD(4*d_id+3,r)*LDF);

			_varD ff2 = 0.0;
			if(d_id > 0)
				ff2 += VLOADD(i, f+TRD(d_id,r+1)*LDF);
			if(d_id < POW2(K2-2*r-2)-1)
				ff2 -= VLOADD(i, f+TRD(d_id+1,r+1)*LDF);

			VSTORED(ff1_mul*ff1 + ff2_mul*ff2, i, g);
		}
	} else {
		int i;
		int j = trid_id - POW2(2*r);
		_var ff1_mul = M1P(j-1);
		_var ff2_mul = SIN4(j);
		_var ff3_mul = sinpi(1.0*(2*j-1)/POW2(2*r+2));
		for(i = local_id; i < (POW2(K3)/D); i += OTHER_WG_SIZE_2D21) {
			_varD ff1 =  VLOADD(i, f+TRD(4*d_id+2,r)*LDF);

			_varD ff2 = 
				+ VLOADD(i, f+TRD(4*d_id+1,r)*LDF) 
				+ VLOADD(i, f+TRD(4*d_id+3,r)*LDF);

			_varD ff3 = 0.0;
			if(d_id > 0)
				ff3 += VLOADD(i, f+TRD(d_id,r+1)*LDF);
			if(d_id < POW2(K2-2*r-2)-1)
				ff3 += VLOADD(i, f+TRD(d_id+1,r+1)*LDF);

			VSTORED(ff1_mul*ff1 + ff2_mul*ff2 + ff3_mul*ff3, i, g);
		}
	}
}

/*
 2D solver, back substitution stage, right hand side update, kernel A
 o_f	Right hand side vector get_num_groups(1)*((4^K2-1)*LDF*sizeof(_var) bytes)
 o_g	Workspace (get_num_groups(1)*3*POW2(2*K2-2)*POW2(K3)*sizeof(_var) bytes)
 r		Recursion index, r = 1, 2, ..., K2-1
 get_num_groups(0) = 3*4^(K2-r-1)
 get_num_groups(1) = Number of 2D systems
*/
__kernel 
__attribute__((reqd_work_group_size(OTHER_WG_SIZE_2D22A,1,1))) 
void b4pfm_22a(__global _var *o_g, int r) {
	const int local_id = get_local_id(0);
	
	const int g_id2d = get_group_id(0);
	const int g_id3d = get_group_id(1);

	const int d_count = POW2(K2-2*r-2);
	const int part_count = get_num_groups(0) / d_count;
	const int d_id = g_id2d / part_count;
	const int part_id = g_id2d % part_count;
	const int part_size = 3*POW2(2*r) / part_count;

	__global _var *g = o_g + (g_id3d * 3*POW2(K2-2) + d_id*3*POW2(2*r) + part_id*part_size)* POW2(K3);

	const int j_start = part_id*part_size + 1;

	int i;
	for(i = local_id; i < POW2(K3)/D; i += OTHER_WG_SIZE_2D22A) {
		_varD res1 = 0.0;
		_varD res2 = 0.0;
		_varD res3 = 0.0;

		int j = j_start;
		int l = 0;
		for(; l < part_size && j <= POW2(2*r); l++, j++) {
			_varD tmp = M1P(j-1)*VLOADD(i, g+l*POW2(K3));
			res1 += tmp;
			res3 -= tmp;
		}

		for(j -= POW2(2*r); l < part_size && j <= POW2(2*r+1); l++, j++) {
			_varD tmp1 = VLOADD(i, g+l*POW2(K3));
			_varD tmp2 = SIN4(j)*tmp1;
			res1 += tmp2;
			res2 += M1P(j-1)*tmp1;
			res3 += tmp2;
		}
		
		VSTORED(res1, i, g);
		VSTORED(res2, i, g+POW2(K3));
		VSTORED(res3, i, g+2*POW2(K3));
	}
}

/*
 2D solver, back substitution stage, right hand side update, kernel B
 o_f	Right hand side vector get_num_groups(1)*((4^K2-1)*LDF*sizeof(_var) bytes)
 o_g	Workspace (get_num_groups(1)*3*POW2(2*K2-2)*POW2(K3)*sizeof(_var) bytes)
 r		Recursion index, r = 1, 2, ..., K2-1
 get_num_groups(0) = 3*4^(K2-r-1)
 get_num_groups(1) = Number of 2D systems
*/
__kernel 
__attribute__((reqd_work_group_size(OTHER_WG_SIZE_2D22B,1,1))) 
void b4pfm_22b(__global _var *o_f, __global _var *o_g, int r, int prev_count) {
	const int local_id = get_local_id(0);
	
	const int g_id2d = get_group_id(0);
	const int g_id3d = get_group_id(1);

	const int d_count = POW2(K2-2*r-2);
	const int part_count = get_num_groups(0) / (3*d_count);
	const int part_size = 3*POW2(2*r) / part_count;

	const int d_id    = (g_id2d/3) / part_count;
	const int part_id = (g_id2d/3) % part_count;
	const int line_id = g_id2d % 3;

	__global _var *g = o_g + (g_id3d * 3*POW2(K2-2) + d_id*3*POW2(2*r) + part_id*part_size)* POW2(K3);

	const int prev_size = 3*POW2(2*r) / prev_count;

	if(part_count == 1) {
		__global _var *f = o_f + (g_id3d*N2 + TRD(4*d_id+line_id+1,r))*LDF;

		int i;
		for(i = local_id; i < POW2(K3)/D; i += OTHER_WG_SIZE_2D22B) {
			_varD res = 0.0;
			int j;
			for(j = 0; j <  min(prev_count,MAX_SUM_SIZE2B); j++) 
				res += VLOADD(i, g+(j*prev_size+line_id)*POW2(K3));	
		
			VSTORED((1.0/POW2(2*r+1))*res, i, f);
		}
	} else {
		int i;
		for(i = local_id; i < POW2(K3)/D; i += OTHER_WG_SIZE_2D22B) {
			_varD res = 0.0;

			int j;
			for(j = 0; j <  min(prev_count,MAX_SUM_SIZE2B); j++) 
				res += VLOADD(i, g+(j*prev_size+line_id)*POW2(K3));	
		
			VSTORED(res, i, g+line_id*POW2(K3));
		}		
	}

}

#endif

