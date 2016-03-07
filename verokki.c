/*
 *  Created on: May 27, 2013
 *      Author: Mirko Myllykoski (mirko.myllykoski@gmail.com)
 */

#include <omp.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#define CR 0

#ifndef THREADS
#define THREADS 1
#endif

#ifndef DOUBLE
#error "DOUBLE is not defined."
#endif

#if DOUBLE
typedef double _var;
#else
typedef float _var;
#endif

#define PI (acos(-1.0))

/* 2^j */
#define POW2(j) (1<<(j))

/* 4^j */
#define POW4(j) POW2(2*(j))

/* (-1)^j */
#define M1P(j) (1-((j)&0x1)*2)

/* sin(pi*(2*j-1)/4) */
#define SIN4(j) ((1-(((j)-1)&0x2))*sqrt(1.0/2.0))

#define TRD2(i,r) ((i)*POW2(r)-1)
#define TRD4(i,r) ((i)*POW4(r)-1)

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

inline void zeros(_var *buff, int n) {
	int i;

	for(i = 0; i < n; i++)
		buff[i] = 0.0;
}

inline void cpy(_var *dest, _var *sour, int n) {
	int i;

	for(i = 0; i < n; i++)
		dest[i] = sour[i];
}

inline void cpyneg(_var *dest, _var *sour, int n) {
	int i;

	for(i = 0; i < n; i++)
		dest[i] = -1.0*sour[i];
}

inline void cpycoef(_var coef, _var *dest, _var *sour, int n) {
	int i;

	for(i = 0; i < n; i++)
		dest[i] = coef*sour[i];
}

inline void add(_var *dest, _var *sour, int n) {
	int i;

	for(i = 0; i < n; i++)
		dest[i] += sour[i];
}

inline void sub(_var *dest, _var *sour, int n) {
	int i;

	for(i = 0; i < n; i++)
		dest[i] -= sour[i];
}

inline void addcoef(_var coef, _var *dest, _var *sour, int n) {
	int i;

	for(i = 0; i < n; i++)
		dest[i] += coef*sour[i];
}

#if !CR

void soltri(int j, int r, _var shift, _var *f, _var *dd, int n) {
	_var b[n];	

	int i;

	double mins = 2*cos(PI*(2*j-1)/POW2(r+1)) - shift;
	for(i = 0; i < n; i++) 
		b[i] = dd[i]-mins;

	for (i = 1; i < n; i++) {
		_var m = (-1.0)/b[i-1];
		b[i] = b[i] - m*(-1);
		f[i] = f[i] - m*f[i-1];
	}
 
	f[n-1] = f[n-1]/b[n-1];
 
	for (i = n - 2; i >= 0; i--)
		f[i]=(f[i]-(-1)*f[i+1])/b[i];
}

#else

void soltri(int jjj, int rrr, _var shift, _var *f, _var *ddd, int n) {
	int k = log(n+1)/log(2.0);

	_var dd[k];
	_var tt[k];

	dd[0] = ddd[0] - 2*cos(PI*(2*jjj-1)/POW2(rrr+1)) + shift;
	tt[0] = -1.0;

	int r;
	for(r = 1; r <= k-1; r++) {
		_var d = dd[r-1];
		_var t = tt[r-1];

		dd[r] = d-(2.0*t*t)/d;
		tt[r] = -(t*t)/d;

		_var mul = (t/d);

		int i;
		for(i = 1; i <= POW2(k-r)-1; i++) 
			f[TRD2(i,r)] -= mul*(f[TRD2(2*i-1,r-1)]+f[TRD2(2*i+1,r-1)]);
	}

	for(r = k-1; 0 <= r; r--) {
		_var d = dd[r];
		_var t = tt[r];

		int i;
		for(i = 1; i <= POW2(k-r)-1; i += 2) {
			_var upper = 0.0;
			_var lower = 0.0;

			if(i > 1)
				upper = f[TRD2((i-1)/2,r+1)];    
			if(i < POW2(k-r)-1)
         		lower = f[TRD2((i-1)/2+1,r+1)];			

			f[TRD2(i,r)] = (f[TRD2(i,r)] - t*(upper+lower))/d;

		}
	}

}

#endif


void _base4solver(_var *f, _var *dd, _var shift, int k1, int n2, int ldf, int parallel) {

	int thread_count = parallel ? THREADS : 1;

	/* Reduction stage */
	int r;	  
	for(r = 1; r <= k1/2-(k1 & 1 ? 0 : 1) && POW2(k1-2*r)-1 >= thread_count; r++) {
		int i;	

		#pragma omp parallel for private(i) num_threads(thread_count)
		for(i = 1; i <= POW2(k1-2*r)-1; i++) {

			_var nu1[n2];
			_var ff1[n2];
			_var ff2[n2];
			_var tmp[n2];

			zeros(nu1,n2);

			cpyneg(ff1,f+TRD4(4*i-3,r-1)*ldf,n2);
			add(ff1,f+TRD4(4*i-1,r-1)*ldf,n2);
			add(ff1,f+TRD4(4*i+1,r-1)*ldf,n2);
			sub(ff1,f+TRD4(4*i+3,r-1)*ldf,n2);

			int j;
			for(j = 1; j <= POW2(2*r-2); j++) {
				cpycoef(M1P(j-1),tmp,ff1,n2);
				soltri(j,2*r-2,shift,tmp,dd,n2);
				addcoef(sin(PI*(2*j-1)/POW2(2*r-1)),nu1,tmp,n2);
			}
				
			cpy(ff1,f+TRD4(4*i-2,r-1)*ldf,n2);
			add(ff1,f+TRD4(4*i+2,r-1)*ldf,n2);

			cpy(ff2,f+TRD4(4*i-3,r-1)*ldf,n2);
			add(ff2,f+TRD4(4*i-1,r-1)*ldf,n2);
			add(ff2,f+TRD4(4*i+1,r-1)*ldf,n2);
			add(ff2,f+TRD4(4*i+3,r-1)*ldf,n2);
      		
      		for(j = 1; j <= POW2(2*r-1); j++) {
				cpycoef(M1P(j-1),tmp,ff1,n2);
				addcoef(SIN4(j),tmp,ff2,n2);
				soltri(j,2*r-1,shift,tmp,dd,n2);
				addcoef(sin(PI*(2*j-1)/POW2(2*r)),nu1,tmp,n2);
			}
      
			addcoef(1.0/POW2(2*r-1),f+TRD4(i,r)*ldf,nu1,n2);	 
		}	
	}

	for(; r <= k1/2-(k1 & 1 ? 0 : 1); r++) {
		
		_var ff1[n2];
		_var ff2[n2];
		_var ff3[n2];

		int i;	
		for(i = 1; i <= POW2(k1-2*r)-1; i++) {

			cpyneg(ff1,f+TRD4(4*i-3,r-1)*ldf,n2);
			add(ff1,f+TRD4(4*i-1,r-1)*ldf,n2);
			add(ff1,f+TRD4(4*i+1,r-1)*ldf,n2);
			sub(ff1,f+TRD4(4*i+3,r-1)*ldf,n2);

			cpy(ff2,f+TRD4(4*i-2,r-1)*ldf,n2);
			add(ff2,f+TRD4(4*i+2,r-1)*ldf,n2);

			cpy(ff3,f+TRD4(4*i-3,r-1)*ldf,n2);
			add(ff3,f+TRD4(4*i-1,r-1)*ldf,n2);
			add(ff3,f+TRD4(4*i+1,r-1)*ldf,n2);
			add(ff3,f+TRD4(4*i+3,r-1)*ldf,n2);

			#pragma omp parallel num_threads(thread_count)
			{
				_var nu1[n2];
				_var tmp[n2];

				zeros(nu1,n2);

				int part_size = 3*POW2(2*r-2) / thread_count;
				int begin = omp_get_thread_num()*part_size + 1;
				int end = begin + part_size;
					
				int j;
				for(j = begin; j < end; j++) {
					if(j <= POW2(2*r-2)) {
						cpycoef(M1P(j-1),tmp,ff1,n2);
						soltri(j,2*r-2,shift,tmp,dd,n2);
						addcoef(sin(PI*(2*j-1)/POW2(2*r-1)),nu1,tmp,n2);
					} else  {
						int jj = j - POW2(2*r-2);
						cpycoef(M1P(jj-1),tmp,ff2,n2);
						addcoef(SIN4(jj),tmp,ff3,n2);
						soltri(jj,2*r-1,shift,tmp,dd,n2);
						addcoef(sin(PI*(2*jj-1)/POW2(2*r)),nu1,tmp,n2);
					}
				}
      
				#pragma omp critical
				addcoef(1.0/POW2(2*r-1),f+TRD4(i,r)*ldf,nu1,n2);	 
			}
		}	
	}

	if(k1 & 1) {

		_var o_f[n2];
		cpy(o_f,f+TRD2(1,k1-1)*ldf,n2);
		zeros(f+TRD2(1,k1-1)*ldf,n2);

		#pragma omp parallel num_threads(thread_count)
		{

			_var nu1[n2];
			_var tmp[n2];

			zeros(nu1,n2);

			int begin, end;
			if(parallel) {
				int part_size = POW2(k1-1) / thread_count;
				begin = omp_get_thread_num()*part_size + 1;
				end = begin + part_size;
			} else {
				begin = 1;
				end = POW2(k1-1) + 1;
			}

			int j;
			for(j = begin; j < end; j++) {
				cpy(tmp,o_f,n2);
				soltri(j,k1-1,shift,tmp,dd,n2);
				add(nu1,tmp,n2);
			}

			#pragma omp critical
      		addcoef(1.0/POW2(k1-1),f+TRD2(1,k1-1)*ldf,nu1,n2);
		}

	}


  	/* Back substitution stage */
	for(r = k1/2-1; r >= 0 && POW2(k1-2*r-2) < thread_count; r--) {

		int d;
		for(d = 0; d < POW2(k1-2*r-2); d++) {

			_var ff1[n2];
			_var ff2[n2];
			_var ff3[n2];
			_var ff4[n2];
			_var ff5[n2];

			cpy(ff1,f+TRD4(4*d+1,r)*ldf,n2);
			sub(ff1,f+TRD4(4*d+3,r)*ldf,n2);
      
			zeros(ff2,n2);
			if(d > 0)
				add(ff2,f+TRD4(d,r+1)*ldf,n2);    
			if(d < POW2(k1-2*r-2)-1)
         		sub(ff2,f+TRD4(d+1,r+1)*ldf,n2);

			cpy(ff3,f+TRD4(4*d+2,r)*ldf,n2);
			
			cpy(ff4,f+TRD4(4*d+1,r)*ldf,n2);
			add(ff4,f+TRD4(4*d+3,r)*ldf,n2);
      
      		zeros(ff5,n2);
			if(d > 0)
				add(ff5,f+TRD4(d,r+1)*ldf,n2);
			if(d < POW2(k1-2*r-2)-1)
         		add(ff5,f+TRD4(d+1,r+1)*ldf,n2);

			zeros(f+TRD4(4*d+1,r)*ldf,n2);
			zeros(f+TRD4(4*d+2,r)*ldf,n2);
			zeros(f+TRD4(4*d+3,r)*ldf,n2);

			#pragma omp parallel num_threads(thread_count)
			{

				_var nu1[n2];
				_var nu2[n2];
				_var nu3[n2];
				_var tmp[n2];

				zeros(nu1,n2);
				zeros(nu2,n2);
				zeros(nu3,n2);

				int part_size = 3*POW2(2*r) / thread_count;
				int begin = omp_get_thread_num()*part_size + 1;
				int end = begin + part_size;

				int j;
				for(j = begin; j < end; j++) {
					if(j <= POW2(2*r)) {
						cpycoef(M1P(j-1),tmp,ff1,n2);
						addcoef(sin(PI*(2*j-1)/POW2(2*r+1)),tmp,ff2,n2);
						soltri(j,2*r,shift,tmp,dd,n2);
						addcoef(M1P(j-1),nu1,tmp,n2);
						addcoef(M1P(j),nu3,tmp,n2);
					} else {
						int jj = j - POW2(2*r);
						cpycoef(M1P(jj-1),tmp,ff3,n2);
						addcoef(SIN4(jj),tmp,ff4,n2);
						addcoef(sin(PI*(2*jj-1)/(POW2(2*r+2))),tmp,ff5,n2);
						soltri(jj,2*r+1,shift,tmp,dd,n2);
						addcoef(SIN4(jj),nu1,tmp,n2);
						addcoef(M1P(jj-1),nu2,tmp,n2);
						addcoef(SIN4(jj),nu3,tmp,n2);
					}
				}

				#pragma omp critical (b42da)
      			addcoef(1.0/POW2(2*r+1),f+TRD4(4*d+1,r)*ldf,nu1,n2);

				#pragma omp critical (b42db)
      			addcoef(1.0/POW2(2*r+1),f+TRD4(4*d+2,r)*ldf,nu2,n2);

				#pragma omp critical (b42dc)
      			addcoef(1.0/POW2(2*r+1),f+TRD4(4*d+3,r)*ldf,nu3,n2);

			}
		}
	}

	for(; r >= 0; r--) {

		int d;
		#pragma omp parallel for private(d) num_threads(thread_count)
		for(d = 0; d < POW2(k1-2*r-2); d++) {

			_var nu1[n2];
			_var nu2[n2];
			_var nu3[n2];
			_var ff1[n2];
			_var ff2[n2];
			_var ff3[n2];
			_var tmp[n2];

			zeros(nu1,n2);
			zeros(nu2,n2);
			zeros(nu3,n2);

			cpy(ff1,f+TRD4(4*d+1,r)*ldf,n2);
			sub(ff1,f+TRD4(4*d+3,r)*ldf,n2);
      
			zeros(ff2,n2);
			if(d > 0)
				add(ff2,f+TRD4(d,r+1)*ldf,n2);    
			if(d < POW2(k1-2*r-2)-1)
         		sub(ff2,f+TRD4(d+1,r+1)*ldf,n2);

			int j;
			for(j = 1; j <= POW2(2*r); j++) {
				cpycoef(M1P(j-1),tmp,ff1,n2);
				addcoef(sin(PI*(2*j-1)/POW2(2*r+1)),tmp,ff2,n2);
				soltri(j,2*r,shift,tmp,dd,n2);
				addcoef(M1P(j-1),nu1,tmp,n2);
				addcoef(M1P(j),nu3,tmp,n2);
			}

			cpy(ff1,f+TRD4(4*d+2,r)*ldf,n2);
			
			cpy(ff2,f+TRD4(4*d+1,r)*ldf,n2);
			add(ff2,f+TRD4(4*d+3,r)*ldf,n2);
      
      		zeros(ff3,n2);
			if(d > 0)
				add(ff3,f+TRD4(d,r+1)*ldf,n2);
			if(d < POW2(k1-2*r-2)-1)
         		add(ff3,f+TRD4(d+1,r+1)*ldf,n2);

			for(j = 1; j <= POW2(2*r+1); j++) {
				cpycoef(M1P(j-1),tmp,ff1,n2);
				addcoef(SIN4(j),tmp,ff2,n2);
				addcoef(sin(PI*(2*j-1)/(POW2(2*r+2))),tmp,ff3,n2);
				soltri(j,2*r+1,shift,tmp,dd,n2);
				addcoef(SIN4(j),nu1,tmp,n2);
				addcoef(M1P(j-1),nu2,tmp,n2);
				addcoef(SIN4(j),nu3,tmp,n2);
			}

      		cpycoef(1.0/POW2(2*r+1),f+TRD4(4*d+1,r)*ldf,nu1,n2);
      		cpycoef(1.0/POW2(2*r+1),f+TRD4(4*d+2,r)*ldf,nu2,n2);
      		cpycoef(1.0/POW2(2*r+1),f+TRD4(4*d+3,r)*ldf,nu3,n2);

		}
	}

}

base4solver(_var *f, _var *dd, _var shift, int k1, int n2, int ldf) {
	_base4solver(f, dd, shift, k1, n2, ldf, 1);
}

void base4solver3d(_var *f, _var *dd, int k1, int k2, int n3, int ldf) {

	int n2 = POW2(k2)-1;

	_var *buff = malloc(THREADS*7*n2*ldf*sizeof(_var));

	/* Reduction stage */
	int r;	  
	for(r = 1; r <= k1/2-(k1 & 1 ? 0 : 1) && POW2(k1-2*r) >= THREADS; r++) {
		int i;		

		#pragma omp parallel for private(i) num_threads(THREADS)
		for(i = 1; i <= POW2(k1-2*r)-1; i++) {

			_var *nu1 = buff + (omp_get_thread_num()*4+0)*n2*ldf;
			_var *ff1 = buff + (omp_get_thread_num()*4+1)*n2*ldf;
			_var *ff2 = buff + (omp_get_thread_num()*4+2)*n2*ldf;
			_var *tmp = buff + (omp_get_thread_num()*4+3)*n2*ldf;

			zeros(nu1,n2*ldf);

			cpyneg(ff1,f+TRD4(4*i-3,r-1)*n2*ldf,n2*ldf);
			add(ff1,f+TRD4(4*i-1,r-1)*n2*ldf,n2*ldf);
			add(ff1,f+TRD4(4*i+1,r-1)*n2*ldf,n2*ldf);
			sub(ff1,f+TRD4(4*i+3,r-1)*n2*ldf,n2*ldf);

			int j;
			for(j = 1; j <= POW2(2*r-2); j++) {
				cpycoef(M1P(j-1),tmp,ff1,n2*ldf);
				_base4solver(tmp, dd, -2*cos(PI*(2*j-1)/POW2(2*r-1)), k2, n3, ldf, 0);
				addcoef(sin(PI*(2*j-1)/POW2(2*r-1)),nu1,tmp,n2*ldf);
			}
				
			cpy(ff1,f+TRD4(4*i-2,r-1)*n2*ldf,n2*ldf);
			add(ff1,f+TRD4(4*i+2,r-1)*n2*ldf,n2*ldf);

			cpy(ff2,f+TRD4(4*i-3,r-1)*n2*ldf,n2*ldf);
			add(ff2,f+TRD4(4*i-1,r-1)*n2*ldf,n2*ldf);
			add(ff2,f+TRD4(4*i+1,r-1)*n2*ldf,n2*ldf);
			add(ff2,f+TRD4(4*i+3,r-1)*n2*ldf,n2*ldf);
      		
      		for(j = 1; j <= POW2(2*r-1); j++) {
				cpycoef(M1P(j-1),tmp,ff1,n2*ldf);
				addcoef(SIN4(j),tmp,ff2,n2*ldf);
				_base4solver(tmp, dd, -2*cos(PI*(2*j-1)/POW2(2*r)), k2, n3, ldf, 0);
				addcoef(sin(PI*(2*j-1)/POW2(2*r)),nu1,tmp,n2*ldf);
			}
      
			addcoef(1.0/POW2(2*r-1),f+TRD4(i,r)*n2*ldf,nu1,n2*ldf);	 
		}	
	}

	for(; r <= k1/2-(k1 & 1 ? 0 : 1); r++) {
		int i;	

		for(i = 1; i <= POW2(k1-2*r)-1; i++) {

			_var *ff1 = buff + 0*n2*ldf;
			_var *ff2 = buff + 1*n2*ldf;
			_var *ff3 = buff + 2*n2*ldf;

			cpyneg(ff1,f+TRD4(4*i-3,r-1)*n2*ldf,n2*ldf);
			add(ff1,f+TRD4(4*i-1,r-1)*n2*ldf,n2*ldf);
			add(ff1,f+TRD4(4*i+1,r-1)*n2*ldf,n2*ldf);
			sub(ff1,f+TRD4(4*i+3,r-1)*n2*ldf,n2*ldf);

			cpy(ff2,f+TRD4(4*i-2,r-1)*n2*ldf,n2*ldf);
			add(ff2,f+TRD4(4*i+2,r-1)*n2*ldf,n2*ldf);

			cpy(ff3,f+TRD4(4*i-3,r-1)*n2*ldf,n2*ldf);
			add(ff3,f+TRD4(4*i-1,r-1)*n2*ldf,n2*ldf);
			add(ff3,f+TRD4(4*i+1,r-1)*n2*ldf,n2*ldf);
			add(ff3,f+TRD4(4*i+3,r-1)*n2*ldf,n2*ldf);

			#pragma omp parallel num_threads(THREADS)
			{
				_var *nu1 = buff + (omp_get_thread_num()*2+3)*n2*ldf;
				_var *tmp = buff + (omp_get_thread_num()*2+4)*n2*ldf;
				zeros(nu1,n2*ldf);

				int part_size = 3*POW2(2*r-2) / THREADS;
				int begin = omp_get_thread_num()*part_size + 1;
				int end = begin + part_size;

				int j;
				for(j = begin; j < end; j++) {
					if(j <= POW2(2*r-2)) {
						cpycoef(M1P(j-1),tmp,ff1,n2*ldf);
						_base4solver(tmp, dd, -2*cos(PI*(2*j-1)/POW2(2*r-1)), k2, n3, ldf, 0);
						addcoef(sin(PI*(2*j-1)/POW2(2*r-1)),nu1,tmp,n2*ldf);
					} else  {
						int jj = j - POW2(2*r-2);
						cpycoef(M1P(jj-1),tmp,ff2,n2*ldf);
						addcoef(SIN4(jj),tmp,ff3,n2*ldf);
						_base4solver(tmp, dd, -2*cos(PI*(2*jj-1)/POW2(2*r)), k2, n3, ldf, 0);
						addcoef(sin(PI*(2*jj-1)/POW2(2*r)),nu1,tmp,n2*ldf);
					}
				}
      
				#pragma omp critical
				addcoef(1.0/POW2(2*r-1),f+TRD4(i,r)*n2*ldf,nu1,n2*ldf);	 
			}
		}	
	}

	if(k1 & 1) {

		_var *o_f = buff;
		cpy(o_f,f+TRD2(1,k1-1)*n2*ldf,n2*ldf);
		zeros(f+TRD2(1,k1-1)*n2*ldf,n2*ldf);

		#pragma omp parallel num_threads(THREADS)
		{

			_var *nu1 = buff + (omp_get_thread_num()*2+1)*n2*ldf;
			_var *tmp = buff + (omp_get_thread_num()*2+2)*n2*ldf;
			zeros(nu1,n2*ldf);

			int begin, end;
			if(1 < THREADS) {
				int part_size = POW2(k1-1) / THREADS;
				begin = omp_get_thread_num()*part_size + 1;
				end = begin + part_size;
			} else {
				begin = 1;
				end = POW2(k1-1) + 1;
			}

			int j;
			for(j = begin; j < end; j++) {
				cpy(tmp,o_f,n2*ldf);
				_base4solver(tmp, dd, -2*cos(PI*(2*j-1)/POW2(k1)), k2, n3, ldf, 0);
				add(nu1,tmp,n2*ldf);
			}

			#pragma omp critical
	      	addcoef(1.0/POW2(k1-1),f+TRD2(1,k1-1)*n2*ldf,nu1,n2*ldf);
		}

	}


  	/* Back substitution stage */
	for(r = k1/2-1; r >= 0 && POW2(k1-2*r-2) < THREADS; r--) {

		int d;
		for(d = 0; d < POW2(k1-2*r-2); d++) {

			_var *ff1 = buff + 0*n2*ldf;
			_var *ff2 = buff + 1*n2*ldf;
			_var *ff3 = buff + 2*n2*ldf;
			_var *ff4 = buff + 3*n2*ldf;
			_var *ff5 = buff + 4*n2*ldf;

			cpy(ff1,f+TRD4(4*d+1,r)*n2*ldf,n2*ldf);
			sub(ff1,f+TRD4(4*d+3,r)*n2*ldf,n2*ldf);
      
			zeros(ff2,n2*ldf);
			if(d > 0)
				add(ff2,f+TRD4(d,r+1)*n2*ldf,n2*ldf);    
			if(d < POW2(k1-2*r-2)-1)
         		sub(ff2,f+TRD4(d+1,r+1)*n2*ldf,n2*ldf);

			cpy(ff3,f+TRD4(4*d+2,r)*n2*ldf,n2*ldf);
			
			cpy(ff4,f+TRD4(4*d+1,r)*n2*ldf,n2*ldf);
			add(ff4,f+TRD4(4*d+3,r)*n2*ldf,n2*ldf);
      
      		zeros(ff5,n2*ldf);
			if(d > 0)
				add(ff5,f+TRD4(d,r+1)*n2*ldf,n2*ldf);
			if(d < POW2(k1-2*r-2)-1)
         		add(ff5,f+TRD4(d+1,r+1)*n2*ldf,n2*ldf);

			zeros(f+TRD4(4*d+1,r)*n2*ldf,n2*ldf);
			zeros(f+TRD4(4*d+2,r)*n2*ldf,n2*ldf);
			zeros(f+TRD4(4*d+3,r)*n2*ldf,n2*ldf);

			#pragma omp parallel num_threads(THREADS)
			{

				_var *nu1 = buff + (omp_get_thread_num()*4+5)*n2*ldf;
				_var *nu2 = buff + (omp_get_thread_num()*4+6)*n2*ldf;
				_var *nu3 = buff + (omp_get_thread_num()*4+7)*n2*ldf;
				_var *tmp = buff + (omp_get_thread_num()*4+8)*n2*ldf;

				zeros(nu1,n2*ldf);
				zeros(nu2,n2*ldf);
				zeros(nu3,n2*ldf);

				int part_size = 3*POW2(2*r) / THREADS;
				int begin = omp_get_thread_num()*part_size + 1;
				int end = begin + part_size;

				int j;
				for(j = begin; j < end; j++) {
					if(j <= POW2(2*r)) {
						cpycoef(M1P(j-1),tmp,ff1,n2*ldf);
						addcoef(sin(PI*(2*j-1)/POW2(2*r+1)),tmp,ff2,n2*ldf);
						_base4solver(tmp, dd, -2*cos(PI*(2*j-1)/POW2(2*r+1)), k2, n3, ldf, 0);
						addcoef(M1P(j-1),nu1,tmp,n2*ldf);
						addcoef(M1P(j),nu3,tmp,n2*ldf);
					} else {
						int jj = j - POW2(2*r);
						cpycoef(M1P(jj-1),tmp,ff3,n2*ldf);
						addcoef(SIN4(jj),tmp,ff4,n2*ldf);
						addcoef(sin(PI*(2*jj-1)/(POW2(2*r+2))),tmp,ff5,n2*ldf);
						_base4solver(tmp, dd, -2*cos(PI*(2*jj-1)/POW2(2*r+2)), k2, n3, ldf, 0);
						addcoef(SIN4(jj),nu1,tmp,n2*ldf);
						addcoef(M1P(jj-1),nu2,tmp,n2*ldf);
						addcoef(SIN4(jj),nu3,tmp,n2*ldf);
					}
				}

				#pragma omp critical (b43da)
      			addcoef(1.0/POW2(2*r+1),f+TRD4(4*d+1,r)*n2*ldf,nu1,n2*ldf);

				#pragma omp critical (b43db)
      			addcoef(1.0/POW2(2*r+1),f+TRD4(4*d+2,r)*n2*ldf,nu2,n2*ldf);

				#pragma omp critical (b43dc)
      			addcoef(1.0/POW2(2*r+1),f+TRD4(4*d+3,r)*n2*ldf,nu3,n2*ldf);

			}
		}
	}

	for(; r >= 0; r--) { 

		int d;
		#pragma omp parallel for private(d) num_threads(THREADS)
		for(d = 0; d < POW2(k1-2*r-2); d++) {

			_var *nu1 = buff + (omp_get_thread_num()*7+0)*n2*ldf;
			_var *nu2 = buff + (omp_get_thread_num()*7+1)*n2*ldf;
			_var *nu3 = buff + (omp_get_thread_num()*7+2)*n2*ldf;
			_var *ff1 = buff + (omp_get_thread_num()*7+3)*n2*ldf;
			_var *ff2 = buff + (omp_get_thread_num()*7+4)*n2*ldf;
			_var *ff3 = buff + (omp_get_thread_num()*7+5)*n2*ldf;
			_var *tmp = buff + (omp_get_thread_num()*7+6)*n2*ldf;

			zeros(nu1,n2*ldf);
			zeros(nu2,n2*ldf);
			zeros(nu3,n2*ldf);

			cpy(ff1,f+TRD4(4*d+1,r)*n2*ldf,n2*ldf);
			sub(ff1,f+TRD4(4*d+3,r)*n2*ldf,n2*ldf);
      
			zeros(ff2,n2*ldf);
			if(d > 0)
				add(ff2,f+TRD4(d,r+1)*n2*ldf,n2*ldf);    
			if(d < POW2(k1-2*r-2)-1)
         		sub(ff2,f+TRD4(d+1,r+1)*n2*ldf,n2*ldf);

			int j;
			for(j = 1; j <= POW2(2*r); j++) {
				cpycoef(M1P(j-1),tmp,ff1,n2*ldf);
				addcoef(sin(PI*(2*j-1)/POW2(2*r+1)),tmp,ff2,n2*ldf);
				_base4solver(tmp, dd, -2*cos(PI*(2*j-1)/POW2(2*r+1)), k2, n3, ldf, 0);
				addcoef(M1P(j-1),nu1,tmp,n2*ldf);
				addcoef(M1P(j),nu3,tmp,n2*ldf);
			}

			cpy(ff1,f+TRD4(4*d+2,r)*n2*ldf,n2*ldf);
			
			cpy(ff2,f+TRD4(4*d+1,r)*n2*ldf,n2*ldf);
			add(ff2,f+TRD4(4*d+3,r)*n2*ldf,n2*ldf);
      
      		zeros(ff3,n2*ldf);
			if(d > 0)
				add(ff3,f+TRD4(d,r+1)*n2*ldf,n2*ldf);
			if(d < POW2(k1-2*r-2)-1)
         		add(ff3,f+TRD4(d+1,r+1)*n2*ldf,n2*ldf);

			for(j = 1; j <= POW2(2*r+1); j++) {
				cpycoef(M1P(j-1),tmp,ff1,n2*ldf);
				addcoef(SIN4(j),tmp,ff2,n2*ldf);
				addcoef(sin(PI*(2*j-1)/(POW2(2*r+2))),tmp,ff3,n2*ldf);
				_base4solver(tmp, dd, -2*cos(PI*(2*j-1)/POW2(2*r+2)), k2, n3, ldf, 0);
				addcoef(SIN4(j),nu1,tmp,n2*ldf);
				addcoef(M1P(j-1),nu2,tmp,n2*ldf);
				addcoef(SIN4(j),nu3,tmp,n2*ldf);
			}

      		cpycoef(1.0/POW2(2*r+1),f+TRD4(4*d+1,r)*n2*ldf,nu1,n2*ldf);
      		cpycoef(1.0/POW2(2*r+1),f+TRD4(4*d+2,r)*n2*ldf,nu2,n2*ldf);
      		cpycoef(1.0/POW2(2*r+1),f+TRD4(4*d+3,r)*n2*ldf,nu3,n2*ldf);

		}
	}

	free(buff);

}



void _base2solver(_var *f, _var *dd, _var shift, int k1, int n2, int ldf, int parallel) {

	int thread_count = parallel ? THREADS : 1;

	/* Reduction stage */
	int r;	  
	for(r = 1; r <= k1-1 && POW2(k1-r)-1 >= thread_count; r++) {
		int i;		
		#pragma omp parallel for private(i) num_threads(thread_count)
		for(i = 1; i <= POW2(k1-r)-1; i++) {

			_var nu[n2];
			_var ff[n2];
			_var tmp[n2];

			zeros(nu,n2);

			cpy(ff,f+TRD2(2*i-1,r-1)*ldf,n2);
			add(ff,f+TRD2(2*i+1,r-1)*ldf,n2);

			int j;
			for(j = 1; j <= POW2(r-1); j++) {
				cpycoef(M1P(j-1)*sin(PI*(2*j-1)/POW2(r)),tmp,ff,n2);
				soltri(j,r-1,shift,tmp,dd,n2);
				add(nu,tmp,n2);
			}
      
			addcoef(1.0/POW2(r-1),f+TRD2(i,r)*ldf,nu,n2);	  
		}	
	}

	for(; r <= k1-1; r++) {

		int i;		
		for(i = 1; i <= POW2(k1-r)-1; i++) {

			_var ff[n2];

			cpy(ff,f+TRD2(2*i-1,r-1)*ldf,n2);
			add(ff,f+TRD2(2*i+1,r-1)*ldf,n2);

			#pragma omp parallel num_threads(thread_count)
			{
				_var nu[n2];
				_var tmp[n2];
				zeros(nu,n2);

				int part_size = POW2(r-1) / thread_count;
				int begin = omp_get_thread_num()*part_size + 1;
				int end = begin + part_size;

				int j;
				for(j = begin; j < end; j++) {
					cpycoef(M1P(j-1)*sin(PI*(2*j-1)/POW2(r)),tmp,ff,n2);
					soltri(j,r-1,shift,tmp,dd,n2);
					add(nu,tmp,n2);
				}

    			#pragma omp critical
				addcoef(1.0/POW2(r-1),f+TRD2(i,r)*ldf,nu,n2);	  
			}
		}	
	}

  	/* Back substitution stage */
	for(r = k1-1; r >= 0 && POW2(k1-r-1) < thread_count; r--) {

		int i;
		for(i = 1; i <= POW2(k1-r)-1; i += 2) {

			_var ff[n2];
      
			zeros(ff,n2);
			if(i > 1)
				add(ff,f+TRD2((i-1)/2,r+1)*ldf,n2);    
			if(i < POW2(k1-r)-1)
         		add(ff,f+TRD2((i-1)/2+1,r+1)*ldf,n2);

			_var o_f[n2];
			cpy(o_f,f+TRD2(i,r)*ldf,n2);
			zeros(f+TRD2(i,r)*ldf,n2);

			#pragma omp parallel num_threads(thread_count)
			{
				_var nu[n2];
				_var tmp[n2];
				zeros(nu,n2);

				int part_size = POW2(r) / thread_count;
				int begin = omp_get_thread_num()*part_size + 1;
				int end = begin + part_size;

				int j;
				for(j = begin; j < end; j++) {
					cpy(tmp,o_f,n2);
					addcoef(M1P(j-1)*sin(PI*(2*j-1)/POW2(r+1)),tmp,ff,n2);
					soltri(j,r,shift,tmp,dd,n2);
					add(nu,tmp,n2);
				}

				#pragma omp critical
      			addcoef(1.0/POW2(r),f+TRD2(i,r)*ldf,nu,n2);
			}
		}
	}

	for(; r >= 0; r--) {

		int i;
		#pragma omp parallel for private(i) num_threads(thread_count)
		for(i = 1; i <= POW2(k1-r)-1; i += 2) {

			_var nu[n2];
			_var ff[n2];
			_var tmp[n2];

			zeros(nu,n2);
      
			zeros(ff,n2);
			if(i > 1)
				add(ff,f+TRD2((i-1)/2,r+1)*ldf,n2);    
			if(i < POW2(k1-r)-1)
         		add(ff,f+TRD2((i-1)/2+1,r+1)*ldf,n2);

			int j;
			for(j = 1; j <= POW2(r); j++) {
				cpy(tmp,f+TRD2(i,r)*ldf,n2);
				addcoef(M1P(j-1)*sin(PI*(2*j-1)/POW2(r+1)),tmp,ff,n2);
				soltri(j,r,shift,tmp,dd,n2);
				add(nu,tmp,n2);
			}

      		cpycoef(1.0/POW2(r),f+TRD2(i,r)*ldf,nu,n2);

		}
	}

}

void base2solver(_var *f, _var *dd, _var shift, int k1, int n2, int ldf) {
	_base2solver(f, dd, shift, k1, n2, ldf, 1);
}


void base2solver3d(_var *f, _var *dd, int k1, int k2, int n3, int ldf) {

	int n2 = POW2(k2)-1;

	_var *buff = malloc(THREADS*3*n2*ldf*sizeof(_var));

	/* Reduction stage */
	int r;
	for(r = 1; r <= k1-1 && POW2(k1-r)-1 >= THREADS; r++) {
		int i;		

		#pragma omp parallel for num_threads(THREADS)
		for(i = 1; i <= POW2(k1-r)-1; i++) {

			_var *nu  = buff + (omp_get_thread_num()*3+0)*n2*ldf;
			_var *ff  = buff + (omp_get_thread_num()*3+1)*n2*ldf;
			_var *tmp = buff + (omp_get_thread_num()*3+2)*n2*ldf;

			zeros(nu,n2*ldf);

			cpy(ff,f+TRD2(2*i-1,r-1)*n2*ldf,n2*ldf);
			add(ff,f+TRD2(2*i+1,r-1)*n2*ldf,n2*ldf);

			int j;
			for(j = 1; j <= POW2(r-1); j++) {
				cpycoef(M1P(j-1)*sin(PI*(2*j-1)/POW2(r)),tmp,ff,n2*ldf);
				_base2solver(tmp, dd, -2*cos(PI*(2*j-1)/POW2(r)), k2, n3, ldf, 0);
				add(nu,tmp,n2*ldf);
			}
      
			addcoef(1.0/POW2(r-1),f+TRD2(i,r)*n2*ldf,nu,n2*ldf);	  
		}	
	}

	for(; r <= k1-1; r++) {

		int i;		
		for(i = 1; i <= POW2(k1-r)-1; i++) {

			_var *ff  = buff;

			cpy(ff,f+TRD2(2*i-1,r-1)*n2*ldf,n2*ldf);
			add(ff,f+TRD2(2*i+1,r-1)*n2*ldf,n2*ldf);

			#pragma omp parallel num_threads(THREADS)
			{
				_var *nu  = buff + (omp_get_thread_num()*2+1)*n2*ldf;
				_var *tmp = buff + (omp_get_thread_num()*2+2)*n2*ldf;

				zeros(nu,n2*ldf);

				int part_size = POW2(r-1) / THREADS;
				int begin = omp_get_thread_num()*part_size + 1;
				int end = begin + part_size;

				int j;
				for(j = begin; j < end; j++) {
					cpycoef(M1P(j-1)*sin(PI*(2*j-1)/POW2(r)),tmp,ff,n2*ldf);
					_base2solver(tmp, dd, -2*cos(PI*(2*j-1)/POW2(r)), k2, n3, ldf, 0);
					add(nu,tmp,n2*ldf);
				}

    			#pragma omp critical
				addcoef(1.0/POW2(r-1),f+TRD2(i,r)*n2*ldf,nu,n2*ldf);	  
			}
		}	
	}

  	/* Back substitution stage */
	for(r = k1-1; r >= 0 && POW2(k1-r-1) < THREADS; r--) {

		int i;
		for(i = 1; i <= POW2(k1-r)-1; i += 2) {

			_var *ff  = buff;
      
			zeros(ff,n2*ldf);
			if(i > 1)
				add(ff,f+TRD2((i-1)/2,r+1)*n2*ldf,n2*ldf);    
			if(i < POW2(k1-r)-1)
         		add(ff,f+TRD2((i-1)/2+1,r+1)*n2*ldf,n2*ldf);

			_var *o_f  = buff + n2*ldf;
			cpy(o_f,f+TRD2(i,r)*n2*ldf,n2*ldf);
			zeros(f+TRD2(i,r)*n2*ldf,n2*ldf);

			#pragma omp parallel num_threads(THREADS)
			{
				_var *nu  = buff + (omp_get_thread_num()*2+2)*n2*ldf;
				_var *tmp = buff + (omp_get_thread_num()*2+3)*n2*ldf;
				zeros(nu,n2*ldf);

				int part_size = POW2(r) / THREADS;
				int begin = omp_get_thread_num()*part_size + 1;
				int end = begin + part_size;

				int j;
				for(j = begin; j < end; j++) {
					cpy(tmp,o_f,n2*ldf);
					addcoef(M1P(j-1)*sin(PI*(2*j-1)/POW2(r+1)),tmp,ff,n2*ldf);
					_base2solver(tmp, dd, -2*cos(PI*(2*j-1)/POW2(r+1)), k2, n3, ldf, 0);
					add(nu,tmp,n2*ldf);
				}

				#pragma omp critical
      			addcoef(1.0/POW2(r),f+TRD2(i,r)*n2*ldf,nu,n2*ldf);
			}
		}
	}

	for(; r >= 0; r--) {

		int i;
		#pragma omp parallel for private(i) num_threads(THREADS)
		for(i = 1; i <= POW2(k1-r)-1; i += 2) {

			_var *nu  = buff + (omp_get_thread_num()*3+0)*n2*ldf;
			_var *ff  = buff + (omp_get_thread_num()*3+1)*n2*ldf;
			_var *tmp = buff + (omp_get_thread_num()*3+2)*n2*ldf;

			zeros(nu,n2*ldf);
			zeros(ff,n2*ldf);

			if(i > 1)
				add(ff,f+TRD2((i-1)/2,r+1)*n2*ldf,n2*ldf);    
			if(i < POW2(k1-r)-1)
         		add(ff,f+TRD2((i-1)/2+1,r+1)*n2*ldf,n2*ldf);

			int j;
			for(j = 1; j <= POW2(r); j++) {
				cpy(tmp,f+TRD2(i,r)*n2*ldf,n2*ldf);
				addcoef(M1P(j-1)*sin(PI*(2*j-1)/POW2(r+1)),tmp,ff,n2*ldf);
				_base2solver(tmp, dd, -2*cos(PI*(2*j-1)/POW2(r+1)), k2, n3, ldf, 0);
				add(nu,tmp,n2*ldf);
			}

      		cpycoef(1.0/POW2(r),f+TRD2(i,r)*n2*ldf,nu,n2*ldf);

		}
	}

	free(buff);

}

