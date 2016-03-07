# BCR-GPU
A OpenCL-based block cyclic reduction solver.

Mirko Myllykoski, Tuomo Rossi, Jari Toivanen: Fast Poisson Solvers for Graphics Processing Units, In Applied Parallel and Scientific Computing, Lecture Notes in Computer Science, Vol. 7782, Manninen P., Öster P. (eds), Springer Berlin Heidelberg: Berlin, Germany, 2013, pp. 265-279, doi: 10.1007/978-3-642-36803-5_19

Download article (postprint): http://urn.fi/URN:NBN:fi:jyu-201509022794

**CAUTION**: This implementation has not been tested since 2012. 

## API (see ```b4pfm.h```)

### Debug flags:
```
#define B4PFM_DEBUG_NONE		0 // No debug messages
#define B4PFM_DEBUG_NORMAL		1 // Some debug messages
#define B4PFM_DEBUG_FULL		2 // All debug messages
#define B4PFM_DEBUG_TIMING		3 // Timing information (benchmarking)
```

### 2D problems:
#### Initialize a 2D solver:
```
b4pfm2D b4pfm2D_init_solver(cl_context context, b4pfm2D_params *opt_params, int k1, int k2, int ldf, int prec, int force_radix2, int debug, int *error);
```
 * ```context```: OpenCL context (0 => Use the first available OpenCL device)
 * ```opt_params```: Parameters, see ```b4pfm2D_default_opts``` and ```b4pfm2D_auto_opts``` (0 => Use default parameters)
 * Problem size: (2^```k1```-1) * (2^```k2```-1) unknowns
 * ```ldf```: Padding, i.e., 2^```k2```-1 <= ```ldf```
 * Precision modes (```prec```): 0 => single precision, 1 => double precision
 * ```force_radix2```: Use only radix-2 solver (benchmarking)
 * ```debug```: Debug flag
 * ```error```: Error flag
 * Return value: Solver handle

#### Run a 2D solver:
```
int b4pfm2D_run_solver(b4pfm2D solver, cl_command_queue queue, cl_mem f, cl_mem tmp, int debug);
```
 * ```solver```: Solver handle
 * ```queue```: OpenCL command queue (0 => Create a new command queue)
 * ```f```: Right-hand side vector
 * ```tmp```: Temporary buffer (0 => Allocate automatically)
 * Return value: Error flag

#### Transfer a float vector and run a 2D solver:
```
int b4pfm2D_load_and_run_solver_float(b4pfm2D solver, float *f, int debug);
```

#### Transfer a double vector and run a 2D solver:
```
int b4pfm2D_load_and_run_solver_double(b4pfm2D solver, double *f, int debug);
```

#### Get default b4pfm2D_params:
```
int b4pfm2D_default_opts(cl_context context, b4pfm2D_params *opt_params, int k1, int k2, int ldf, int prec, int debug);
```

#### Get optimized b4pfm2D_params:
```
int b4pfm2D_auto_opts(cl_context context, cl_mem tmp1, cl_mem tmp2, b4pfm2D_params *opt_params, int force_radix2, int k1, int k2, int ldf, int prec, int debug);
```
#### Release solver:
```
int b4pfm2D_free_solver(b4pfm2D solver);
```

### 3D problems:
```
b4pfm3D b4pfm3D_init_solver(cl_context context, b4pfm3D_params *opt_params, int k1, int k2, int k3, int ldf, int prec, int force_radix2, int debug, int *error);
```
```
int b4pfm3D_run_solver(b4pfm3D solver, cl_command_queue queue, cl_mem f, cl_mem tmp1, cl_mem tmp2, int debug);
```
```
int b4pfm3D_load_and_run_solver_float(b4pfm3D solver, float *f, int debug);
```
```
int b4pfm3D_load_and_run_solver_double(b4pfm3D solver, double *f, int debug);
```
```
int b4pfm3D_default_opts(cl_context context, b4pfm3D_params *opt_params, int k1, int k2, int k3, int ldf, int prec, int debug);
```
```
int b4pfm3D_auto_opts(cl_context context, cl_mem tmp1, cl_mem tmp2, cl_mem tmp3, b4pfm3D_params *opt_params, int force_radix2, int k1, int k2, int k3, int ldf, int prec, int debug);
```
```
int b4pfm3D_free_solver(b4pfm3D solver);
```

## Example:
```
#define K1 10
#define K2 10
#define N1 ((1<<K1)-1)
#define N2 ((1<<K2)-1)
#define LDF (1<<K2) // = 2^K2
#define DOUBLE 1
#define FORCE_RADIX2 0

int err;

double *f = malloc(N1*LDF);
for(int i = 0; i < N1; i++)
      for(int j = 0; j < N2; j++)
          f[i*LDF+j] = get_right_hand_size(i, j);

// Get optimized parameters. Will use the first available OpenCL device 
// and allocates the tmp1 and tmp2 buffers automatically.
b4pfm2D_params params;
err = b4pfm2D_auto_opts(
      0, 0, 0, &params, FORCE_RADIX2, K1, K2, LDF, DOUBLE, B4PFM_DEBUG_NORMAL);
if(err != B4PFM_OK) 
      error();

// Initialize the solver using the optimized parameters. Will use the 
// first available OpenCL device.
b4pfm2D solver = b4pfm2D_init_solver(
      0, &params, K1, K2, LDF, DOUBLE, FORCE_RADIX2, B4PFM_DEBUG_NORMAL, &err);
if(err != B4PFM_OK) 
      error();

// Run solver. 
err = b4pfm2D_load_and_run_solver_double(solver, f, B4PFM_DEBUG_NORMAL);
if(err != B4PFM_OK) 
      error();

// Free allocated data
b4pfm2D_free_solver(solver);

```
