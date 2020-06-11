/*
Collatz code for CS 4380 / CS 5351

Copyright (c) 2019 Texas State University. All rights reserved.

Redistribution in source or binary form, with or without modification,
is *not* permitted. Use in source and binary forms, with or without
modification, is only permitted for academic use in CS 4380 or CS 5351
at Texas State University.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Author: Martin Burtscher
*/

//author:nathan stanton

#include <cstdio>
#include <algorithm>
#include <sys/time.h>
#include <mpi.h>

void GPU_Init();
void GPU_Exec(const long start, const long stop);
int GPU_Fini();



static int collatz(const long start, const long stop)
{


  int maxlen = 0;

  // todo: OpenMP code with default(none), a reduction, and a cyclic schedule (assume start to be odd) based on Project 4
  #pragma omp parallel for default(none) reduction(max:maxlen) schedule(static, 1)
  for (long i = start ; i<= stop;i+= 2) {
      long val = i;
      int len = 1;
      while (val != 1) {
        len++;
        if ((val % 2) == 0) {
          val = val / 2;  // even
        } else {
          val = 3 * val + 1;  // odd
        }
      }
      maxlen = std::max(maxlen, len);
    }
  return maxlen;
}

int main(int argc, char *argv[])
{

  // set up MPI
  int comm_sz, my_rank;
  MPI_Init(NULL, NULL);//from prog2
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  // set up global max lens
  int maxlen_global_cpu = 0;
  int maxlen_global_gpu = 0;


 if (my_rank == 0)  {printf("Collatz v1.2\n");} //check rank b4 print

  // check command line
  if (argc != 3) {fprintf(stderr, "USAGE: %s upper_bound cpu_percentage\n", argv[0]); exit(-1);}
  const long upper = atol(argv[1]);
  if (upper < 5) {fprintf(stderr, "ERROR: upper_bound must be at least 5\n"); exit(-1);}
  if ((upper % 2) != 1) {fprintf(stderr, "ERROR: upper_bound must be an odd number\n"); exit(-1);}
  const int percentage = atof(argv[2]);
  if ((percentage < 0) || (percentage > 100)) {fprintf(stderr, "ERROR: cpu_percentage must be between 0 and 100\n"); exit(-1);}

  const long cpu_start = (my_rank * upper / comm_sz) | 1;
  const long gpu_stop = ((my_rank + 1) * upper / comm_sz) | 1;
  const long my_range = gpu_stop - cpu_start + 1;
  const long cpu_stop = (cpu_start + my_range * percentage / 100) & ~1LL;
  const long gpu_start = cpu_stop | 1;

if (my_rank == 0) //check rank b4 print
{
  printf("upper bound: %ld\n", upper);
  printf("CPU percentage: %d\n", percentage);
}

  GPU_Init();

  // start time
  timeval start, end;
  MPI_Barrier(MPI_COMM_WORLD); //barrier before timer
  gettimeofday(&start, NULL);

  // execute timed code
  GPU_Exec(gpu_start, gpu_stop);
  const int cpu_maxlen = collatz(cpu_start, cpu_stop);
  const int gpu_maxlen = GPU_Fini();


  //reduce results to process 0
  MPI_Reduce(&cpu_maxlen, &maxlen_global_cpu, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD); //reduce cpu
  MPI_Reduce(&gpu_maxlen, &maxlen_global_gpu, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD); //reduce gpu

  const int maxlen = std::max(maxlen_global_cpu, maxlen_global_gpu); //check maxlens after reduction


  // end time
  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  if (my_rank == 0){printf("compute time: %.4f s\n", runtime);} //check rank b4 print

  // print result
  if (my_rank == 0){printf("longest sequence: %d elements\n", maxlen);} //check rank b4 print

  return 0;
}
