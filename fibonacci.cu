#include <cstdint>
#include <iostream>
#include "Xdevice/runtime.hpp"

#if (!defined TARGET_CUDA && !defined TARGET_HIP)
#  error "Must define either TARGET_CUDA or TARGET_HIP."
#endif

#ifndef FIB
# define FIB 64
#endif
#define HOSTNAME_SIZE 80

__device__ __forceinline__ void fibonacci(float * __restrict__ buffer)
{
  /* Computes the first FIB fibonacci numbers */
  for (int idx = 0;  idx < FIB; idx++)
  {
    if (idx < 2)
    {
      buffer[idx] = idx;
    }
    else
    {
      buffer[idx] = buffer[idx-1]+buffer[idx-2];
    }
  }
}


__global__ void timed_fibonacci(float * __restrict__ buffer, uint64_t * timer)
{
  /* Count the number of clocks it takes to run the fibonacci function */

  // timers
  uint64_t start, end;

  // start the cycle count
  start = XClock64();

  // Do some work
  fibonacci(buffer);

  // end cycle count
  end = XClock64();

  // Diff the timers and write the result back
  end -= start;
  (*timer) = uint64_t(end);
}


int time_fib()
{
  float * d_buffer;
  uint64_t * d_timer;
  uint64_t h_timer;

  // Allocate the device buffers
  XMalloc((void**)&d_buffer, sizeof(float)*FIB);
  XMalloc((void**)&d_timer, sizeof(uint64_t));

  // calculate the Fib numbers
  timed_fibonacci<<<1,1>>>(d_buffer, d_timer);
  XDeviceSynchronize();

  // Copy the timing data back to the host
  XMemcpy(&h_timer, d_timer, sizeof(uint64_t), XMemcpyDeviceToHost);

  // Free the buffers
  XFree(d_buffer);
  XFree(d_timer);

  return (int)h_timer;
}



int main(int argc, char ** argv)
{
  // Get host name
  char nid_name[HOSTNAME_SIZE];
  gethostname(nid_name, HOSTNAME_SIZE);

  // Make sure we've got devices aboard
  int num_devices;
  XGetDeviceCount(num_devices);
  if (num_devices == 0)
  {
    std::cout << "No devices found on host " << nid_name << std::endl;
    return 1;
  }
  else
  {
    printf("[%s] Found %d device(s).\n", nid_name, num_devices);
  }

  // Time the Fib list creation for each device
  for (int i = 0; i < num_devices; i++)
  {
    printf("[device %d] Clocks taken to compute the first %d Fibonacci numbers: %d\n", i, FIB, time_fib());
  }

  return 0;
}
