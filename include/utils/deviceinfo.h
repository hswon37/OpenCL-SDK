
#ifndef HISTOGRAM_H_
#define HISTOGRAM_H_


#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

#include <windows.h>
#include <math.h>


LARGE_INTEGER beginClock, endClock, clockFreq;
LARGE_INTEGER tot_beginClock, tot_endClock, tot_clockFreq;

cl_mem d_inputArray_A;
cl_mem d_inputArray_B;
cl_mem d_outputArray;

cl_platform_id platform;

cl_context context;
cl_device_id device;
cl_command_queue queue;

cl_program program;

cl_kernel simpleKernel;


#endif /* #ifndef HISTOGRAM_H_ */
