
// TODO: Add OpenCL kernel code here.
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

__kernel
void simpleKernel(__global int* inputArray_A,
	__global int* inputArray_B,
	__global int* outputArray
)
{ // size_t globalSize[3] = {2, 2, 1000 }; //내필기
	uint dstIndex = get_global_id(2) * 4 + get_global_id(0) * 2 + get_global_id(1);
	uint globalRow = get_global_id(0); // j, 행
	uint globalCol = get_global_id(1); // k, 열
	uint globalArr = get_global_id(2); // i, 몇번째 배열
	
	//printf("%d %d %d %d\n", dstIndex, globalArr, globalRow, globalCol);
	outputArray[dstIndex] = (inputArray_A[globalArr*4 + globalRow*2 + 0]
						* inputArray_B[globalArr*4 + 0*2 + globalCol])
						+ (inputArray_A[globalArr*4 + globalRow*2 + 1]
						* inputArray_B[globalArr*4 + 1*2 + globalCol]);
}