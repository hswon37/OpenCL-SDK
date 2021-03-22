#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <Windows.h>
#include <string.h>
#include <tchar.h>
#include <stdbool.h>

#include "deviceInfo.h"
#include <chrono>
using namespace std;
// kernel을 읽어서 char pointer생성
char *readSource(char *kernelPath)
{

    cl_int status;
    FILE *fp;
    char *source;
    long int size;

    printf("Program file is: %s\n", kernelPath);

    fp = fopen(kernelPath, "rb");
    if (!fp)
    {
        printf("Could not open kernel file\n");
        exit(-1);
    }
    status = fseek(fp, 0, SEEK_END);
    if (status != 0)
    {
        printf("Error seeking to end of file\n");
        exit(-1);
    }
    size = ftell(fp);
    if (size < 0)
    {
        printf("Error getting file position\n");
        exit(-1);
    }

    rewind(fp);

    source = (char *)malloc(size + 1);

    int i;
    for (i = 0; i < size + 1; i++)
    {
        source[i] = '\0';
    }

    if (source == NULL)
    {
        printf("Error allocating space for the kernel source\n");
        exit(-1);
    }

    fread(source, 1, size, fp);
    source[size] = '\0';

    return source;
}

//디바이스 init, 커널 생성
void CLInit()
{
    int i, j;
    char *value;
    size_t valueSize;
    cl_uint platformCount; // cl_uint: unsigned int(uint), 부호없는 32비트 정수,
                           // API type for application
    cl_platform_id *platforms;
    cl_uint deviceCount;
    cl_device_id *devices;
    cl_uint maxComputeUnits;

    // get all platforms(opencl서비스를 제공할 수 있는 환경)
    // clGetPlatformIDs(num_entries, platforms, num_platforms)
    clGetPlatformIDs(
        0, NULL,
        &platformCount); //사용 가능한 플랫폼 목록 획득, platformCount변수에
                         //사용가능한 플랫폼 수 저장.
    platforms = (cl_platform_id *)malloc(
        sizeof(cl_platform_id) * platformCount); // platforms 공간 할당
    clGetPlatformIDs(platformCount, platforms,
                     NULL); // 할당된 공간만큼의 플랫폼 불러옴

    for (i = 0; i < platformCount; i++)
    {

        // get all devices(계산을 수행할수있는 유닛의 집합. GPU로 설명하면
        // GPU내의 수많은 코어들의 집합), 플랫폼 받아오는 과정과 동일
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount);
        devices = (cl_device_id *)malloc(sizeof(cl_device_id) * deviceCount);
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices,
                       NULL);

        // for each device print critical attributes
        for (j = 0; j < deviceCount; j++)
        {

            // print device name
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &valueSize);
            value = (char *)malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, valueSize, value, NULL);
            printf("platform %d. Device %d: %s\n", i + 1, j + 1, value);
            free(value);

            // print hardware device version
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, 0, NULL, &valueSize);
            value = (char *)malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, valueSize, value,
                            NULL);
            printf(" %d.%d Hardware version: %s\n", i + 1, 1, value);
            free(value);

            // print software driver version
            clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, 0, NULL, &valueSize);
            value = (char *)malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, valueSize, value,
                            NULL);
            printf(" %d.%d Software version: %s\n", i + 1, 2, value);
            free(value);

            // print c version supported by compiler for device
            clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, 0, NULL,
                            &valueSize);
            value = (char *)malloc(valueSize);
            clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, valueSize,
                            value, NULL);
            printf(" %d.%d OpenCL C version: %s\n", i + 1, 3, value);
            free(value);

            // print parallel compute units
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS,
                            sizeof(maxComputeUnits), &maxComputeUnits, NULL);
            printf(" %d.%d Parallel compute units: %d\n", i + 1, 4,
                   maxComputeUnits);
        }
    }
    int platformNum;
    int deviceNum;
    printf("\n\nSELECT PLATFORM('1' ~ '%d') : ", platformCount);
    scanf("%d", &platformNum);
    printf("\n");
    printf("SELECT DEVICE('1' ~ '%d') : ", deviceCount);
    scanf("%d", &deviceNum);
    printf("\n");
    clGetDeviceIDs(platforms[platformNum - 1], CL_DEVICE_TYPE_ALL, deviceCount,
                   devices, NULL);

    device = devices[deviceNum - 1];

    // create context(CL 커널이 실행되는 환경으로, 동기화와 메모리 관리가 정의.
    // OpenCL디바이스에서 실행할 OpenCL함수들을 포함)
    context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);

    // create command queue(호스트에서 디바이스 별로 생성되는 것으로 하나의
    // 디바이스에서 여러개의 커맨드 큐가 연결 가능. 커맨드 큐를 이용헤 커널을
    // 실행하고 메모리의 매핑과 언매핑, 동기화 등 가능)
    queue = clCreateCommandQueue(context, device, 0, NULL);

    // 텍스트파일로부터 프로그램 읽기
    char *source = readSource("convolution.cl");

    // compile program
    program = clCreateProgramWithSource(context, 1, (const char **)&source,
                                        NULL, NULL);
    cl_int build_status;
    build_status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    //커널 포인터 생성
    simpleKernel = clCreateKernel(program, "simpleKernel", NULL);
}

//버퍼생성 및 write
void bufferWrite()
{
    // 메모리 버퍼 생성 (gpu 메모리 버퍼 포인터생성)
    d_inputArray_A = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    4000 * sizeof(int), NULL, NULL);
    d_inputArray_B = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    4000 * sizeof(int), NULL, NULL);
    d_outputArray = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   4000 * sizeof(int), NULL, NULL);

    // gpu 메모리에 넣을 배열 값
    int inputArray_A[4000];
    int inputArray_B[4000];
    // srand(time(NULL));
    for (int i = 0; i < 4000; i++)
    {
        inputArray_A[i] = rand() % 10;
        inputArray_B[i] = rand() % 10;
    }

    int i, j;
    for (int k = 990; k < 1000; k++)
    { // k+1번째 배열(2)
        printf("Array %dth A : \n", k + 1);
        for (i = 0; i < 2; i++)
        {
            for (j = 0; j < 2; j++)
                printf("%d ", inputArray_A[k * 4 + i * 2 + j]);
            printf("\n");
        }
        printf("\n");
        printf("Array %dth B : \n", k + 1);
        for (i = 0; i < 2; i++)
        {
            for (j = 0; j < 2; j++)
                printf("%d ", inputArray_B[k * 4 + i * 2 + j]);
            printf("\n");
        }
        printf("\n");
    }
    //버퍼에 쓰기
    clEnqueueWriteBuffer(queue, d_inputArray_A, CL_TRUE, 0, 4000 * sizeof(int),
                         inputArray_A, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, d_inputArray_B, CL_TRUE, 0, 4000 * sizeof(int),
                         inputArray_B, 0, NULL, NULL);
}

void runKernel()
{
    int i, j;

    //스레드 생성 개수 결정: 반복횟수만큼 필요한 듯
    int totalWorkItemsX = 2;
    int totalWorkItemsY = 2;
    int totalWorkItemsZ = 1000;

    size_t globalSize[3] = { totalWorkItemsX, totalWorkItemsY,
                             totalWorkItemsZ }; // 한번에 실행할 2차원 커널 표현

    float *minVal, *maxVal; //?


    // 커널 매개변수 설정
    clSetKernelArg(simpleKernel, 0, sizeof(cl_mem), &d_inputArray_A);
    clSetKernelArg(simpleKernel, 1, sizeof(cl_mem), &d_inputArray_B);
    clSetKernelArg(simpleKernel, 2, sizeof(cl_mem), &d_outputArray);

    // 커널 실행
    clEnqueueNDRangeKernel(queue, simpleKernel, 3, NULL, globalSize, NULL, 0,
                           NULL, NULL);

    // 완료 대기
    clFinish(queue);

    // read 및 결과출력
    int outputArray[4000];
    clEnqueueReadBuffer(queue, d_outputArray, CL_TRUE, 0, 4000 * sizeof(int),
                        outputArray, 0, NULL, NULL);

    for (int k = 990; k < 1000; k++)
    { // k+1번째 배열
        printf("output  %dth: \n", k + 1);
        for (i = 0; i < 2; i++)
        {
            for (j = 0; j < 2; j++)
                printf("%d ", outputArray[k * 4 + i * 2 + j]);
            printf("\n");
        }
        printf("\n");
    }
}

void Release()
{
    // 릴리즈
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

int main(int argc, char **argv)
{


    QueryPerformanceFrequency(&tot_clockFreq);

    // OpenCL 디바이스, 커널 셋업
    CLInit();

    QueryPerformanceCounter(&tot_beginClock); //시간측정 시작

    //디바이스 쪽 버퍼 생성 및 write
    bufferWrite();

    //커널 실행
    runKernel();

    QueryPerformanceCounter(&tot_endClock);
    double totalTime = (double)(tot_endClock.QuadPart - tot_beginClock.QuadPart)
        / tot_clockFreq.QuadPart;
    printf("Total processing Time : %f ms\n", totalTime * 1000);

    system("pause");

    Release();
    return 0;
}
