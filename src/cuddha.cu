#include <iostream>
#include <sstream>
#include <boost/format.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#include <GL/glew.h>
#include <cuda_gl_interop.h>

#include <GLFW/glfw3.h>

using namespace std;
using namespace boost;

static const int imgWidth = 800;
static const int imgHeight = 600;

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define cuCheckErr(value) {													\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		cerr << "Error "<< cudaGetErrorString(_m_cudaStat) << " at line " <<\
			__LINE__ << " in file " << __FILE__ << endl;					\
		exit(1);															\
	} }

__device__ float function(/*args*/) {
	return 0;
}

__global__ void cuBrot(uint8_t* dataPtr) {
	//threadIdx.xyz
	dataPtr[(threadIdx.x + threadIdx.y * imgWidth) * 4 + 0] /*Red*/   = (uint8_t)((float)threadIdx.x / imgWidth * 255);
	dataPtr[(threadIdx.x + threadIdx.y * imgWidth) * 4 + 1] /*Green*/ = (uint8_t)((float)threadIdx.y / imgHeight * 255);
	dataPtr[(threadIdx.x + threadIdx.y * imgWidth) * 4 + 2] /*Blue*/  = 0;
	dataPtr[(threadIdx.x + threadIdx.y * imgWidth) * 4 + 3] /*Alpha*/ = 255;
}

bool renderImage();
int doSome();
void glfwError(int error, const char* description);
void freeStuff();
void destroyStuff();


GLFWwindow* window;
GLuint imageGLName;
cudaGraphicsResource* imageCUDAName;

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(void) {

	//init OGL
	if(!glewInit()) return -1;
	if(!glfwInit()) return -1;
	glfwSetErrorCallback(glfwError);

	window = glfwCreateWindow(imgWidth, imgHeight, "Cuddha", NULL, NULL);
	if(!window) {
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	cudaGLSetGLDevice(0);
	glGenTextures(1,&imageGLName);
	glBindTexture(GL_TEXTURE_2D, imageGLName);
	//unsigned int imgSize = imgWidth * imgHeight * 4 * sizeof(uint8_t);
	glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8, imgWidth, imgHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glBindTexture(GL_TEXTURE_2D, 0);

	cuCheckErr(cudaGraphicsGLRegisterImage(&imageCUDAName, imageGLName, GL_TEXTURE_2D,
			cudaGraphicsRegisterFlagsWriteDiscard));

	/*cuCheckErr(cudaMalloc((void**) &hostindata, sizeof(int) * WORK_SIZE));
	cuCheckErr(
			cudaMemcpy(hostindata, idata, sizeof(int) * WORK_SIZE, cudaMemcpyHostToDevice));*/

	for (int numDone = 0; numDone != -1; numDone = doSome());

	cuCheckErr(cudaDeviceReset());

	return 0;
}

int doSome() {
	//TODO file:///usr/local/cuda-5.5/doc/html/cuda-c-programming-guide/index.html#opengl-interoperability

	cudaGraphicsMapResources(1, &imageCUDAName, 0);
	void* dataPtr;
	size_t dataLen;
	cudaGraphicsResourceGetMappedPointer(&dataPtr, &dataLen, imageCUDAName);

	if(dataLen != (imgWidth*imgHeight*4/*RGBA*/*sizeof(char))) {
		cerr << "Error: data length expected " << (imgWidth*imgHeight*1*sizeof(char)) << ", got " << dataLen << endl;
		return -1;
	}

	int workDone = 0;

	// do CUDA work
	cuBrot<<<1, dim3(imgWidth, imgHeight, 1)>>>((uint8_t*)dataPtr);
	workDone++;

	cuCheckErr(cudaThreadSynchronize());	// Wait for the GPU launched work to complete
	cuCheckErr(cudaGetLastError());

	// unmap
	cudaGraphicsUnmapResources(1, &imageCUDAName, 0);

	// render
	if(!renderImage()) {
		//TODO save?
		freeStuff();
		destroyStuff();
		return -1;
	}
	return workDone;
}

bool renderImage() {

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glBindTexture(GL_TEXTURE_2D, imageGLName);
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f);
	glVertex2f(0.0f, 0.0f);
	glTexCoord2f(1.0f, 0.0f);
	glVertex2f(1.0f, 0.0f);
	glTexCoord2f(1.0f, 1.0f);
	glVertex2f(1.0f, 1.0f);
	glTexCoord2f(0.0f, 1.0f);
	glVertex2f(0.0f, 1.0f);
	glEnd();
	glBindTexture(GL_TEXTURE_2D, 0);

	glfwSwapBuffers(window);
	glfwPollEvents();
	return !glfwWindowShouldClose(window);
}

void glfwError(int error, const char* description) {
    cerr << "GLFW Error " << error << description << endl;
}

void freeStuff() {
	cudaGraphicsUnregisterResource(imageCUDAName);
	glDeleteTextures(1,&imageGLName);
}

void destroyStuff() {
	glfwTerminate();
}

