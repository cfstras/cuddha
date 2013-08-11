#include <iostream>
#include <sstream>
#include <boost/format.hpp>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#include <GLFW/glfw3.h>

using namespace std;
using namespace boost;


static const int WORK_SIZE = 256;
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

__device__ double bbp(unsigned int k) {
	double s = 4 / (double)(8*k+1);
	s -= 2 / (double)(8*k+4);
	s -= 1 / (double)(8*k+5);
	s -= 1 / (double)(8*k+6);
	s /= (double)(powf(16,k));
	return s;
}

/**
 * CUDA kernel function that reverses the order of bits in each element of the array.
 */
__global__ void bbp(void *in, void *out) {
	unsigned int *indata = (unsigned int*) in;
	double *outdata = (double*) out;
	outdata[threadIdx.x] = bbp(indata[threadIdx.x]);
}

boolean renderImage();
int doSome();

GLFWwindow* window;
GLuint imageGLName;
cudaGraphicsResource* imageCUDAName;

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(void) {

	//init OGL

	if(!glfwInint()) return -1;

	window = glfwCreateWindow(imgWidth, imgHeight, "Cuddha", NULL, NULL);
	if(!window) {
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	cudaGLSetGLDevice(0);
	glGenTextures(1,&imageGLName);
	glBindTexture(GL_TEXTURE_2D, imageGLName);
	unsigned int imgSize = imgWidth * imgHeight * 4 * sizeof(uint8_t);
	glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8, imgWidth, imgHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glBindTexture(GL_TEXTURE_2D, 0);

	cuCheckErr(cudaGraphicsGLRegisterImage(&imageCUDAName, imageGLName,
			cudaGraphicsRegisterFlagsWriteDiscard));

	/*cuCheckErr(cudaMalloc((void**) &hostindata, sizeof(int) * WORK_SIZE));
	cuCheckErr(
			cudaMemcpy(hostindata, idata, sizeof(int) * WORK_SIZE, cudaMemcpyHostToDevice));*/


	bbp<<<1, WORK_SIZE>>>(hostindata,hostoutdata);

	cuCheckErr(cudaThreadSynchronize());	// Wait for the GPU launched work to complete
	cuCheckErr(cudaGetLastError());


	cout << "summing up" << endl;

	cuCheckErr(cudaFree((void*) hostoutdata));
	cuCheckErr(cudaFree((void*) hostindata));
	cuCheckErr(cudaDeviceReset());

	return 0;
}

int doSome() {
	//TODO file:///usr/local/cuda-5.5/doc/html/cuda-c-programming-guide/index.html#opengl-interoperability

	cudaGraphicsMapResources(1, &imageCUDAName, 0);
	uint8_t* dataPtr;
	size_t dataLen;
	cudaGraphicsResourceGetMappedPointer(&dataPtr, &dataLen, imageCUDAName);

	// do CUDA work

	// unmap
	cudaGraphicsUnmapResources(1, &imageCUDAName, 0);

	// render
	renderImage();
}

boolean renderImage() {

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
