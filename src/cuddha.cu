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

	cuCheckErr(cudaGraphicsGLRegisterImage(&imageCUDAName, imageGLName,
			cudaGraphicsRegisterFlagsWriteDiscard));

	for (i = 0; i < WORK_SIZE; i++)
		idata[i] = (unsigned int) i;

	cuCheckErr(cudaMalloc((void**) &hostindata, sizeof(int) * WORK_SIZE));
	cuCheckErr(
			cudaMemcpy(hostindata, idata, sizeof(int) * WORK_SIZE, cudaMemcpyHostToDevice));

	cuCheckErr(cudaMalloc((void**) &hostoutdata, sizeof(double) * WORK_SIZE));

	bbp<<<1, WORK_SIZE>>>(hostindata,hostoutdata);

	cuCheckErr(cudaThreadSynchronize());	// Wait for the GPU launched work to complete
	cuCheckErr(cudaGetLastError());
	cuCheckErr(cudaMemcpy(odata, hostoutdata, sizeof(double) * WORK_SIZE, cudaMemcpyDeviceToHost));
	cuCheckErr(cudaMemcpy(idata, hostindata, sizeof(int) * WORK_SIZE, cudaMemcpyDeviceToHost));

	cout << "summing up" << endl;

	double sum = 0;
	for (i = 0; i < WORK_SIZE; i++)
		sum += odata[i];

	string calcpi = boost::str(boost::format("%1.100f") % sum);

	string realpi = "3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679";
	int len = realpi.length();
	int len2 = calcpi.length();
	int signif = -1;

	for(i=0; i<len && i<len2; i++) {
		if(realpi[i] != calcpi[i]) {
			signif = i;
			i = len;
		}
	}

	cout << "significant numbers: "<< signif << endl <<
			"pi is approx. "<< format("%1.50f") % sum << endl;
	cuCheckErr(cudaFree((void*) hostoutdata));
	cuCheckErr(cudaFree((void*) hostindata));
	cuCheckErr(cudaDeviceReset());

	return 0;
}

boolean renderImage() {

	glfwSwapBuffers(window);
	glfwPollEvents();
	return !glfwWindowShouldClose(window);
}
