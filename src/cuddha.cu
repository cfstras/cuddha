#include <iostream>
#include <sstream>
#include <vector>
#include <boost/format.hpp>
#include <cstring>
#include <cstdio>

#include "bmp.h"

#ifdef __linux__
#include <unistd.h>
inline void sleep(int usecs) {
	usleep(usecs);
}
#endif
#ifdef _WIN32
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
inline void sleep(int usecs) {
	Sleep(usecs/1000);
}
#endif

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <GL/gl.h>
#include <GL/glu.h>

#include <curand_kernel.h>

#define BMP_QUICKSAVE
#define OPTIMISE

using namespace std;
using namespace boost;

static const int imgWidth = 1024;
static const int imgHeight = 1024;
static const int maxIterations = 10*1;
static const int minDrawIterations = 1;
static const int imgSize = imgWidth * imgHeight;

// total job number: XYRES*XYRES*XYRESMULT*XYRESMULT
// job resolution: XYRES*XYRESMULT
static int XYRES = 16;
static int XYRESMULT = 32;

static const int nums[] = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,
59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,
167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,257,263,269,271,
277,281,283,293,307,311,313,317,331,337,347,349,353,359,367,373,379,383,389,397,
401,409,419,421,431,433,439,443,449,457,461,463,467,479,487,491,499,503,509,521,
523,541,547,557,563,569,571,577,587,593,599,601,607,613,617,619,631,641,643,647,
653,659,661,673,677,683,691,701,709,719,727,733,739,743,751,757,761,769,773,787,
797,809,811,821,823,827,829,839,853,857,859,863,877,881,883,887,907,911,919,929,
937,941,947,953,967,971,977,983,991,997,1009,1013,1019,1021,1031,1033,1039,1049,
1051,1061,1063,1069,1087,1091,1093,1097,1103,1109,1117,1123,1129,1151,1153,1163,
1171,1181,1187,1193,1201,1213,1217,1223,1229,1231,1237,1249,1259,1277,1279,1283,
1289,1291}; // hurr durr
const std::vector<int> prime_numbers(nums, nums + 210);

#define out(dim) "[x="<<dim.x<<",y="<<dim.y<<",z="<<dim.z<<"]"
#define UINT64_MAX ((uint64_t)(1) << 63)
/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define cuCheckErr(value) {													\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		cerr << "cuda Error " <<_m_cudaStat << ": "							\
			<< cudaGetErrorString(_m_cudaStat) << " at line " <<			\
			__LINE__ << " in file " << __FILE__ << endl;					\
		exit(1);															\
	} }

/**
 * GL Error checking
 */
static inline void glCheck() {
	GLenum err = glGetError();
	if (err != GL_NO_ERROR) {
		cerr << "GL Error: " << glewGetErrorString(err) << endl;
		exit(2);
	}
}

__device__ uint64_t atomicAdd(uint64_t* address, uint64_t val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    int i = 0;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        (val + assumed));
        if(i++ > 32) {
        	//printf("exposing failed\n");
        	break;
        }

    } while (assumed != old);
    return old;
}

template<class T>
__device__ inline int coordToIndex(T x, T y) {
	//double ix = 0.3 * (x+0.5) + (double)imgWidth / 2;
	//double iy = 0.3 * y + (double)imgHeight / 2;
	int ix = (int)(imgWidth  * ((x + 2.0) / 3.0));
	int iy = (int)(imgHeight * ((y + 1.5) / 3.0));

	if(ix >= imgWidth || iy >= imgHeight || ix < 0 || iy < 0) {
		return -1;
	}
	int basePos = iy + ix * imgWidth; // swap x&y
	if (basePos >= imgSize) {
		printf("droppingi %3.4f, %3.4f\n", ix, iy);
		return -1;
	}
	return basePos;
}

__device__ inline void getRand(curandState* state, int serial) {
	long xInd = (threadIdx.x + blockDim.x * blockIdx.x);
	long yInd = (threadIdx.y + blockDim.y * blockIdx.y) * blockDim.x;
	curand_init(serial*31 + xInd*2 + yInd*4, 0, 0, state);
}

#pragma unroll
template<class T>
__global__ void cuBrot(uint64_t* exposure, int maxIterations,
		uint currPrime, uint primePosX, uint primePosY,
		int serial) {
	curandState state;
	getRand(&state, serial);
	// [0,1]
	double xInd = (threadIdx.x + blockDim.x * blockIdx.x) / (double)(blockDim.x * gridDim.x);
	double yInd = (threadIdx.y + blockDim.y * blockIdx.y) / (double)(blockDim.y * gridDim.y);
	double xIndn = (threadIdx.x+1 + blockDim.x * blockIdx.x) / (double)(blockDim.x * gridDim.x);
	double yIndn = (threadIdx.y+1 + blockDim.y * blockIdx.y) / (double)(blockDim.y * gridDim.y);

	T x, y, xx, yy, xC, yC, xCn, yCn, xDiff, yDiff;

	//xC = xInd * 6 - 3.0;  // range -2.0, 1.0 //old, now both -3,3
    //yC = yInd * 6 - 3.0;  // range -1.5, 1.5

	xC = xInd * 3 - 2;
	yC = yInd * 3 - 1.5;

	//TODO do this prime subiteration from the middle instead of top-left

	//TODO performance hint: for found orbit, mirror on x axis and try again

	xCn = (xIndn) * 3 - 2;
	yCn = (yIndn) * 3 - 1.5;
	xDiff = xCn - xC;
	yDiff = yCn - yC;

	xC += curand_uniform(&state) * xDiff;
	yC += curand_uniform(&state) * yDiff;
	//xC += ( primePosX / (float)currPrime ) * xDiff;
	//yC += ( primePosY / (float)currPrime ) * yDiff;

	//printf("xC %2.5f yC %2.5f\n", xC, yC);

	// yep
	int basePos = coordToIndex<T>(xC, yC);
	if (basePos == -1) return;
	atomicAdd(&exposure[basePos], 1);
	return;

#ifdef OPTIMISE
                        if (
                           (xC >  -1.2 && xC <=  -1.1 && yC >  -0.1 && yC < 0.1)
                        || (xC >  -1.1 && xC <=  -0.9 && yC >  -0.2 && yC < 0.2)
                        || (xC >  -0.9 && xC <=  -0.8 && yC >  -0.1 && yC < 0.1)
                        || (xC > -0.69 && xC <= -0.61 && yC >  -0.2 && yC < 0.2)
                        || (xC > -0.61 && xC <=  -0.5 && yC > -0.37 && yC < 0.37)
                        || (xC >  -0.5 && xC <= -0.39 && yC > -0.48 && yC < 0.48)
                        || (xC > -0.39 && xC <=  0.14 && yC > -0.55 && yC < 0.55)
                        || (xC >  0.14 && xC <   0.29 && yC > -0.42 && yC < -0.07)
                        || (xC >  0.14 && xC <   0.29 && yC >  0.07 && yC < 0.42)
                        ) return;
#endif

	y = x = yy = xx = 0;
	// x0 & y0 = 0, so xx0 = 0, too

	bool out = false;
	int i;
	for(i=1;i <= maxIterations && !out; i++) {
		y = x * y * T(2.0) + yC;
		x = xx - yy + xC ;
		yy = y * y;
		xx = x * x;
		out = xx+yy > T(4);
	}
	maxIterations = i; // only go up there
	if(out) {
		y = x = yy = xx = 0;
		for(int i=1;i <= maxIterations; i++) {
			y = x * y * T(2.0) + yC;
			x = xx - yy + xC ;
			yy = y * y;
			xx = x * x;
			if (i >= minDrawIterations) {
				//double ix = 0.3 * (x+0.5) + (double)imgWidth / 2;
				//double iy = 0.3 * y + (double)imgHeight / 2;
				int basePos = coordToIndex<T>(x, y);
				if (basePos == -1) {
					return;
				}
				//exposure[basePos]=255;// = i;
				atomicAdd(&exposure[basePos], 1);
			}
		}
	}
}

__global__ void cuConvertImage(GLubyte* outImage, int len, uint64_t* exposure, uint64_t maxExp, uint64_t minExp) {
	uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	uint32_t x = idx % imgWidth;
	uint32_t y = idx / imgWidth;

	int basePos = (x+y*imgWidth);
	if(basePos*4+4 > len) {
		printf("x=%d, y=%d Error: usedlen %4d > len %4d\n", x,y, basePos*4, len);
		return;
	}

	uint64_t exp = exposure[basePos];
	exp -= minExp;
	//float expF = ((float)exp) / maxExp;
	float expF = exp / (float)(maxExp - minExp);
	expF = (__logf(expF +( 0.135335 ))+2)/2.1269;

	outImage[(basePos*4) + 0] = (uint8_t)(expF*255);
	outImage[(basePos*4) + 1] = (uint8_t)(expF*255);
	outImage[(basePos*4) + 2] = (uint8_t)(expF*255);
	outImage[(basePos*4) + 3] = 255;
}

#define GLSL(version, shader)  "#version " #version "\n" #shader

static const GLchar* vertShaderSrc = GLSL(430 core,
	layout(location = 0) in vec3 posA;
	out smooth vec3 posV;
	out smooth vec2 texV;

	void main() {
		posV = posA;
		texV = (posA.xy+vec2(1,1))/2.0;
		texV = vec2(texV.x, 1-texV.y);
		gl_Position = vec4(posV,1);
	}
);

static const GLchar* fragShaderSrc = GLSL(430 core,
	uniform samplerBuffer image;
	uniform ivec2 imageSize;
	in smooth vec3 posV;
	in smooth vec2 texV;
	out vec4 col;

	void main() {
		//col = vec4(imageSize.xy/float(1024),0,1);
		//col = vec4(texV,0, 1);
		ivec2 texVP = ivec2(texV * imageSize);
		int p = (texVP.x + texVP.y * imageSize.x);
		//col = vec4(0, p/float(512*512), 0, 1);
		col = texelFetch(image,p);
	}
);

bool checkHW(char *name, const char *gpuType, int dev)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    strcpy(name, deviceProp.name);

    if (!strncasecmp(deviceProp.name, gpuType, strlen(gpuType)))
    {
        return true;
    }
    else
    {
        return false;
    }
}

int gpuGLDeviceInit()
{
	int deviceCount;
	cuCheckErr(cudaGetDeviceCount(&deviceCount));

	if (deviceCount == 0)
	{
		fprintf(stderr, "CUDA error: no devices supporting CUDA.\n");
		exit(EXIT_FAILURE);
	}

	int dev = 0;
	if (dev > deviceCount-1)
	{
		fprintf(stderr, "\n");
		fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n", deviceCount);
		fprintf(stderr, ">> gpuGLDeviceInit (-device=%d) is not a valid GPU device. <<\n", dev);
		fprintf(stderr, "\n");
		return -dev;
	}

	cudaDeviceProp deviceProp;
	cuCheckErr(cudaGetDeviceProperties(&deviceProp, dev));

	if (deviceProp.computeMode == cudaComputeModeProhibited)
	{
		fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
		return -1;
	}

	if (deviceProp.major < 1)
	{
		fprintf(stderr, "Error: device does not support CUDA.\n");
		exit(EXIT_FAILURE);
	}
	fprintf(stderr, "Using device %d: %s\n", dev, deviceProp.name);

	cuCheckErr(cudaGLSetGLDevice(dev));
	return dev;
}

bool renderImage();
int64_t doSome();
void glfwError(int error, const char* description);
void syncGL();
void freeStuff();
void destroyStuff();

GLFWwindow* window;
GLuint imageGLName, bufferTexture;
GLuint vertShader, fragShader, shaderProg;
GLuint sImageUnif, sImageSizeUnif;
GLuint quad_vertexbuffer;
cudaGraphicsResource* imageCUDAName;
uint64_t* exposures;
uint64_t* exposuresRAM;
GLubyte* dataBytes;

int64_t totalExps;
uint currPrimeInd;
uint primePos;

int dimBlock;
dim3 dimGrid;

uint64_t maxExp;

double timeNow = 0;
double targetFPS = 60;
double fps = 0;

int main(void) {

	//init OGL
	if(!glfwInit()) return 4;

	glfwSetErrorCallback(glfwError);

	window = glfwCreateWindow(imgWidth, imgHeight, "Cuddha", NULL, NULL);
	if(!window) {
		glfwTerminate();
		cerr << "couldn't open window" << endl;
		return -1;
	}
	cout << "Window: " << window << endl;
	glfwMakeContextCurrent(window);

	GLenum err = glewInit();
	if (GLEW_OK != err) {
	  /* Problem: glewInit failed, something is seriously wrong. */
	  cerr << "Error: " << err << " " << glewGetErrorString(err) << endl;
	  //return 5;
	}
	cout << "GLEW " << glewGetString(GLEW_VERSION) << endl;

	if (! glewIsSupported("GL_VERSION_4_0")) {
		cerr << "ERROR: Support for necessary OpenGL extensions missing." << endl;
		return false;
	}

	gpuGLDeviceInit();

	cout << "preparing GL context..." << endl;
	glGenBuffers(1, &imageGLName);
	cout << "glGenBuffers" <<endl;
	glCheck();
	glBindBuffer(GL_ARRAY_BUFFER, imageGLName);
	glBufferData(GL_ARRAY_BUFFER, imgWidth*imgHeight*sizeof(GLubyte)*4, NULL, GL_DYNAMIC_DRAW);
	glCheck();
	cout << "Buffer Texture..." << endl;
	glGenTextures(1, &bufferTexture);
	glBindTexture(GL_TEXTURE_BUFFER, bufferTexture);
	glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA8, imageGLName);
	glCheck();
	cout << "Shader..." << endl;
	shaderProg = glCreateProgram();
	vertShader = glCreateShader(GL_VERTEX_SHADER);
	fragShader = glCreateShader(GL_FRAGMENT_SHADER);
	glCheck();
	glShaderSource(vertShader, 1, &vertShaderSrc, 0);
	glShaderSource(fragShader, 1, &fragShaderSrc, 0);
	glCheck();
	glCompileShader(vertShader);
	glCompileShader(fragShader);
	cout << "check Compile..." << endl;
	GLint vertOK, fragOK;
	glGetShaderiv(vertShader, GL_COMPILE_STATUS, &vertOK);
	glGetShaderiv(fragShader, GL_COMPILE_STATUS, &fragOK);

	if(vertOK != GL_TRUE) {
		GLint infoLen;
		glGetShaderiv(vertShader, GL_INFO_LOG_LENGTH, &infoLen);
		GLchar* info = new GLchar[infoLen];
		glGetShaderInfoLog(vertShader, infoLen, &infoLen, info);
		cerr << "Vertex Shader Error: " << info << endl;
		delete[] info;
		return 3;
	}
	if(fragOK != GL_TRUE) {
		GLint infoLen;
		glGetShaderiv(fragShader, GL_INFO_LOG_LENGTH, &infoLen);
		GLchar* info = new GLchar[infoLen];
		glGetShaderInfoLog(fragShader, infoLen, &infoLen, info);
		cerr << "Fragment Shader Error: " << info << endl;
		delete[] info;
		return 3;
	}

	cout << "Link..." << endl;
	glAttachShader(shaderProg, vertShader);
	glAttachShader(shaderProg, fragShader);
	glCheck();
	glLinkProgram(shaderProg);
	GLint shaderOK;
	glGetProgramiv(shaderProg, GL_LINK_STATUS, &shaderOK);
	if(shaderOK != GL_TRUE) {
		GLint infoLen;
		glGetProgramiv(shaderProg, GL_INFO_LOG_LENGTH, &infoLen);
		GLchar* info = new GLchar[infoLen];
		glGetProgramInfoLog(shaderProg, infoLen, &infoLen, info);
		cerr << "Shader Link Error: " << info << endl;
		delete[] info;
		return 2;
	}
	glCheck();

	cout << "Uniforms..." << endl;
	sImageUnif = glGetUniformLocation(shaderProg, "image");
	sImageSizeUnif = glGetUniformLocation(shaderProg, "imageSize");
	glCheck();

	cout << "Quad Buffer..." << endl;
	static const GLfloat g_quad_vertex_buffer_data[] = {
	    -1.0f, -1.0f, 0.0f,
	    1.0f, -1.0f, 0.0f,
	    -1.0f,  1.0f, 0.0f,
	    -1.0f,  1.0f, 0.0f,
	    1.0f, -1.0f, 0.0f,
	    1.0f,  1.0f, 0.0f,
	};
	glGenBuffers(1, &quad_vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, quad_vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(g_quad_vertex_buffer_data), g_quad_vertex_buffer_data, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glCheck();

	cout << "preparing cuda interop..." << endl;
	cuCheckErr(cudaGraphicsGLRegisterBuffer(&imageCUDAName, imageGLName, cudaGraphicsRegisterFlagsWriteDiscard));
	glBindBuffer(GL_ARRAY_BUFFER, 0); glCheck();

	cout << "cudaMalloc "<< (sizeof(uint64_t) * imgSize / 1024) << "KB" << endl;
	cuCheckErr(cudaMalloc((void**) &exposures, sizeof(uint64_t) * imgSize));
	cout << "malloc" << endl;
	cudaMemset(exposures,0,imgSize*sizeof(uint64_t));
	exposuresRAM = new uint64_t[imgSize];
	memset(exposuresRAM, 0, imgSize*sizeof(uint64_t));
	dataBytes = new GLubyte[imgSize];

	cuCheckErr(cudaThreadSynchronize());cuCheckErr(cudaGetLastError());

	int n = imgWidth * imgHeight;
	dimBlock = 1024;
	dimGrid = dim3(n/dimBlock, n/dimBlock, 1);

	cuCheckErr(cudaThreadSynchronize());cuCheckErr(cudaGetLastError());

	syncGL();
	totalExps = 0;
	currPrimeInd = 0;
	primePos = 0;
	cout << "beginning..." << endl;
	for (int64_t numDone = 0; numDone != -1;) {
		numDone = doSome();
	}

	cuCheckErr(cudaDeviceReset());

	cout << "exiting..." << endl;
	return 0;
}

void bmp_quicksave(uint64_t maxExp, int primeInd) {
	for(int i = 0; i < imgSize; i++) {
		float ramp = exposuresRAM[i] / (maxExp / 2.5);
		if (ramp > 1)  {
			ramp = 1;
		}
		dataBytes[i] = (GLubyte) (ramp * 255);
	}

	char numstr[123];
	sprintf(numstr, "frame-p%d-max%d-exp%ld.bmp", primeInd, maxIterations, totalExps);
	drawbmp(numstr, dataBytes, imgWidth, imgHeight);
}

int64_t doSome() {
	int64_t workDone = 0;

	// Step 1: do some Broting

	// map primePos as primePosX&primePosY to a position in (x,y)
	// where x&y are smaller than prime_numbers[currPrimeInd]
	int prime, primePosX, primePosY;
	do {
		prime = prime_numbers[currPrimeInd];
		primePos++;
		if (primePos >= (uint)(prime*prime)) {
			currPrimeInd++;
			if (currPrimeInd >= prime_numbers.size()) {
				bmp_quicksave(maxExp, currPrimeInd);
				return -1;
			} else {
#ifdef BMP_QUICKSAVE
				bmp_quicksave(maxExp, currPrimeInd);
#endif
			}
			if (currPrimeInd == 5) {
				return -1;
			}
			primePos = 0;
			prime = prime_numbers[currPrimeInd];
		}
		primePosX = primePos % prime;
		primePosY = primePos / prime;
	} while (primePosX == 0 || primePosY == 0); // don't iterate the edges


	cout << "Broting prime: "<<prime << " pos " << primePos << " x " << primePosX << " y " << primePosY << endl;
	dim3 dimBlockBrot(XYRES, XYRES);
	dim3 dimGridBrot(XYRESMULT,XYRESMULT);
	cuBrot<double><<<dimGridBrot, dimBlockBrot>>>(exposures, maxIterations, prime, primePosX, primePosY, workDone+totalExps);
	cuCheckErr(cudaThreadSynchronize()); cuCheckErr(cudaGetLastError());


	// Step 2: get Min/Max values of broted stuff
	cout << "minmax..." << endl;
	cuCheckErr(cudaMemcpy(exposuresRAM, exposures, sizeof(uint64_t) * imgSize, cudaMemcpyDeviceToHost));
	maxExp = 0;
	uint64_t minExp = UINT64_MAX;
	for(int i = 0; i < imgSize; i++) {
		workDone += exposuresRAM[i];
		if(exposuresRAM[i] > maxExp) maxExp = exposuresRAM[i];
		if(exposuresRAM[i] < minExp) minExp = exposuresRAM[i];
	}
	uint64_t totalOld = totalExps;
	totalExps = workDone;
	workDone -= totalOld;
	if(UINT64_MAX == minExp) minExp = 0;
	cout << "exposures this run: " << workDone << " total: " << totalExps << endl;
	cout << "min=" << minExp << " max=" << maxExp << endl;

	// Step 3: convert to Image
	cout << "convert..." << endl;
	size_t dataLen;
	void* dataPtr;
	cuCheckErr(cudaGraphicsMapResources(1, &imageCUDAName, 0));
	cuCheckErr(cudaGraphicsResourceGetMappedPointer(&dataPtr, &dataLen, imageCUDAName));

	if(dataLen != (imgWidth*imgHeight*4*sizeof(GLubyte))) {
		cerr << "Error: data length expected " << (imgWidth*imgHeight*4*sizeof(GLubyte)) << ", got " << dataLen << endl;
		return -1;
	}
	//cout << "dimGrid: " << out(dimGrid) << " dimBlock: " << dimBlock << endl;
	cuConvertImage<<<dimGrid, dimBlock>>>((GLubyte*)dataPtr, (int)(dataLen/sizeof(GLubyte)), exposures, maxExp, minExp);

	cuCheckErr(cudaThreadSynchronize());	// Wait for the GPU launched work to complete
	cuCheckErr(cudaGetLastError());

	// unmap
	cudaGraphicsUnmapResources(1, &imageCUDAName, 0);

	// render
	if(!renderImage()) {
		cout << "window close requested..." << endl;
		//TODO save all state
		bmp_quicksave(maxExp, currPrimeInd);
		freeStuff();
		destroyStuff();
		return -1;
	}
	//sleep(100*1000);
	syncGL();
	return workDone;
}

bool renderImage() {

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(shaderProg);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_BUFFER, bufferTexture);
	glUniform2i(sImageSizeUnif, imgWidth, imgHeight);
	glUniform1i(sImageUnif, 0);

	glBindBuffer(GL_ARRAY_BUFFER, quad_vertexbuffer);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(
		0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
		3,                  // size
		GL_FLOAT,           // type
		GL_FALSE,           // normalized?
		0,                  // stride
		(void*)0            // array buffer offset
	);
	glDrawArrays(GL_TRIANGLES, 0, 6);

	glDisableVertexAttribArray(0);
	glBindTexture(GL_TEXTURE_BUFFER, 0);
	glCheck();

	glfwSwapBuffers(window);
	glfwPollEvents();
	return !glfwWindowShouldClose(window);
}

void syncGL() {
	if(timeNow == 0) {
		timeNow = glfwGetTime();
		return;
	}
	double newTime = glfwGetTime();
	double frameTime = newTime - timeNow;
	double targetFrameTime = 1/targetFPS;
	int wait = (int)((targetFrameTime-frameTime)*1000*1000);
	if(frameTime < targetFrameTime) {
		sleep(wait);
		newTime = glfwGetTime();
	}
	fps = 1/(newTime - timeNow);
	timeNow = newTime;
	cout << "frameTime: " << frameTime << " fps: " << fps << " wait: " << (targetFrameTime-frameTime) << "sec" << endl;
	if(fps < 5 && XYRESMULT >=2) {
		cout << "decreasing samples per run" << endl;
		XYRESMULT /= 2;
	}
}

void glfwError(int error, const char* description) {
    cerr << "GLFW Error " << error << description << endl;
}

void freeStuff() {
	cout << "freeing mems..." << endl;
	delete[] exposuresRAM;
	delete[] dataBytes;
	cudaFree(exposures);
	cudaGraphicsUnregisterResource(imageCUDAName);
	glDeleteBuffers(1, &imageGLName);
	glDeleteTextures(1, &bufferTexture);
	glDeleteProgram(shaderProg);
	glDeleteShader(fragShader);
	glDeleteShader(vertShader);
	glCheck();
}

void destroyStuff() {
	cout << "destroying windows..." << endl;
	glfwTerminate();
}

