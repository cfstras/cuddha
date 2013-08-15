#include <iostream>
#include <sstream>
#include <boost/format.hpp>
#include <cstring>
#include <cstdio>

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

using namespace std;
using namespace boost;

static const int imgWidth = 512;
static const int imgHeight = 512;
static const int imgSize = imgWidth * imgHeight;

#define out(dim) "[x="<<dim.x<<",y="<<dim.y<<",z="<<dim.z<<"]"

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define cuCheckErr(value) {													\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		cerr << "Error " <<_m_cudaStat << ": "								\
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

__device__ float function(/*args*/) {
	return 0;
}

__global__ void cuBrot(GLubyte* data, int len) {
	long int idx = threadIdx.x + blockDim.x * blockIdx.x;

	long int x = idx % imgWidth;
	long int y = idx / imgWidth;

	//printf("%4ld x=%4ld y=%4ld\n",idx, x, y);

	int basePos = (x+y*imgWidth)*4;
	if(basePos+4 > len) {
		printf("x=%d, y=%d Error: usedlen %4d > len %4d\n", x,y, basePos, len);
		return;
	}/* else {
		printf("x=%d, y=%d usedlen %4d, len %4d\n", x,y, usedLen, len);
	}*/

	data[(basePos) + 0] = (uint8_t)((float)x / imgWidth * 255);
	data[(basePos) + 1] = (uint8_t)((float)y / imgHeight * 255);
	data[(basePos) + 2] = 0;
	data[(basePos) + 3] = 255;
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
int doSome();
void glfwError(int error, const char* description);
void sync();
void freeStuff();
void destroyStuff();

GLFWwindow* window;
GLuint imageGLName, bufferTexture;
GLuint vertShader, fragShader, shaderProg;
GLuint sImageUnif, sImageSizeUnif;
GLuint quad_vertexbuffer;
cudaGraphicsResource* imageCUDAName;

double timeNow = 0;
double targetFPS = 0.5;
double fps = 0;

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
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

	/*cuCheckErr(cudaMalloc((void**) &hostindata, sizeof(int) * WORK_SIZE));
	cuCheckErr(
			cudaMemcpy(hostindata, idata, sizeof(int) * WORK_SIZE, cudaMemcpyHostToDevice));*/
	cout << "beginning..." << endl;
	for (int numDone = 0; numDone != -1;) {
		numDone = doSome();
		//break;
	}

	cuCheckErr(cudaDeviceReset());

	cout << "exiting..." << endl;
	return 0;
}

int doSome() {
	//TODO file:///usr/local/cuda-5.5/doc/html/cuda-c-programming-guide/index.html#opengl-interoperability

	size_t dataLen;
	void* dataPtr;
	cuCheckErr(cudaGraphicsMapResources(1, &imageCUDAName, 0));
	cuCheckErr(cudaGraphicsResourceGetMappedPointer(&dataPtr, &dataLen, imageCUDAName));

	if(dataLen != (imgWidth*imgHeight*4*sizeof(GLubyte))) {
		cerr << "Error: data length expected " << (imgWidth*imgHeight*1*sizeof(GLubyte)) << ", got " << dataLen << endl;
		return -1;
	}

	int workDone = 0;

	// do CUDA work

	int n = imgWidth * imgHeight;
	int dimBlock(1024);
	dim3 dimGrid(n/dimBlock);
	cout << "dimGrid: " << out(dimGrid) << " dimBlock: " << dimBlock << endl;
	cuBrot<<<dimGrid, dimBlock>>>((GLubyte*)dataPtr, (int)(dataLen/sizeof(GLubyte)));
	workDone++;

	cuCheckErr(cudaThreadSynchronize());	// Wait for the GPU launched work to complete
	cuCheckErr(cudaGetLastError());

	// unmap
	cudaGraphicsUnmapResources(1, &imageCUDAName, 0);

	// render
	if(!renderImage()) {
		cout << "window close requested..." << endl;
		//TODO save?
		freeStuff();
		destroyStuff();
		return -1;
	}
	sync();
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

void sync() {
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
}

void glfwError(int error, const char* description) {
    cerr << "GLFW Error " << error << description << endl;
}

void freeStuff() {
	cout << "freeing mems..." << endl;
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

