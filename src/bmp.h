
#ifndef _BMP_H
#define _BMP_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <GL/gl.h>
#include <GL/glu.h>


void drawbmp (char * filename, GLubyte* data, int width, int height);

#ifdef __cplusplus
}
#endif

#endif
