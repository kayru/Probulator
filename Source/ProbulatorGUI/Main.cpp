#include <Probulator/Common.h>
#include <Probulator/Math.h>

#include <GL/gl3w.h>
#include <GLFW/glfw3.h>

#include <stdio.h>
#include <memory>

using namespace Probulator;

class ProbulatorGui
{
public:
	ProbulatorGui()
	{

	}

	~ProbulatorGui()
	{

	}

	void render()
	{
		glViewport(0, 0, m_windowSize.x, m_windowSize.y);

		const vec3 clearColor = vec3(0.1f, 0.2f, 0.3f);
		glClearColor(clearColor.r, clearColor.g, clearColor.b, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
	}

	void setWindowSize(ivec2 size)
	{
		m_windowSize = size;
	}

	ivec2 m_windowSize = ivec2(1280, 720);
};

static void cbWindowSize(GLFWwindow* window, int w, int h)
{
	ProbulatorGui* app = (ProbulatorGui*)glfwGetWindowUserPointer(window);
	app->setWindowSize(ivec2(w, h));
}

static void cbRefresh(GLFWwindow * window)
{
	ProbulatorGui* app = (ProbulatorGui*)glfwGetWindowUserPointer(window);
	app->render();
	glfwSwapBuffers(window);
}

int main(int argc, char** argv)
{
	printf("Probulator GUI starting ...\n");

	glfwInit();

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	const ivec2 defaultWindowSize = ivec2(1280, 720);
	GLFWwindow * window = glfwCreateWindow(
		defaultWindowSize.x, defaultWindowSize.y,
		"ProbulatorGUI", nullptr, nullptr);

	glfwMakeContextCurrent(window);

	if (gl3wInit())
	{
		printf("ERROR: failed to initialize OpenGL\n");
		return 1;
	}

	if (!gl3wIsSupported(3, 2))
	{
		printf("ERROR: OpenGL 3.2 is not supported\n");
		return 1;
	}

	ProbulatorGui* app = new ProbulatorGui();
	app->setWindowSize(defaultWindowSize);

	glfwSetWindowUserPointer(window, app);
	glfwSetWindowSizeCallback(window, cbWindowSize);
	glfwSetWindowRefreshCallback(window, cbRefresh);

	glfwSwapInterval(1); // vsync ON

	do
	{
		app->render();
		glfwSwapBuffers(window);
		glfwPollEvents();
	} while(!glfwWindowShouldClose(window));

	delete app;

	glfwTerminate();
	
	return 0;
}
