#include "Common.h"
#include "Renderer.h"
#include "Blitter.h"
#include "Model.h"
#include "Camera.h"

#include <Probulator/Math.h>
#include <Probulator/Image.h>
#include <Probulator/DiscreteDistribution.h>
#include <Probulator/Experiments.h>

#include <glm/gtx/transform.hpp>
#include <GL/gl3w.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw_gl3.h>
#include <nfd.h>

#include <stdio.h>
#include <memory>

static Image imageRadianceToIrradiance(const Image& radianceImage, ivec2 outputSize = ivec2(256, 128), u32 sampleCount = 4000, bool scramblingEnabled = false)
{
	// TODO: share this code with CLI experiments framework

	Image downsampledRadianceImage;
	if (radianceImage.getSize() == outputSize)
	{
		downsampledRadianceImage = radianceImage;
	}
	else
	{
		downsampledRadianceImage = imageResize(radianceImage, outputSize);
	}

	const ivec2 outputSizeMinusOne = outputSize - 1;
	const vec2 outputSizeMinusOneRcp = vec2(1.0) / vec2(outputSizeMinusOne);

	std::vector<float> texelWeights;
	std::vector<float> texelAreas;
	float weightSum = 0.0;
	
	ImageBase<vec3> directionImage(outputSize);
	for (int y = 0; y < outputSize.y; ++y)
	{
		for (int x = 0; x < outputSize.x; ++x)
		{
			ivec2 pixelPos(x, y);
			float area = latLongTexelArea(pixelPos, outputSize);

			const vec3 radiance = (vec3)downsampledRadianceImage.at(pixelPos);
			float intensity = dot(vec3(1.0f / 3.0f), radiance);
			float weight = intensity * area;

			weightSum += weight;
			texelWeights.push_back(weight);
			texelAreas.push_back(area);

			vec2 uv = (vec2(pixelPos) + vec2(0.5f)) * outputSizeMinusOneRcp;
			directionImage.at(pixelPos) = latLongTexcoordToCartesian(uv);
		}
	}

	Image irradianceImage = Image(outputSize);

	DiscreteDistribution<float> discreteDistribution(texelWeights.data(), texelWeights.size(), weightSum);

	irradianceImage.parallelForPixels2D([&](vec4& pixel, ivec2 pixelPos)
	{
		u32 pixelIndex = pixelPos.x + pixelPos.y * outputSize.x;
		u32 seed = scramblingEnabled ? pixelIndex : 0;
		std::mt19937 rng(seed);

		vec2 uv = (vec2(pixelPos) + vec2(0.5f)) * outputSizeMinusOneRcp;
		vec3 normal = latLongTexcoordToCartesian(uv);
		vec3 accum = vec3(0.0f);

		for (u32 sampleIt = 0; sampleIt < sampleCount; ++sampleIt)
		{
			u32 sampleIndex = (u32)discreteDistribution(rng);
			float sampleProbability = texelWeights[sampleIndex] / weightSum;
			vec3 sampleDirection = directionImage.at(sampleIndex);
			float cosTerm = dotMax0(normal, sampleDirection);
			float sampleArea = (float)texelAreas[sampleIndex];
			vec3 sampleRadiance = (vec3)downsampledRadianceImage.at(sampleIndex) * sampleArea;
			accum += sampleRadiance * cosTerm / sampleProbability;
		}

		accum /= sampleCount * pi;

		pixel = vec4(accum, 1.0f);
	});

	return irradianceImage;
}

struct ExperimentResults
{
    ExperimentResults(const std::string& l = "", ImTextureID r = 0, ImTextureID ir = 0, ImTextureID s = 0)
        : m_label(l), m_radianceImage(r), m_irradianceImage(ir), m_sampeImage(s), m_shouldRender(false)
        { }

    std::string m_label;
    ImTextureID m_radianceImage;
    ImTextureID m_irradianceImage;
    ImTextureID m_sampeImage;
    bool m_shouldRender;
};
typedef std::vector<std::unique_ptr<ExperimentResults>> ExperimentResultsList;

class ProbulatorGui
{
public:

	ProbulatorGui()
	{
		memset(m_mouseButtonDown, 0, sizeof(m_mouseButtonDown));
		memset(m_mouseButtonDownPosition, 0, sizeof(m_mouseButtonDownPosition));
		memset(m_keyDown, 0, sizeof(m_keyDown));
		loadResources();

        addAllExperiments(m_experimentList);
        for (auto& e : m_experimentList)
            m_experimentNames.push_back(e->m_name);

		m_camera.m_position = vec3(0.0f, 0.0f, 1.5f);
		m_camera.m_near = 0.01f;
		m_camera.m_far = 100.0f;

		m_smoothCamera = m_camera;
	}

	~ProbulatorGui()
	{

	}

    static bool getComboItem(void* data, int idx, const char** outText)
    {
        *outText = (*((std::vector<std::string>*)data))[idx].c_str();
        return true;
    }

	ImTextureID getImTextureID(TexturePtr texture)
	{
		m_guiTextureReferences.push_back(m_radianceTexture);
		return reinterpret_cast<ImTextureID>((u64)texture->m_native);
	}

	void updateImGui()
	{
		m_guiTextureReferences.clear();

		const vec4 highlightColor = vec4(0.8f, 1.0f, 0.9f, 1.0f);

		ImGui::SetNextWindowSize(vec2(m_menuWidth, m_windowSize.y));
		ImGui::SetNextWindowPos(vec2(m_windowSize.x - m_menuWidth, 0.0f));

		ImGui::Begin("Menu", nullptr, 
			ImGuiWindowFlags_NoResize | 
			ImGuiWindowFlags_NoMove | 
			ImGuiWindowFlags_NoCollapse |
			ImGuiWindowFlags_NoTitleBar);

		if (ImGui::CollapsingHeader("Environment", nullptr, true, false))
		{
			ImGui::Text("File:");
			ImGui::SameLine();
			ImGui::TextColored(highlightColor, "%s", m_envmapFilename.c_str());

			if (ImGui::Button("Load HDR"))
			{
				nfdchar_t* filename = nullptr;
				if (NFD_OpenDialog("hdr", nullptr, &filename) == NFD_OKAY)
				{
					loadEnvmap(filename);
				}
				free(filename);
			}

			ImGui::Text("Radiance:");
			ImGui::Image(getImTextureID(m_radianceTexture), vec2(m_menuWidth, m_menuWidth / 2));

			ImGui::Text("Irradiance:");
			ImGui::Image(getImTextureID(m_irradianceTexture), vec2(m_menuWidth, m_menuWidth / 2));
		}

		if (ImGui::CollapsingHeader("Object", nullptr, true, false))
		{
			ImGui::Text("File:");
			ImGui::SameLine();
			ImGui::TextColored(highlightColor, "%s", m_objectFilename.c_str());

			if (ImGui::Button("Load OBJ"))
			{
				nfdchar_t* filename = nullptr;
				if (NFD_OpenDialog("obj", nullptr, &filename) == NFD_OKAY)
				{
					loadModel(filename);
				}
				free(filename);
			}
		}

		if (ImGui::CollapsingHeader("Camera", nullptr, true, false))
		{
			const char* cameraModeNames[CameraModeCount];
			for (u32 i = 0; i < CameraModeCount; ++i)
			{
				cameraModeNames[i] = toString((CameraMode)i);
			}
			ImGui::Combo("Mode", reinterpret_cast<int*>(&m_cameraController.m_mode), cameraModeNames, CameraModeCount);
			ImGui::SliderFloat("FOV", &m_camera.m_fov, 0.1f, pi);
			ImGui::SliderFloat("Near", &m_camera.m_near, 0.01f, 10.0f);
			ImGui::SliderFloat("Far", &m_camera.m_far, m_camera.m_near, 1000.0f);
			ImGui::DragFloat3("Position", &m_camera.m_position.x, 0.01f);
			if (m_cameraController.m_mode == CameraMode_Orbit)
			{
				ImGui::DragFloat3("Orbit center", &m_cameraController.m_orbitCenter.x, 0.01f);
			}
		}

        if (ImGui::CollapsingHeader("Basis Experiments"), nullptr, true, true)
        {
            

            static int currItem = 0;
            if (m_experimentNames.size())
            {
                ImGui::Combo("Add Experiment", &currItem, getComboItem, (void*)&m_experimentNames, 
                                              (int)m_experimentNames.size());
                ImGui::SameLine();
                if (ImGui::Button("+"))
                {
                    ExperimentResults* e = new ExperimentResults(m_experimentNames[currItem], 
                                                                 (ImTextureID)m_irradianceTexture->m_native, 
                                                                 (ImTextureID)m_irradianceTexture->m_native,
                                                                 (ImTextureID)m_irradianceTexture->m_native);
                    m_experimentResultsList.push_back(std::unique_ptr<ExperimentResults>(e));

                    m_experimentNames.erase(m_experimentNames.begin() + currItem);
                    currItem = 0;
                }
            }

            // render each experiment

            ImGui::Separator();
            ImGui::Columns(4, "Experiment Results", true);
            ImGui::Text("Mode");
            ImGui::NextColumn();
            ImGui::Text("Radiance");
            ImGui::NextColumn();
            ImGui::Text("Irradiance");
            ImGui::NextColumn();
            ImGui::Text("Irradiance Error");
            ImGui::Text("(sMAPE)");
            ImGui::Separator();
            ImGui::NextColumn();

            std::string deleteMe;
            int guiIdx = 0;
            for (const auto& e : m_experimentResultsList)
            {
                ImGui::BeginGroup();
                {
                    std::string label = e->m_label;
                    std::string::size_type pos = label.find_first_of('[');
                    if (pos != std::string::npos)
                    {
                        std::string mainName = std::string(&label[0], &label[pos]);
                        std::string subName  = std::string(&label[pos], &label[label.size()]);
                        ImGui::Text(mainName.c_str());
                        ImGui::Text(subName.c_str());
                    }
                    else
                        ImGui::Text(label.c_str());
                }
                ImGui::Spacing();
                if (ImGui::Button((std::string("Delete##") + std::string(1, char(guiIdx++))).c_str()))
                    deleteMe = e->m_label.c_str();
                if (ImGui::Checkbox(std::string("Render##" + std::string(1, char(guiIdx++))).c_str(), &e->m_shouldRender))
                {
                    if (e->m_shouldRender)
                    {   
                        // turn off all other experiments
                        for (const auto& o : m_experimentResultsList)
                        {
                            if (o != e)
                                o->m_shouldRender = false;
                        }
                    }
                }
                ImGui::EndGroup();
                ImGui::NextColumn();
                ImGui::Image((ImTextureID)m_radianceTexture->m_native, vec2(m_menuWidth, m_menuWidth / 2) / 4.0f);
                ImGui::NextColumn();
                ImGui::Image((ImTextureID)m_radianceTexture->m_native, vec2(m_menuWidth, m_menuWidth / 2) / 4.0f);
                ImGui::NextColumn();
                ImGui::Image((ImTextureID)m_radianceTexture->m_native, vec2(m_menuWidth, m_menuWidth / 2) / 4.0f);
                ImGui::Separator();
                ImGui::NextColumn();
            }

            if (deleteMe != "")
            {
                // iterate through experiment results and remove
                auto i = std::begin(m_experimentResultsList);
                while (i != std::end(m_experimentResultsList))
                {
                    if ((*i)->m_label == deleteMe)
                        i = m_experimentResultsList.erase(i);
                    else
                        ++i;
                }

                // add basis back to drop down list
                m_experimentNames.push_back(deleteMe);
            }

            ImGui::Columns(1);
        }

		ImGui::End();
	}

	void updateObject()
	{
		if (m_mouseButtonDown[1])
		{
			const float rotateSpeed = 0.005f;

			vec2 mouseDelta = m_mousePosition - m_oldMousePosition;

			mat4 rotUp = glm::rotate(mouseDelta.x * rotateSpeed, m_camera.m_orientation[1]);
			mat4 rotRight = glm::rotate(mouseDelta.y * rotateSpeed, m_camera.m_orientation[0]);

			m_worldMatrix = rotUp * rotRight * m_worldMatrix;
		}
	}

	void updateCamera()
	{
		CameraController::InputState cameraControllerInput;

		if (m_mouseButtonDown[0])
		{
			vec2 mouseDelta = m_mousePosition - m_oldMousePosition;
			cameraControllerInput.rotateAroundUp = mouseDelta.x;
			cameraControllerInput.rotateAroundRight = mouseDelta.y;
		}

		if (m_keyDown[GLFW_KEY_LEFT_CONTROL])
		{
			cameraControllerInput.moveSpeedMultiplier *= 0.1f;
		}
		if (m_keyDown[GLFW_KEY_LEFT_SHIFT])
		{
			cameraControllerInput.moveSpeedMultiplier *= 10.0f;
		}
		if (m_keyDown[GLFW_KEY_W])
		{
			cameraControllerInput.moveForward += 1.0f;
		}
		if (m_keyDown[GLFW_KEY_S])
		{
			cameraControllerInput.moveForward -= 1.0f;
		}
		if (m_keyDown[GLFW_KEY_A])
		{
			cameraControllerInput.moveRight -= 1.0f;
		}
		if (m_keyDown[GLFW_KEY_D])
		{
			cameraControllerInput.moveRight += 1.0f;
		}
		if (m_keyDown[GLFW_KEY_E])
		{
			cameraControllerInput.moveUp += 1.0f;
		}
		if (m_keyDown[GLFW_KEY_Q])
		{
			cameraControllerInput.moveUp -= 1.0f;
		}

		cameraControllerInput.rotateSpeedMultiplier = min(1.0f, m_camera.m_fov);

		m_cameraController.update(cameraControllerInput, m_camera);

		const float positionAlpha = 0.1f;
		const float orientationAlpha = 0.25f;
		m_smoothCamera = m_cameraController.interpolate(
			m_smoothCamera, m_camera, 
			positionAlpha, orientationAlpha);
	}

	void update()
	{
		updateObject();
		updateCamera();
		updateImGui();

		m_oldMousePosition = m_mousePosition;
	}

	void render()
	{
		// setup constants

		m_sceneViewport.x = m_windowSize.x - m_menuWidth;
		m_sceneViewport.y = m_windowSize.y;

		m_camera.m_aspect = (float)m_sceneViewport.x / (float)m_sceneViewport.y;
		m_smoothCamera.m_aspect = m_camera.m_aspect;

		m_viewMatrix = m_smoothCamera.getViewMatrix();
		m_projectionMatrix = m_smoothCamera.getProjectionMatrix();

		m_viewPojectionMatrix = m_projectionMatrix * m_viewMatrix;
		
		// draw scene

		glViewport(0, 0, m_sceneViewport.x, m_sceneViewport.y);

		const vec3 clearColor = vec3(0.5f);
		glClearColor(clearColor.r, clearColor.g, clearColor.b, 1.0f);
		glClearDepth(1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		m_blitter.drawLatLongEnvmap(*m_radianceTexture, m_viewMatrix, m_projectionMatrix);

		m_model->draw(*m_irradianceTexture, m_worldMatrix, m_viewPojectionMatrix);

		// draw UI on top

		ImGui::Render();
	}

	void setWindowSize(ivec2 size)
	{
		m_windowSize = size;
	}

	void loadResources()
	{
		loadEnvmap(m_envmapFilename.c_str());
		loadModel(m_objectFilename.c_str());
	}

	void loadModel(const char* filename)
	{
		printf("Loading model '%s'\n", filename);

		m_objectFilename = filename;
		m_model = std::unique_ptr<Model>(new Model(m_objectFilename.c_str()));

		vec3 dimensions = m_model->m_dimensions;
		float largestSide = max(max(dimensions.x, dimensions.y), dimensions.z);

		m_worldMatrix = mat4(1.0f / largestSide);
		m_worldMatrix[3] = vec4(-m_model->m_center / largestSide, 1.0f);
	}

	void loadEnvmap(const char* filename)
	{
		printf("Loading envmap '%s'\n", filename);

		m_envmapFilename = filename;

		bool imageLoaded = m_radianceImage.readHdr(m_envmapFilename.c_str());
		if (!imageLoaded)
		{
			m_radianceImage = Image(4, 4);
			m_radianceImage.fill(vec4(0.0f, 0.0f, 0.0f, 1.0f));
		}

		Image irradianceImage = imageRadianceToIrradiance(m_radianceImage);

		TextureFilter filter = makeTextureFilter(GL_REPEAT, GL_LINEAR);
		filter.wrapV = GL_CLAMP_TO_EDGE;

		m_radianceTexture = createTextureFromImage(m_radianceImage, filter);
		m_irradianceTexture = createTextureFromImage(irradianceImage, filter);
	}

	void onMouseButton(int button, int action, int mods)
	{
		if (button >= MouseButtonCount)
			return;

		m_mouseButtonDown[button] = action == GLFW_PRESS;
		if (m_mouseButtonDown[button])
		{
			m_mouseButtonDownPosition[button] = m_mousePosition;
		}
	}

	void onMouseMove(const vec2& mousePosition)
	{
		m_mousePosition = mousePosition;
	}

	void onKey(int key, int scancode, int action, int mods)
	{
		if (key < 0 || key > GLFW_KEY_LAST)
			return;

		m_keyDown[key] = action != GLFW_RELEASE;
	}

	std::string m_objectFilename = "Data/Models/bunny.obj";
	std::string m_envmapFilename = "Data/Probes/wells.hdr";
	ivec2 m_windowSize = ivec2(1280, 720);
	ivec2 m_sceneViewport = ivec2(1, 1);
	int m_menuWidth = 660;
	Image m_radianceImage;
	TexturePtr m_radianceTexture;
	TexturePtr m_irradianceTexture;
	Blitter m_blitter;
	std::unique_ptr<Model> m_model;

	Camera m_camera;
	Camera m_smoothCamera;
	CameraController m_cameraController;

    std::vector<std::string> m_experimentNames;
    Probulator::ExperimentList m_experimentList;
    ExperimentResultsList m_experimentResultsList;

	mat4 m_worldMatrix = mat4(1.0f);
	mat4 m_viewMatrix = mat4(1.0f);
	mat4 m_projectionMatrix = mat4(1.0f);
	mat4 m_viewPojectionMatrix = mat4(1.0f);

	// If textures are used in ImGui, they must be kept alive until ImGui rendering is complete
	std::vector<TexturePtr> m_guiTextureReferences;

	enum { MouseButtonCount = 3 };
	bool m_mouseButtonDown[MouseButtonCount];
	vec2 m_mouseButtonDownPosition[MouseButtonCount];
	vec2 m_mousePosition = vec2(0.0f);
	vec2 m_oldMousePosition = vec2(0.0f);
	bool m_keyDown[GLFW_KEY_LAST+1];
};

static void cbWindowSize(GLFWwindow* window, int w, int h)
{
	ProbulatorGui* app = (ProbulatorGui*)glfwGetWindowUserPointer(window);
	app->setWindowSize(ivec2(w, h));
}

static void cbMouseButton(GLFWwindow* window, int button, int action, int mods)
{
	ProbulatorGui* app = (ProbulatorGui*)glfwGetWindowUserPointer(window);
	ImGui_ImplGlfwGL3_MouseButtonCallback(window, button, action, mods);
	if (!ImGui::GetIO().WantCaptureMouse)
	{
		app->onMouseButton(button, action, mods);
	}
}

static void cbScroll(GLFWwindow* window, double xoffset, double yoffset)
{
	ImGui_ImplGlfwGL3_ScrollCallback(window, xoffset, yoffset);

	if (!ImGui::GetIO().WantCaptureMouse)
	{
		// ProbulatorGui* app = (ProbulatorGui*)glfwGetWindowUserPointer(window);
		// app->onScroll();
	}
}

static void cbCursorPos(GLFWwindow* window, double x, double y)
{
	ProbulatorGui* app = (ProbulatorGui*)glfwGetWindowUserPointer(window);
	if (!ImGui::GetIO().WantCaptureMouse)
	{
		app->onMouseMove(vec2(x, y));
	}
}

static void cbKey(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	ProbulatorGui* app = (ProbulatorGui*)glfwGetWindowUserPointer(window);
	ImGui_ImplGlfwGL3_KeyCallback(window, key, scancode, action, mods);

	if (!ImGui::GetIO().WantCaptureKeyboard)
	{
		app->onKey(key, scancode, action, mods);
	}	
}

static void cbChar(GLFWwindow* window, unsigned int c)
{
	ImGui_ImplGlfwGL3_CharCallback(window, c);

	if (!ImGui::GetIO().WantCaptureKeyboard)
	{
		// ProbulatorGui* app = (ProbulatorGui*)glfwGetWindowUserPointer(window);
		// app->onChar();
	}
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
		"Probulator GUI", nullptr, nullptr);

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

	ImGui_ImplGlfwGL3_Init(window, true);
	ImGui::GetStyle().WindowRounding = 0.0f;

	ProbulatorGui* app = new ProbulatorGui();
	app->setWindowSize(defaultWindowSize);

	glfwSetWindowUserPointer(window, app);
	glfwSetWindowSizeCallback(window, cbWindowSize);

	glfwSetMouseButtonCallback(window, cbMouseButton);
	glfwSetScrollCallback(window, cbScroll);
	glfwSetKeyCallback(window, cbKey);
	glfwSetCharCallback(window, cbChar);
	glfwSetCursorPosCallback(window, cbCursorPos);

	glfwSwapInterval(1); // vsync ON

	do
	{
		ImGui_ImplGlfwGL3_NewFrame();

		app->update();
		app->render();

		glfwSwapBuffers(window);
		glfwPollEvents();
	} while(!glfwWindowShouldClose(window));

	delete app;

	ImGui_ImplGlfwGL3_Shutdown();

	glfwTerminate();

	return 0;
}