#include "Common.h"
#include "Renderer.h"
#include "Blitter.h"
#include "Model.h"
#include "Camera.h"

#include <Probulator/Math.h>
#include <Probulator/Image.h>
#include <Probulator/DiscreteDistribution.h>
#include <Probulator/Experiments.h>

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/transform.hpp>
#include <GL/gl3w.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw_gl3.h>
#include <nfd.h>

#include <stdio.h>
#include <memory>

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
		addAllExperiments(m_experimentList);
		for (auto& e : m_experimentList)
		{
			m_availableExperimentNames.push_back(e->m_name);
		}

		memset(m_mouseButtonDown, 0, sizeof(m_mouseButtonDown));
		memset(m_mouseButtonDownPosition, 0, sizeof(m_mouseButtonDownPosition));
		memset(m_keyDown, 0, sizeof(m_keyDown));
		loadResources();

        m_sphereModel = std::unique_ptr<Model>(new Model(""));
        m_sphereModel->generateSphere();
        m_basisModel = std::unique_ptr<Model>(new Model("", "Data/Shaders/BasisVisualizer.vert"));
        m_basisModel->generateSphere();

		m_allExperimentNames = m_availableExperimentNames;

		m_camera.m_position = vec3(0.0f, 0.0f, 0.0f);
		m_camera.m_near = 0.01f;
		m_camera.m_far = 100.0f;
		m_camera.m_orbitRadius = 1.5f;

		m_cameraController.m_orbitRadius = m_camera.m_orbitRadius;

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

		if (ImGui::CollapsingHeader("Mode", nullptr, true, true))
		{
			ImGui::PushItemWidth(-1.0f);
			if (ImGui::Combo("", &m_currentExperiment, getComboItem, (void*)&m_allExperimentNames, (int)m_allExperimentNames.size()))
			{
				generateIrradianceImage();
			}
			ImGui::PopItemWidth();

			Experiment* experiment = m_experimentList[m_currentExperiment].get();
			std::vector<Experiment::Property> properties;
			experiment->getProperties(properties);

			for (Experiment::Property& p : properties)
			{
				switch (p.m_type)
				{
				case Experiment::PropertyType_Bool:
					ImGui::Checkbox(p.m_name, p.m_data.asBool);
					break;
				case Experiment::PropertyType_Float:
					ImGui::InputFloat(p.m_name, p.m_data.asFloat);
					break;
				case Experiment::PropertyType_Int:
					ImGui::InputInt(p.m_name, p.m_data.asInt);
					break;
				case Experiment::PropertyType_Vec2:
					ImGui::InputFloat2(p.m_name, glm::value_ptr(*p.m_data.asVec2));
					break;
				case Experiment::PropertyType_Vec3:
					ImGui::InputFloat3(p.m_name, glm::value_ptr(*p.m_data.asVec3));
					break;
				case Experiment::PropertyType_Vec4:
					ImGui::InputFloat4(p.m_name, glm::value_ptr(*p.m_data.asVec4));
					break;
				default:
					assert(false && "Unexpected property type");
				}
			}

			if (ImGui::Button("Execute"))
			{
				experiment->reset();
				generateIrradianceImage();
			}
		}

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

            const char* renderTypeStrs[] = { "Render Object", "Render Sphere", "Render Basis Visualizer" };
            int sz = int(sizeof(renderTypeStrs) / sizeof(const char*));
            ImGui::Combo("Render Type", &m_renderType, renderTypeStrs, sz);
		}

		if (ImGui::CollapsingHeader("Camera", nullptr, true, false))
		{
			for (int i = 0; i < CameraModeCount; ++i)
			{
				if (i != 0) ImGui::SameLine();
				ImGui::RadioButton(toString((CameraMode)i), reinterpret_cast<int*>(&m_cameraController.m_mode), i);
			}

			ImGui::SliderFloat("Exposure", &m_shaderUniforms.exposure, 0.0f, 100.0f, "%.3f", 4.0f);
			ImGui::SliderFloat("FOV", &m_camera.m_fov, 0.1f, pi);
			ImGui::SliderFloat("Near", &m_camera.m_near, 0.01f, 10.0f);
			ImGui::SliderFloat("Far", &m_camera.m_far, m_camera.m_near, 1000.0f);
			ImGui::DragFloat3("Position", glm::value_ptr(m_camera.m_position), 0.01f);
			ImGui::SliderFloat("Orbit radius", &m_cameraController.m_orbitRadius, 0.0f, 10.0f);
		}

        if (ImGui::CollapsingHeader("Basis Experiments", nullptr, true, true))
        {
            

            static int currItem = 0;
            if (m_availableExperimentNames.size())
            {
                ImGui::Combo("Add Experiment", &currItem, getComboItem, (void*)&m_availableExperimentNames, 
                                              (int)m_availableExperimentNames.size());
                ImGui::SameLine();
                if (ImGui::Button("+"))
                {
                    ExperimentResults* e = new ExperimentResults(m_availableExperimentNames[currItem], 
                                                                 getImTextureID(m_irradianceTexture), 
                                                                 getImTextureID(m_irradianceTexture),
                                                                 getImTextureID(m_irradianceTexture));
                    m_experimentResultsList.push_back(std::unique_ptr<ExperimentResults>(e));

                    m_availableExperimentNames.erase(m_availableExperimentNames.begin() + currItem);
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
                ImGui::Image(getImTextureID(m_radianceTexture), vec2(m_menuWidth, m_menuWidth / 2) / 4.0f);
                ImGui::NextColumn();
                ImGui::Image(getImTextureID(m_radianceTexture), vec2(m_menuWidth, m_menuWidth / 2) / 4.0f);
                ImGui::NextColumn();
                ImGui::Image(getImTextureID(m_radianceTexture), vec2(m_menuWidth, m_menuWidth / 2) / 4.0f);
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
                m_availableExperimentNames.push_back(deleteMe);
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

		cameraControllerInput.scrollDelta = m_mouseScrollDelta;

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
		m_mouseScrollDelta = vec2(0.0f);
	}

	void render()
	{
		// setup constants

		m_sceneViewport.x = m_windowSize.x - m_menuWidth;
		m_sceneViewport.y = m_windowSize.y;

		m_camera.m_aspect = (float)m_sceneViewport.x / (float)m_sceneViewport.y;
		m_smoothCamera.m_aspect = m_camera.m_aspect;

		m_shaderUniforms.viewMatrix = m_smoothCamera.getViewMatrix();
		m_shaderUniforms.projMatrix = m_smoothCamera.getProjectionMatrix();

		m_shaderUniforms.viewProjMatrix = m_shaderUniforms.projMatrix * m_shaderUniforms.viewMatrix;
		
		// draw scene

		glViewport(0, 0, m_sceneViewport.x, m_sceneViewport.y);

		const vec3 clearColor = vec3(0.5f);
		glClearColor(clearColor.r, clearColor.g, clearColor.b, 1.0f);
		glClearDepth(1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		m_blitter.drawLatLongEnvmap(*m_radianceTexture, m_shaderUniforms);

        if (m_renderType == eRenderObject)
		    m_model->draw(*m_irradianceTexture, m_shaderUniforms, m_worldMatrix);
        if (m_renderType == eRenderSphere)
            m_sphereModel->draw(*m_irradianceTexture, m_shaderUniforms, m_worldMatrix);
        if (m_renderType == eRenderBasisVisualizer)
            m_basisModel->draw(*m_irradianceTexture, m_shaderUniforms, m_worldMatrix);

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

	void generateIrradianceImage()
	{
		assert(m_experimentData);

		Experiment* experiment = m_experimentList[m_currentExperiment].get();

		experiment->runWithDepencencies(*m_experimentData);

		TextureFilter filter = makeTextureFilter(GL_REPEAT, GL_LINEAR);
		filter.wrapV = GL_CLAMP_TO_EDGE;

		m_irradianceTexture = createTextureFromImage(experiment->m_irradianceImage, filter);
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

		TextureFilter filter = makeTextureFilter(GL_REPEAT, GL_LINEAR);
		filter.wrapV = GL_CLAMP_TO_EDGE;

		m_radianceTexture = createTextureFromImage(m_radianceImage, filter);

		m_experimentData = std::unique_ptr<Experiment::SharedData>(new Experiment::SharedData(m_sampleCount, m_irradianceImageSize, m_radianceImage));
		resetAllExperiments(m_experimentList);

		generateIrradianceImage();
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

	void onScroll(float x, float y)
	{
		m_mouseScrollDelta += vec2(x, y);
	}

    enum  {
        eRenderObject,
        eRenderSphere,
        eRenderBasisVisualizer,
    };
    int m_renderType = eRenderObject;

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
    std::unique_ptr<Model> m_sphereModel;
	std::unique_ptr<Model> m_basisModel;
	CommonShaderUniforms m_shaderUniforms;

	Camera m_camera;
	Camera m_smoothCamera;
	CameraController m_cameraController;

	const u32 m_sampleCount = 20000;
	const ivec2 m_irradianceImageSize = ivec2(256, 128);
	std::unique_ptr<Experiment::SharedData> m_experimentData;
	std::vector<std::string> m_allExperimentNames;
    std::vector<std::string> m_availableExperimentNames;
    Probulator::ExperimentList m_experimentList;
    ExperimentResultsList m_experimentResultsList;
	int m_currentExperiment = 0;

	mat4 m_worldMatrix = mat4(1.0f);

	// If textures are used in ImGui, they must be kept alive until ImGui rendering is complete
	std::vector<TexturePtr> m_guiTextureReferences;

	enum { MouseButtonCount = 3 };
	bool m_mouseButtonDown[MouseButtonCount];
	vec2 m_mouseButtonDownPosition[MouseButtonCount];
	vec2 m_mousePosition = vec2(0.0f);
	vec2 m_oldMousePosition = vec2(0.0f);
	vec2 m_mouseScrollDelta = vec2(0.0f);
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
		ProbulatorGui* app = (ProbulatorGui*)glfwGetWindowUserPointer(window);
		app->onScroll((float)xoffset, (float)yoffset);
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
	printf("Probulator starting ...\n");

	glfwInit();

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	const ivec2 defaultWindowSize = ivec2(1280, 720);
	GLFWwindow * window = glfwCreateWindow(
		defaultWindowSize.x, defaultWindowSize.y,
		"Probulator", nullptr, nullptr);

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