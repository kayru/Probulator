#include "Common.h"
#include "Renderer.h"
#include "Blitter.h"
#include "Model.h"

#include <Probulator/Math.h>
#include <Probulator/Image.h>
#include <Probulator/DiscreteDistribution.h>

#include <glm/gtc/matrix_transform.hpp>
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

class ProbulatorGui
{
public:
	ProbulatorGui()
	{
		loadResources();
	}

	~ProbulatorGui()
	{

	}

    static bool getComboItem(void* data, int idx, const char** outText)
    {
        *outText = (*((std::vector<std::string>*)data))[idx].c_str();
        return true;
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
			ImGui::TextColored(highlightColor, m_envmapFilename.c_str());

			if (ImGui::Button("Load HDR"))
			{
				nfdchar_t* filename = nullptr;
				if (NFD_OpenDialog("hdr", nullptr, &filename) == NFD_OKAY)
				{
					loadEnvmap(filename);
				}
				free(filename);
			}

			m_guiTextureReferences.push_back(m_radianceTexture);
			ImGui::Text("Radiance:");
			ImGui::Image((ImTextureID)m_radianceTexture->m_native, vec2(m_menuWidth, m_menuWidth / 2));

			m_guiTextureReferences.push_back(m_irradianceTexture);
			ImGui::Text("Irradiance:");
			ImGui::Image((ImTextureID)m_irradianceTexture->m_native, vec2(m_menuWidth, m_menuWidth / 2));
		}

		if (ImGui::CollapsingHeader("Object", nullptr, true, false))
		{
			ImGui::Text("File:");
			ImGui::SameLine();
			ImGui::TextColored(highlightColor, m_objectFilename.c_str());

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
			ImGui::SliderFloat("FOV", &m_cameraFov, 0.1f, pi);
			ImGui::DragFloat3("Target", &m_cameraTarget.x, 0.01f);
			ImGui::DragFloat3("Position", &m_cameraPosition.x, 0.01f);
		}

        if (ImGui::CollapsingHeader("Basis Experiments"), nullptr, true, true)
        {
            static std::vector<std::string> kBasisTypes;
            static bool first = true;
            struct ExperimentResults 
            {
                ExperimentResults(const std::string& l = "", ImTextureID r = 0, ImTextureID ir = 0, ImTextureID s = 0)
                          : label(l), radianceImage(r), irradianceImage(ir), sampeImage(s), shouldRender(false)
                { }
                std::string label;
                ImTextureID radianceImage;
                ImTextureID irradianceImage;
                ImTextureID sampeImage;
                bool shouldRender;
            };

            typedef std::vector<std::unique_ptr<ExperimentResults>> ExperimentResultsList;
            static ExperimentResultsList experimentList;

            if (first)
            {
                kBasisTypes.push_back("basis 1");
                kBasisTypes.push_back("basis 2");
                kBasisTypes.push_back("basis 3");
                first = false;
            }

            static int currItem = 0;
            if (kBasisTypes.size())
            {
                ImGui::Combo("Add Experiment", &currItem, getComboItem, (void*)&kBasisTypes, (int)kBasisTypes.size());
                ImGui::SameLine();
                if (ImGui::Button("+"))
                {
                    experimentList.push_back(std::unique_ptr<ExperimentResults>(new ExperimentResults(kBasisTypes[currItem],
                                                            (ImTextureID)m_irradianceTexture->m_native, 
                                                            (ImTextureID)m_irradianceTexture->m_native,
                                                            (ImTextureID)m_irradianceTexture->m_native)));

                    kBasisTypes.erase(kBasisTypes.begin() + currItem);
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
            for (const auto& e : experimentList)
            {
                ImGui::BeginGroup();
                ImGui::Text(e->label.c_str());
                ImGui::Spacing();
                if (ImGui::Button("Delete"))
                    deleteMe = e->label.c_str();;
                if (ImGui::Checkbox("Render", &e->shouldRender))
                {
                    if (e->shouldRender)
                    {   
                        // turn off all other experiments
                        for (const auto& o : experimentList)
                        {
                            if (o != e)
                                o->shouldRender = false;
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
                auto i = std::begin(experimentList);
                while (i != std::end(experimentList))
                {
                    if ((*i)->label == deleteMe)
                        i = experimentList.erase(i);
                    else
                        ++i;
                }

                // add basis back to drop down list
                kBasisTypes.push_back(deleteMe);
            }

            ImGui::Columns(1);
        }

		ImGui::End();
	}

	void render()
	{
		// setup constants

		m_sceneViewport.x = m_windowSize.x - m_menuWidth;
		m_sceneViewport.y = m_windowSize.y;

		//m_worldMatrix = glm::rotate(m_worldMatrix, 0.01f, vec3(0.0f, 1.0f, 0.0f));
		m_viewMatrix = glm::lookAt(m_cameraPosition, m_cameraTarget, vec3(0.0f, 1.0f, 0.0f));

		const float viewportAspect = (float)m_sceneViewport.x / (float)m_sceneViewport.y;
		m_projectionMatrix = glm::perspective(m_cameraFov, viewportAspect, m_cameraNear, m_cameraFar);
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
		m_worldMatrix[3].w = 1.0;
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

	std::string m_objectFilename = "Data/Models/bunny.obj";
	std::string m_envmapFilename = "Data/Probes/wells.hdr";
	ivec2 m_windowSize = ivec2(1280, 720);
	ivec2 m_sceneViewport = ivec2(1, 1);
	int m_menuWidth = 560; // 1280-720
	Image m_radianceImage;
	TexturePtr m_radianceTexture;
	TexturePtr m_irradianceTexture;
	Blitter m_blitter;
	std::unique_ptr<Model> m_model;

	vec3 m_cameraTarget = vec3(0.0f, 0.5f, 0.0f);
	vec3 m_cameraPosition = vec3(0.0f, 0.5f, 2.0f);
	float m_cameraFov = 1.0f;
	float m_cameraNear = 1.0f;
	float m_cameraFar = 1000.0f;

	mat4 m_worldMatrix = mat4(1.0f);
	mat4 m_viewMatrix = mat4(1.0f);
	mat4 m_projectionMatrix = mat4(1.0f);
	mat4 m_viewPojectionMatrix = mat4(1.0f);

	// If textures are used in ImGui, they must be kept alive until ImGui rendering is complete
	std::vector<TexturePtr> m_guiTextureReferences;
};

static void cbWindowSize(GLFWwindow* window, int w, int h)
{
	ProbulatorGui* app = (ProbulatorGui*)glfwGetWindowUserPointer(window);
	app->setWindowSize(ivec2(w, h));
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

	glfwSwapInterval(1); // vsync ON

	do
	{
		ImGui_ImplGlfwGL3_NewFrame();

		app->updateImGui();
		app->render();

		glfwSwapBuffers(window);
		glfwPollEvents();
	} while(!glfwWindowShouldClose(window));

	delete app;

	ImGui_ImplGlfwGL3_Shutdown();

	glfwTerminate();

	return 0;
}
