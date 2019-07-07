#include <iostream>
#include <fstream>

#include "ExperimentAmbientDice.h"

#include <Eigen/Eigen>
#include <Eigen/nnls.h>

namespace Probulator {
    
    const float AmbientDice::kT = 0.6180339887498949;
    const float AmbientDice::kT2 = kT * kT;
    
    const vec3 AmbientDice::vertexPositions[12] =
    {
        vec3(1.0, kT, 0.0),
        vec3(-1.0, kT, 0.0),
        vec3(1.0, -kT, -0.0),
        vec3(-1.0, -kT, 0.0),
        vec3(0.0, 1.0, kT),
        vec3(-0.0, -1.0, kT),
        vec3(0.0, 1.0, -kT),
        vec3(0.0, -1.0, -kT),
        vec3(kT, 0.0, 1.0),
        vec3(-kT, 0.0, 1.0),
        vec3(kT, -0.0, -1.0),
        vec3(-kT, -0.0, -1.0)
    };
    
    const vec3 AmbientDice::srbfNormalisedVertexPositions[6] =
    {
        normalize(vec3(1.0, kT, 0.0)),
        normalize(vec3(-1.0, kT, 0.0)),
        normalize(vec3(0.0, 1.0, kT)),
        normalize(vec3(-0.0, -1.0, kT)),
        normalize(vec3(kT, 0.0, 1.0)),
        normalize(vec3(kT, -0.0, -1.0))
    };
    
    // Arbitrary orthonormal basis constructed around each vertex.
    const vec3 AmbientDice::tangents[12] =
    {
        vec3(0.27639312, -0.44721365, -0.85065085),
        vec3(0.27639312, 0.44721365, 0.85065085),
        vec3(0.27639312, 0.44721365, 0.85065085),
        vec3(0.27639312, -0.44721365, 0.85065085),
        vec3(1.0, -0.0, -0.0),
        vec3(1.0, -0.0, 0.0),
        vec3(1.0, -0.0, 0.0),
        vec3(1.0, 0.0, 0.0),
        vec3(0.8506508, -0.0, -0.52573115),
        vec3(0.8506508, 0.0, 0.52573115),
        vec3(0.8506508, 0.0, 0.52573115),
        vec3(0.8506508, -0.0, -0.52573115)
    };
    
    // Arbitrary orthonormal basis constructed around each vertex.
    const vec3 AmbientDice::bitangents[12] =
    {
        vec3(-0.44721365, 0.72360677, -0.52573115),
        vec3(0.44721365, 0.72360677, -0.52573115),
        vec3(-0.44721365, -0.72360677, 0.52573115),
        vec3(-0.44721365, 0.72360677, 0.52573115),
        vec3(-0.0, 0.525731, -0.85065085),
        vec3(-0.0, 0.525731, 0.85065085),
        vec3(0.0, -0.525731, -0.85065085),
        vec3(-0.0, -0.525731, 0.85065085),
        vec3(-0.0, 1.0, -0.0),
        vec3(0.0, 1.0, -0.0),
        vec3(-0.0, -1.0, 0.0),
        vec3(0.0, -1.0, 0.0)
    };
    
    const u32 AmbientDice::triangleIndices[20][3] =
    {
        { 0, 4, 8 },
        { 1, 4, 9 },
        { 2, 5, 8 },
        { 3, 5, 9 },
        { 0, 6, 10 },
        { 1, 6, 11 },
        { 2, 7, 10 },
        { 3, 7, 11 },
        { 4, 8, 9 },
        { 5, 8, 9 },
        { 6, 10, 11 },
        { 7, 10, 11 },
        { 0, 2, 8 },
        { 1, 3, 9 },
        { 0, 2, 10 },
        { 1, 3, 11 },
        { 0, 4, 6 },
        { 1, 4, 6 },
        { 2, 5, 7 },
        { 3, 5, 7 },
    };
    
    const vec3 AmbientDice::triangleBarycentricNormals[20][3] =
    {
        { vec3(0.9510565, 0.36327127, -0.58778524), vec3(-0.58778524, 0.9510565, 0.36327127), vec3(0.36327127, -0.58778524, 0.9510565) },
        { vec3(-0.9510565, 0.36327127, -0.58778524), vec3(0.58778524, 0.9510565, 0.36327127), vec3(-0.36327127, -0.58778524, 0.9510565) },
        { vec3(0.9510565, -0.36327127, -0.58778524), vec3(-0.58778524, -0.9510565, 0.36327127), vec3(0.36327127, 0.58778524, 0.9510565) },
        { vec3(-0.9510565, -0.36327127, -0.58778524), vec3(0.58778524, -0.9510565, 0.36327127), vec3(-0.36327127, 0.58778524, 0.9510565) },
        { vec3(0.9510565, 0.36327127, 0.58778524), vec3(-0.58778524, 0.9510565, -0.36327127), vec3(0.36327127, -0.58778524, -0.9510565) },
        { vec3(-0.9510565, 0.36327127, 0.58778524), vec3(0.58778524, 0.9510565, -0.36327127), vec3(-0.36327127, -0.58778524, -0.9510565) },
        { vec3(0.9510565, -0.36327127, 0.58778524), vec3(-0.58778524, -0.9510565, -0.36327127), vec3(0.36327127, 0.58778524, -0.9510565) },
        { vec3(-0.9510565, -0.36327127, 0.58778524), vec3(0.58778524, -0.9510565, -0.36327127), vec3(-0.36327127, 0.58778524, -0.9510565) },
        { vec3(-0.0, 1.1755705, -0.0), vec3(0.9510565, -0.36327127, 0.58778524), vec3(-0.9510565, -0.36327127, 0.58778524) },
        { vec3(0.0, -1.1755705, 0.0), vec3(0.9510565, 0.36327127, 0.58778524), vec3(-0.9510565, 0.36327127, 0.58778524) },
        { vec3(0.0, 1.1755705, -0.0), vec3(0.9510565, -0.36327127, -0.58778524), vec3(-0.9510565, -0.36327127, -0.58778524) },
        { vec3(-0.0, -1.1755705, 0.0), vec3(0.9510565, 0.36327127, -0.58778524), vec3(-0.9510565, 0.36327127, -0.58778524) },
        { vec3(0.58778524, 0.9510565, -0.36327127), vec3(0.58778524, -0.9510565, -0.36327127), vec3(-0.0, -0.0, 1.1755705) },
        { vec3(-0.58778524, 0.9510565, -0.36327127), vec3(-0.58778524, -0.9510565, -0.36327127), vec3(0.0, 0.0, 1.1755705) },
        { vec3(0.58778524, 0.9510565, 0.36327127), vec3(0.58778524, -0.9510565, 0.36327127), vec3(0.0, 0.0, -1.1755705) },
        { vec3(-0.58778524, 0.9510565, 0.36327127), vec3(-0.58778524, -0.9510565, 0.36327127), vec3(-0.0, -0.0, -1.1755705) },
        { vec3(1.1755705, -0.0, -0.0), vec3(-0.36327127, 0.58778524, 0.9510565), vec3(-0.36327127, 0.58778524, -0.9510565) },
        { vec3(-1.1755705, 0.0, 0.0), vec3(0.36327127, 0.58778524, 0.9510565), vec3(0.36327127, 0.58778524, -0.9510565) },
        { vec3(1.1755705, 0.0, 0.0), vec3(-0.36327127, -0.58778524, 0.9510565), vec3(-0.36327127, -0.58778524, -0.9510565) },
        { vec3(-1.1755705, -0.0, -0.0), vec3(0.36327127, -0.58778524, 0.9510565), vec3(0.36327127, -0.58778524, -0.9510565) },
    };
    
    // 1 / (3 * alpha) times the projection of the edge vectors onto the vertex tangents.
    const float AmbientDice::triDerivativeTangentFactors[20][6] =
    {
        { -0.34100485f, -0.238272f, 0.3504874f, 0.21661313f, 0.2981424f, -0.11388027f },
        { 0.34100485f, 0.238272f, -0.3504874f, -0.21661313f, -0.2981424f, 0.11388027f },
        { 0.027519437f, 0.35801283f, 0.3504874f, 0.21661313f, 0.2981424f, -0.11388027f },
        { 0.34100485f, 0.238272f, -0.3504874f, -0.21661313f, -0.2981424f, 0.11388027f },
        { 0.027519437f, 0.35801283f, 0.3504874f, 0.21661313f, 0.2981424f, -0.11388027f },
        { -0.027519437f, -0.35801283f, -0.3504874f, -0.21661313f, -0.2981424f, 0.11388027f },
        { -0.34100485f, -0.238272f, 0.3504874f, 0.21661313f, 0.2981424f, -0.11388027f },
        { -0.027519437f, -0.35801283f, -0.3504874f, -0.21661313f, -0.2981424f, 0.11388027f },
        { 0.21661313f, -0.21661313f, -0.11388027f, -0.36852428f, 0.11388027f, 0.36852428f },
        { 0.21661313f, -0.21661313f, -0.11388027f, -0.36852428f, 0.11388027f, 0.36852428f },
        { 0.21661313f, -0.21661313f, -0.11388027f, -0.36852428f, 0.11388027f, 0.36852428f },
        { 0.21661313f, -0.21661313f, -0.11388027f, -0.36852428f, 0.11388027f, 0.36852428f },
        { 0.1937447f, -0.238272f, 0.1937447f, 0.35801283f, 0.2981424f, 0.2981424f },
        { -0.1937447f, 0.238272f, -0.1937447f, 0.238272f, -0.2981424f, -0.2981424f },
        { 0.1937447f, 0.35801283f, 0.1937447f, -0.238272f, 0.2981424f, 0.2981424f },
        { -0.1937447f, -0.35801283f, -0.1937447f, -0.35801283f, -0.2981424f, -0.2981424f },
        { -0.34100485f, 0.027519437f, 0.3504874f, 0.0f, 0.3504874f, 0.0f },
        { 0.34100485f, -0.027519437f, -0.3504874f, 0.0f, -0.3504874f, 0.0f },
        { 0.027519437f, -0.34100485f, 0.3504874f, 0.0f, 0.3504874f, 0.0f },
        { 0.34100485f, -0.027519437f, -0.3504874f, 0.0f, -0.3504874f, 0.0f }
    };
    
    // 1 / (3 * alpha) times the projection of the edge vectors onto the vertex bitangents.
    const float AmbientDice::triDerivativeBitangentFactors[20][6] =
    {
        { 0.1397348f, -0.2811345f, 0.11388027f, -0.2981424f, 0.21661313f, 0.3504874f },
        { 0.1397348f, -0.2811345f, 0.11388027f, -0.2981424f, 0.21661313f, 0.3504874f },
        { 0.36749536f, 0.08738982f, -0.11388027f, 0.2981424f, -0.21661313f, -0.3504874f },
        { -0.1397348f, 0.2811345f, -0.11388027f, 0.2981424f, -0.21661313f, -0.3504874f },
        { 0.36749536f, 0.08738982f, -0.11388027f, 0.2981424f, -0.21661313f, -0.3504874f },
        { 0.36749536f, 0.08738982f, -0.11388027f, 0.2981424f, -0.21661313f, -0.3504874f },
        { 0.1397348f, -0.2811345f, 0.11388027f, -0.2981424f, 0.21661313f, 0.3504874f },
        { -0.36749536f, -0.08738982f, 0.11388027f, -0.2981424f, 0.21661313f, 0.3504874f },
        { -0.2981424f, -0.2981424f, 0.3504874f, 0.0f, 0.3504874f, 0.0f },
        { 0.2981424f, 0.2981424f, -0.3504874f, 0.0f, -0.3504874f, 0.0f },
        { 0.2981424f, 0.2981424f, -0.3504874f, 0.0f, -0.3504874f, 0.0f },
        { -0.2981424f, -0.2981424f, 0.3504874f, 0.0f, 0.3504874f, 0.0f },
        { -0.31348547f, -0.2811345f, -0.31348547f, 0.08738982f, 0.21661313f, -0.21661313f },
        { -0.31348547f, -0.2811345f, 0.31348547f, 0.2811345f, 0.21661313f, -0.21661313f },
        { -0.31348547f, 0.08738982f, -0.31348547f, -0.2811345f, -0.21661313f, 0.21661313f },
        { -0.31348547f, 0.08738982f, 0.31348547f, -0.08738982f, -0.21661313f, 0.21661313f },
        { 0.1397348f, 0.36749536f, 0.11388027f, 0.36852428f, -0.11388027f, -0.36852428f },
        { 0.1397348f, 0.36749536f, 0.11388027f, 0.36852428f, -0.11388027f, -0.36852428f },
        { 0.36749536f, 0.1397348f, -0.11388027f, -0.36852428f, 0.11388027f, 0.36852428f },
        { -0.1397348f, -0.36749536f, -0.11388027f, -0.36852428f, 0.11388027f, 0.36852428f }
    };
    
    template <typename T>
    void AmbientDice::hybridCubicBezierWeights(u32 triIndex, float b0, float b1, float b2, VertexWeights<T> *w0Out, VertexWeights<T> *w1Out, VertexWeights<T> *w2Out)
    {
        const T alpha = 0.5f * sqrt(0.5 * (5.0f + sqrt(5.0f))); // 0.9510565163
        const T beta = -0.5f * sqrt(0.1 * (5.0f + sqrt(5.0f))); // -0.4253254042
        
        const T a0 = (sqrt(5.0) - 5.0) / 40.0; // -0.06909830056
        const T a1 = (11.0f * sqrt(5.0) - 15.0) / 40.0; // 0.2399186938
        const T a2 = sqrt(5.0) / 10.0; // 0.2236067977
        
        const T fValueFactor = -beta / alpha; // 0.4472135955
        
        const T weightDenom = b1 * b2 + b0 * b2 + b0 * b1;
        
        T w0 = (b1 * b2) / weightDenom;
        T w1 = (b0 * b2) / weightDenom;
        T w2 = (b0 * b1) / weightDenom;
        
        if (b0 == 1.0) {
            w0 = 1.0;
            w1 = 0.0;
            w2 = 0.0;
        } else if (b1 == 1.0) {
            w0 = 0.0;
            w1 = 1.0;
            w2 = 0.0;
        } else if (b2 == 1.0) {
            w0 = 0.0;
            w1 = 0.0;
            w2 = 1.0;
        }
        
        // https://en.wikipedia.org/wiki/Bézier_triangle
        // Notation: cxyz means alpha^x, beta^y, gamma^z.
        
        T v0ValueWeight = 0.0;
        T v1ValueWeight = 0.0;
        T v2ValueWeight = 0.0;
        
        T v0DUWeight = 0.0;
        T v1DUWeight = 0.0;
        T v2DUWeight = 0.0;
        
        T v0DVWeight = 0.0;
        T v1DVWeight = 0.0;
        T v2DVWeight = 0.0;
        
        const T b0_2 = b0 * b0;
        const T b1_2 = b1 * b1;
        const T b2_2 = b2 * b2;
        
        // Add c300, c030, and c003
        T c300Weight = b0_2 * b0;
        T c030Weight = b1_2 * b1;
        T c003Weight = b2_2 * b2;
        
        T c120Weight = 3 * b0 * b1_2;
        T c021Weight = 3 * b1_2 * b2;
        T c210Weight = 3 * b0_2 * b1;
        T c012Weight = 3 * b1 * b2_2;
        T c201Weight = 3 * b0_2 * b2;
        T c102Weight = 3 * b0 * b2_2;
        
        const T c111Weight = 6 * b0 * b1 * b2;
        const T c0_111Weight = w0 * c111Weight;
        const T c1_111Weight = w1 * c111Weight;
        const T c2_111Weight = w2 * c111Weight;
        
        v1ValueWeight += a0 * c0_111Weight;
        v2ValueWeight += a0 * c1_111Weight;
        v0ValueWeight += a0 * c2_111Weight;
        
        c021Weight += a1 * c0_111Weight;
        c012Weight += a1 * c0_111Weight;
        c003Weight += a0 * c0_111Weight;
        c120Weight += a2 * c0_111Weight;
        c102Weight += a2 * c0_111Weight;
        
        c102Weight += a1 * c1_111Weight;
        c201Weight += a1 * c1_111Weight;
        c300Weight += a0 * c1_111Weight;
        c012Weight += a2 * c1_111Weight;
        c210Weight += a2 * c1_111Weight;
        
        c210Weight += a1 * c2_111Weight;
        c120Weight += a1 * c2_111Weight;
        c030Weight += a0 * c2_111Weight;
        c201Weight += a2 * c2_111Weight;
        c021Weight += a2 * c2_111Weight;
        
        v0ValueWeight += fValueFactor * c210Weight;
        v0DUWeight += AmbientDice::triDerivativeTangentFactors[triIndex][0] * c210Weight;
        v0DVWeight += AmbientDice::triDerivativeBitangentFactors[triIndex][0] * c210Weight;
        
        v0ValueWeight += fValueFactor * c201Weight;
        v0DUWeight += AmbientDice::triDerivativeTangentFactors[triIndex][1] * c201Weight;
        v0DVWeight += AmbientDice::triDerivativeBitangentFactors[triIndex][1] * c201Weight;
        
        v1ValueWeight += fValueFactor * c120Weight;
        v1DUWeight += AmbientDice::triDerivativeTangentFactors[triIndex][2] * c120Weight;
        v1DVWeight += AmbientDice::triDerivativeBitangentFactors[triIndex][2] * c120Weight;
        
        v1ValueWeight += fValueFactor * c021Weight;
        v1DUWeight += AmbientDice::triDerivativeTangentFactors[triIndex][3] * c021Weight;
        v1DVWeight += AmbientDice::triDerivativeBitangentFactors[triIndex][3] * c021Weight;
        
        v2ValueWeight += fValueFactor * c102Weight;
        v2DUWeight += AmbientDice::triDerivativeTangentFactors[triIndex][4] * c102Weight;
        v2DVWeight += AmbientDice::triDerivativeBitangentFactors[triIndex][4] * c102Weight;
        
        v2ValueWeight += fValueFactor * c012Weight;
        v2DUWeight += AmbientDice::triDerivativeTangentFactors[triIndex][5] * c012Weight;
        v2DVWeight += AmbientDice::triDerivativeBitangentFactors[triIndex][5] * c012Weight;
        
        v0ValueWeight += c300Weight;
        v1ValueWeight += c030Weight;
        v2ValueWeight += c003Weight;
        
        *w0Out = { v0ValueWeight, v0DUWeight, v0DVWeight };
        *w1Out = { v1ValueWeight, v1DUWeight, v1DVWeight };
        *w2Out = { v2ValueWeight, v2DUWeight, v2DVWeight };
    }
    
    template<typename T>
    void AmbientDice::hybridCubicBezierWeights(vec3 direction, u32 *i0Out, u32 *i1Out, u32 *i2Out, VertexWeights<T> *w0Out, VertexWeights<T> *w1Out, VertexWeights<T> *w2Out)
    {
        
        u32 triIndex, i0, i1, i2;
        float b0, b1, b2;
        AmbientDice::computeBarycentrics(direction, &triIndex, &i0, &i1, &i2, &b0, &b1, &b2);
        
        AmbientDice::hybridCubicBezierWeights(triIndex, b0, b1, b2, w0Out, w1Out, w2Out);
        
        *i0Out = i0;
        *i1Out = i1;
        *i2Out = i2;
    }
    
    template<typename T>
    void AmbientDice::srbfWeights(vec3 direction, T *weightsOut)
    {
        for (u64 i = 0; i < 6; i += 1) {
            float dotProduct = dot(direction, AmbientDice::srbfNormalisedVertexPositions[i]);
            u32 index = dotProduct > 0 ? (2 * i) : (2 * i + 1);
            
            T cos2 = dotProduct * dotProduct;
            T cos4 = cos2 * cos2;
            
            weightsOut[index] = 0.7f * (0.5f * cos2) + 0.3f * (5.f / 6.f * cos4);
        }
    }
    
    vec3 AmbientDice::evaluateSRBF(const vec3& direction) const
    {
        vec3 result = vec3(0.f);
            for (u64 i = 0; i < 6; i += 1) {
                float dotProduct = dot(direction, AmbientDice::srbfNormalisedVertexPositions[i]);
                u32 index = dotProduct > 0 ? (2 * i) : (2 * i + 1);
                
                float cos2 = dotProduct * dotProduct;
                float cos4 = cos2 * cos2;
                
                float weight = 0.7f * (0.5f * cos2) + 0.3f * (5.f / 6.f * cos4);
                
                result += weight * this->vertices[index].value;
            }
        
        return result;
    }
    
    
    Eigen::MatrixXf AmbientDice::computeGramMatrixBezier()
    {
        using namespace Eigen;
        
        const u64 sampleCount = 32768;
        float sampleScale = 4 * M_PI / float(sampleCount);
        
        AmbientDice ambientDice;
        
        MatrixXf gram = MatrixXf::Zero(36, 36);
        
        for (u64 sampleIt = 0; sampleIt < sampleCount; sampleIt += 1) {
            vec2 sample = sampleHammersley(sampleIt, sampleCount);
            vec3 direction = sampleUniformSphere(sample.x, sample.y);
            
            float allWeights[36] = { 0.f };
            
            u32 i0, i1, i2;
            AmbientDice::VertexWeights<float> weights[3];
            AmbientDice::hybridCubicBezierWeights(direction, &i0, &i1, &i2, &weights[0], &weights[1], &weights[2]);
            
            allWeights[3 * i0 + 0] = weights[0].value;
            allWeights[3 * i0 + 1] = weights[0].directionalDerivativeU;
            allWeights[3 * i0 + 2] = weights[0].directionalDerivativeV;
            
            allWeights[3 * i1 + 0] = weights[1].value;
            allWeights[3 * i1 + 1] = weights[1].directionalDerivativeU;
            allWeights[3 * i1 + 2] = weights[1].directionalDerivativeV;
            
            allWeights[3 * i2 + 0] = weights[2].value;
            allWeights[3 * i2 + 1] = weights[2].directionalDerivativeU;
            allWeights[3 * i2 + 2] = weights[2].directionalDerivativeV;
            
            for (u64 lobeAIt = 0; lobeAIt < 36; ++lobeAIt)
            {
                for (u64 lobeBIt = lobeAIt; lobeBIt < 36; ++lobeBIt)
                {
                    float delta = allWeights[lobeAIt] * allWeights[lobeBIt] * sampleScale;
                    gram(lobeAIt, lobeBIt) += delta;
                    
                    if (lobeBIt != lobeAIt) {
                        gram(lobeBIt, lobeAIt) += delta;
                    }
                }
            }
        }
        
        return gram;
    }
    
    Eigen::MatrixXf AmbientDice::computeGramMatrixLinear()
    {
        using namespace Eigen;
        
        const u64 sampleCount = 32768;
        float sampleScale = 4 * M_PI / float(sampleCount);
        
        AmbientDice ambientDice;
        
        MatrixXf gram = MatrixXf::Zero(12, 12);
        
        for (u64 sampleIt = 0; sampleIt < sampleCount; sampleIt += 1) {
            vec2 sample = sampleHammersley(sampleIt, sampleCount);
            vec3 direction = sampleUniformSphere(sample.x, sample.y);
            
            u32 triIndex;
            u32 i0, i1, i2;
            float b0, b1, b2;
            AmbientDice::computeBarycentrics(direction, &triIndex, &i0, &i1, &i2, &b0, &b1, &b2);
            
            gram(i0, i0) += b0 * b0 * sampleScale;
            gram(i0, i1) += b0 * b1 * sampleScale;
            gram(i0, i2) += b0 * b2 * sampleScale;
            gram(i1, i1) += b1 * b1 * sampleScale;
            gram(i1, i2) += b1 * b2 * sampleScale;
            gram(i2, i2) += b2 * b2 * sampleScale;
        }
        
        for (u64 lobeAIt = 0; lobeAIt < 12; ++lobeAIt)
        {
            for (u64 lobeBIt = lobeAIt; lobeBIt < 12; ++lobeBIt)
            {
                gram(lobeBIt, lobeAIt) = gram(lobeAIt, lobeBIt);
            }
        }
        
        return gram;
    }
    
    Eigen::MatrixXf AmbientDice::computeGramMatrixSRBF()
    {
        using namespace Eigen;
        
        const u64 sampleCount = 32768;
        float sampleScale = 4 * M_PI / float(sampleCount);
        
        AmbientDice ambientDice;
        
        MatrixXf gram = MatrixXf::Zero(12, 12);
        
        for (u64 sampleIt = 0; sampleIt < sampleCount; sampleIt += 1) {
            vec2 sample = sampleHammersley(sampleIt, sampleCount);
            vec3 direction = sampleUniformSphere(sample.x, sample.y);
            
            float allWeights[12] = { 0.f };
            AmbientDice::srbfWeights(direction, allWeights);
            
            for (u64 lobeAIt = 0; lobeAIt < 12; ++lobeAIt)
            {
                for (u64 lobeBIt = lobeAIt; lobeBIt < 12; ++lobeBIt)
                {
                    float delta = allWeights[lobeAIt] * allWeights[lobeBIt] * sampleScale;
                    gram(lobeAIt, lobeBIt) += delta;
                    
                    if (lobeBIt != lobeAIt) {
                        gram(lobeBIt, lobeAIt) += delta;
                    }
                }
            }
        }
        
        return gram;
    }
    
    AmbientDice ExperimentAmbientDice::solveAmbientDiceLeastSquaresLinear(ImageBase<vec3>& directions, const Image& irradiance)
    {
        using namespace Eigen;
        
        AmbientDice ambientDice;
        
        VectorXf momentsR = VectorXf::Zero(12);
        VectorXf momentsG = VectorXf::Zero(12);
        VectorXf momentsB = VectorXf::Zero(12);
        
        ivec2 imageSize = directions.getSize();
        directions.forPixels2D([&](const vec3& direction, ivec2 pixelPos)
                               {
                                   float texelArea = latLongTexelArea(pixelPos, imageSize);
                                   
                                   const vec4& color = irradiance.at(pixelPos);
                                   
                                   u32 i0, i1, i2;
                                   u32 triIndex;
                                   float b0, b1, b2;
                                   AmbientDice::computeBarycentrics(direction, &triIndex, &i0, &i1, &i2, &b0, &b1, &b2);
                                   
                                   momentsR[i0] += b0 * color.r * texelArea;
                                   momentsR[i1] += b1 * color.r * texelArea;
                                   momentsR[i2] += b2 * color.r * texelArea;
                                   
                                   momentsG[i0] += b0 * color.g * texelArea;
                                   momentsG[i1] += b1 * color.g * texelArea;
                                   momentsG[i2] += b2 * color.g * texelArea;
                                   
                                   momentsB[i0] += b0 * color.b * texelArea;
                                   momentsB[i1] += b1 * color.b * texelArea;
                                   momentsB[i2] += b2 * color.b * texelArea;
                               });
        
        MatrixXf gramLinear = AmbientDice::computeGramMatrixLinear();
        
        const auto solver = gramLinear.jacobiSvd(ComputeThinU | ComputeThinV);
        VectorXf R = solver.solve(momentsR);
        VectorXf G = solver.solve(momentsG);
        VectorXf B = solver.solve(momentsB);
        
        for (u64 basisIt = 0; basisIt < 12; ++basisIt)
        {
            ambientDice.vertices[basisIt].value[0] = R[basisIt];
            ambientDice.vertices[basisIt].value[1] = G[basisIt];
            ambientDice.vertices[basisIt].value[2] = B[basisIt];
        }
        
        return ambientDice;
    }
    
    AmbientDice ExperimentAmbientDice::solveAmbientDiceLeastSquaresBezier(ImageBase<vec3>& directions, const Image& irradiance)
    {
        using namespace Eigen;
        
        AmbientDice ambientDice;
        
        MatrixXf moments = MatrixXf::Zero(36, 3);
        
        ivec2 imageSize = directions.getSize();
        directions.forPixels2D([&](const vec3& direction, ivec2 pixelPos)
                               {
                                   float texelArea = latLongTexelArea(pixelPos, imageSize);
                                   
                                   const vec4& color = irradiance.at(pixelPos);
                                   
                                   u32 i0, i1, i2;
                                   AmbientDice::VertexWeights<float> weights[3];
                                   AmbientDice::hybridCubicBezierWeights(direction, &i0, &i1, &i2, &weights[0], &weights[1], &weights[2]);
                                   
                                   moments(3 * i0 + 0, 0) += weights[0].value * color.r * texelArea;
                                   moments(3 * i0 + 1, 0) += weights[0].directionalDerivativeU * color.r * texelArea;
                                   moments(3 * i0 + 2, 0) += weights[0].directionalDerivativeV * color.r * texelArea;
                                   moments(3 * i1 + 0, 0) += weights[1].value * color.r * texelArea;
                                   moments(3 * i1 + 1, 0) += weights[1].directionalDerivativeU * color.r * texelArea;
                                   moments(3 * i1 + 2, 0) += weights[1].directionalDerivativeV * color.r * texelArea;
                                   moments(3 * i2 + 0, 0) += weights[2].value * color.r * texelArea;
                                   moments(3 * i2 + 1, 0) += weights[2].directionalDerivativeU * color.r * texelArea;
                                   moments(3 * i2 + 2, 0) += weights[2].directionalDerivativeV * color.r * texelArea;
                                   moments(3 * i0 + 0, 1) += weights[0].value * color.g * texelArea;
                                   moments(3 * i0 + 1, 1) += weights[0].directionalDerivativeU * color.g * texelArea;
                                   moments(3 * i0 + 2, 1) += weights[0].directionalDerivativeV * color.g * texelArea;
                                   moments(3 * i1 + 0, 1) += weights[1].value * color.g * texelArea;
                                   moments(3 * i1 + 1, 1) += weights[1].directionalDerivativeU * color.g * texelArea;
                                   moments(3 * i1 + 2, 1) += weights[1].directionalDerivativeV * color.g * texelArea;
                                   moments(3 * i2 + 0, 1) += weights[2].value * color.g * texelArea;
                                   moments(3 * i2 + 1, 1) += weights[2].directionalDerivativeU * color.g * texelArea;
                                   moments(3 * i2 + 2, 1) += weights[2].directionalDerivativeV * color.g * texelArea;
                                   moments(3 * i0 + 0, 2) += weights[0].value * color.b * texelArea;
                                   moments(3 * i0 + 1, 2) += weights[0].directionalDerivativeU * color.b * texelArea;
                                   moments(3 * i0 + 2, 2) += weights[0].directionalDerivativeV * color.b * texelArea;
                                   moments(3 * i1 + 0, 2) += weights[1].value * color.b * texelArea;
                                   moments(3 * i1 + 1, 2) += weights[1].directionalDerivativeU * color.b * texelArea;
                                   moments(3 * i1 + 2, 2) += weights[1].directionalDerivativeV * color.b * texelArea;
                                   moments(3 * i2 + 0, 2) += weights[2].value * color.b * texelArea;
                                   moments(3 * i2 + 1, 2) += weights[2].directionalDerivativeU * color.b * texelArea;
                                   moments(3 * i2 + 2, 2) += weights[2].directionalDerivativeV * color.b * texelArea;
                               });
        
        MatrixXf gram = AmbientDice::computeGramMatrixBezier();
        
        auto solver = gram.jacobiSvd(ComputeThinU | ComputeThinV);
        
        VectorXf b;
        b.resize(36);
        
        for (u32 channelIt = 0; channelIt < 3; ++channelIt)
        {
            for (u64 lobeIt = 0; lobeIt < 36; ++lobeIt)
            {
                b[lobeIt] = moments(lobeIt, channelIt);
            }
            
            VectorXf x = solver.solve(b);
            
            for (u64 basisIt = 0; basisIt < 12; ++basisIt)
            {
                ambientDice.vertices[basisIt].value[channelIt] = x[3 * basisIt];
                ambientDice.vertices[basisIt].directionalDerivativeU[channelIt] = x[3 * basisIt + 1];
                ambientDice.vertices[basisIt].directionalDerivativeV[channelIt] = x[3 * basisIt + 2];
            }
        }
        
        return ambientDice;
    }
    
    AmbientDice ExperimentAmbientDice::solveAmbientDiceLeastSquaresBezierYCoCg(ImageBase<vec3>& directions, const Image& irradiance)
    {
        using namespace Eigen;
        
        AmbientDice ambientDice;
        
        VectorXf momentsY = VectorXf::Zero(36);
        VectorXf momentsCo = VectorXf::Zero(12);
        VectorXf momentsCg = VectorXf::Zero(12);
        
        ivec2 imageSize = directions.getSize();
        directions.forPixels2D([&](const vec3& direction, ivec2 pixelPos)
                               {
                                   float texelArea = latLongTexelArea(pixelPos, imageSize);
                                   
                                   const vec4& color = irradiance.at(pixelPos);
                                   
                                   vec3 colorYCoCg = rgbToYCoCg(vec3(color.r, color.g, color.b));
                                   
                                   u32 i0, i1, i2;
                                   AmbientDice::VertexWeights<float> weights[3];
                                   AmbientDice::hybridCubicBezierWeights(direction, &i0, &i1, &i2, &weights[0], &weights[1], &weights[2]);
                                   
                                   u32 triIndex;
                                   float b0, b1, b2;
                                   AmbientDice::computeBarycentrics(direction, &triIndex, &i0, &i1, &i2, &b0, &b1, &b2);
                                   
                                   momentsY[3 * i0 + 0] += weights[0].value * colorYCoCg.r * texelArea;
                                   momentsY[3 * i0 + 1] += weights[0].directionalDerivativeU * colorYCoCg.r * texelArea;
                                   momentsY[3 * i0 + 2] += weights[0].directionalDerivativeV * colorYCoCg.r * texelArea;
                                   momentsY[3 * i1 + 0] += weights[1].value * colorYCoCg.r * texelArea;
                                   momentsY[3 * i1 + 1] += weights[1].directionalDerivativeU * colorYCoCg.r * texelArea;
                                   momentsY[3 * i1 + 2] += weights[1].directionalDerivativeV * colorYCoCg.r * texelArea;
                                   momentsY[3 * i2 + 0] += weights[2].value * colorYCoCg.r * texelArea;
                                   momentsY[3 * i2 + 1] += weights[2].directionalDerivativeU * colorYCoCg.r * texelArea;
                                   momentsY[3 * i2 + 2] += weights[2].directionalDerivativeV * colorYCoCg.r * texelArea;
                                   
                                   momentsCo[i0] += b0 * colorYCoCg.g * texelArea;
                                   momentsCo[i1] += b1 * colorYCoCg.g * texelArea;
                                   momentsCo[i2] += b2 * colorYCoCg.g * texelArea;
                                   
                                   momentsCg[i0] += b0 * colorYCoCg.b * texelArea;
                                   momentsCg[i1] += b1 * colorYCoCg.b * texelArea;
                                   momentsCg[i2] += b2 * colorYCoCg.b * texelArea;
                               });
        
        MatrixXf gramBezier = AmbientDice::computeGramMatrixBezier();
        MatrixXf gramLinear = AmbientDice::computeGramMatrixLinear();
        
        VectorXf Y = gramBezier.jacobiSvd(ComputeThinU | ComputeThinV).solve(momentsY);
        
        const auto linearSolver = gramLinear.jacobiSvd(ComputeThinU | ComputeThinV);
        VectorXf Co = linearSolver.solve(momentsCo);
        VectorXf Cg = linearSolver.solve(momentsCg);
        
        for (u64 basisIt = 0; basisIt < 12; ++basisIt)
        {
            ambientDice.vertices[basisIt].value[0] = Y[3 * basisIt];
            ambientDice.vertices[basisIt].directionalDerivativeU[0] = Y[3 * basisIt + 1];
            ambientDice.vertices[basisIt].directionalDerivativeV[0] = Y[3 * basisIt + 2];
            
            ambientDice.vertices[basisIt].value[1] = Co[basisIt];
            ambientDice.vertices[basisIt].value[2] = Cg[basisIt];
        }
        
        return ambientDice;
    }
    
    AmbientDice ExperimentAmbientDice::solveAmbientDiceLeastSquaresSRBF(ImageBase<vec3>& directions, const Image& irradiance)
    {
        using namespace Eigen;
        
        AmbientDice ambientDice;
        
        VectorXf momentsR = VectorXf::Zero(12);
        VectorXf momentsG = VectorXf::Zero(12);
        VectorXf momentsB = VectorXf::Zero(12);
        
        ivec2 imageSize = directions.getSize();
        directions.forPixels2D([&](const vec3& direction, ivec2 pixelPos)
                               {
                                   float texelArea = latLongTexelArea(pixelPos, imageSize);
                                   
                                   const vec4& color = irradiance.at(pixelPos);
                                   
                                   float weights[12] = { 0.f };
                                   AmbientDice::srbfWeights(direction, weights);
                                   
                                   for (size_t i = 0; i < 12; i += 1)
                                   {
                                       momentsR[i] += weights[i] * color.r * texelArea;
                                       momentsG[i] += weights[i] * color.g * texelArea;
                                       momentsB[i] += weights[i] * color.b * texelArea;
                                       
                                   }
                               });
        
        MatrixXf gram = AmbientDice::computeGramMatrixSRBF();
        
        const auto solver = gram.jacobiSvd(ComputeThinU | ComputeThinV);
        VectorXf R = solver.solve(momentsR);
        VectorXf G = solver.solve(momentsG);
        VectorXf B = solver.solve(momentsB);
        
        for (u64 basisIt = 0; basisIt < 12; ++basisIt)
        {
            ambientDice.vertices[basisIt].value[0] = R[basisIt];
            ambientDice.vertices[basisIt].value[1] = G[basisIt];
            ambientDice.vertices[basisIt].value[2] = B[basisIt];
        }
        
        return ambientDice;
    }
    
    void ExperimentAmbientDice::run(SharedData& data)
    {
        
        using namespace Eigen;
        
        m_radianceImage = Image(data.m_outputSize);
        m_irradianceImage = Image(data.m_outputSize);
        
        if (m_diceType == AmbientDiceTypeBezier)
        {
            AmbientDice ambientDiceRadiance = solveAmbientDiceLeastSquaresBezier(data.m_directionImage, m_input->m_radianceImage);
            AmbientDice ambientDiceIrradiance = solveAmbientDiceLeastSquaresBezier(data.m_directionImage, m_input->m_irradianceImage);
            
            data.m_directionImage.parallelForPixels2D([&](const vec3& direction, ivec2 pixelPos)
                                                      {
                                                          vec3 sampleRadiance = ambientDiceRadiance.evaluateBezier(direction);
                                                          m_radianceImage.at(pixelPos) = vec4(max(sampleRadiance, vec3(0.f)), 1.0f);
                                                          
                                                          vec3 sampleIrradiance = ambientDiceIrradiance.evaluateBezier(direction);
                                                          m_irradianceImage.at(pixelPos) = vec4(sampleIrradiance, 1.0f);
                                                      });
        }
        else if (m_diceType == AmbientDiceTypeBezierYCoCg)
        {
            AmbientDice ambientDiceRadiance = solveAmbientDiceLeastSquaresBezierYCoCg(data.m_directionImage, m_input->m_radianceImage);
            AmbientDice ambientDiceIrradiance = solveAmbientDiceLeastSquaresBezierYCoCg(data.m_directionImage, m_input->m_irradianceImage);
            
            data.m_directionImage.parallelForPixels2D([&](const vec3& direction, ivec2 pixelPos)
                                                      {
                                                          vec3 sampleRadiance = ambientDiceRadiance.evaluateBezierYCoCg(direction);
                                                          m_radianceImage.at(pixelPos) = vec4(max(sampleRadiance, vec3(0.f)), 1.0f);
                                                          
                                                          vec3 sampleIrradiance = ambientDiceIrradiance.evaluateBezierYCoCg(direction);
                                                          m_irradianceImage.at(pixelPos) = vec4(sampleIrradiance, 1.0f);
                                                      });
        }
        else if (m_diceType == AmbientDiceTypeSRBF)
        {
            AmbientDice ambientDiceRadiance = solveAmbientDiceLeastSquaresSRBF(data.m_directionImage, m_input->m_radianceImage);
            AmbientDice ambientDiceIrradiance = solveAmbientDiceLeastSquaresSRBF(data.m_directionImage, m_input->m_irradianceImage);
            
            data.m_directionImage.parallelForPixels2D([&](const vec3& direction, ivec2 pixelPos)
                                                      {
                                                          vec3 sampleRadiance = ambientDiceRadiance.evaluateSRBF(direction);
                                                          m_radianceImage.at(pixelPos) = vec4(max(sampleRadiance, vec3(0.f)), 1.0f);
                                                          
                                                          vec3 sampleIrradiance = ambientDiceIrradiance.evaluateSRBF(direction);
                                                          m_irradianceImage.at(pixelPos) = vec4(sampleIrradiance, 1.0f);
                                                      });
        }  else if (m_diceType == AmbientDiceTypeLinear)
        {
            AmbientDice ambientDiceRadiance = solveAmbientDiceLeastSquaresLinear(data.m_directionImage, m_input->m_radianceImage);
            AmbientDice ambientDiceIrradiance = solveAmbientDiceLeastSquaresLinear(data.m_directionImage, m_input->m_irradianceImage);
            
            data.m_directionImage.parallelForPixels2D([&](const vec3& direction, ivec2 pixelPos)
                                                      {
                                                          vec3 sampleRadiance = ambientDiceRadiance.evaluateLinear(direction);
                                                          m_radianceImage.at(pixelPos) = vec4(max(sampleRadiance, vec3(0.f)), 1.0f);
                                                          
                                                          vec3 sampleIrradiance = ambientDiceIrradiance.evaluateLinear(direction);
                                                          m_irradianceImage.at(pixelPos) = vec4(sampleIrradiance, 1.0f);
                                                      });
        }
    }
}
