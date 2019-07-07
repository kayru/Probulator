#pragma once

#include <Probulator/Experiments.h>
#include <Eigen/Eigen>

// Ambient Dice
// Iwanicki and Sloan, 2018
// https://research.activision.com/t5/Publications/Ambient-Dice/ba-p/10284641
namespace Probulator {
    struct AmbientDice
    {
        
        static const float kT;
        static const float kT2;
        
        static const vec3 vertexPositions[12];
        static const vec3 srbfNormalisedVertexPositions[6];
        static const vec3 srbfHemisphereVertexPositions[10];
        static const vec3 tangents[12];
        static const vec3 bitangents[12];
        
        static const u32 triangleIndices[20][3];
        static const vec3 triangleBarycentricNormals[20][3];
        
        static const float triDerivativeTangentFactors[20][6];
        static const float triDerivativeBitangentFactors[20][6];
        
        struct Vertex
        {
            vec3 value;
            vec3 directionalDerivativeU;
            vec3 directionalDerivativeV;
        };
        
        template<typename T>
        struct VertexWeights
        {
            T value;
            T directionalDerivativeU;
            T directionalDerivativeV;
        };
        
        Vertex vertices[12];
        
        inline void indexIcosahedron(const vec3& direction, u32 *i0, u32 *i1, u32 *i2) const
        {
            float kT = AmbientDice::kT;
            float kT2 = kT * kT;
            
            ivec3 octantBit = ivec3(direction.x < 0 ? 1 : 0,
                                    direction.y < 0 ? 1 : 0,
                                    direction.z < 0 ? 1 : 0);
            
            ivec3 octantBitFlipped = ivec3(1) - octantBit;
            
            // Vertex indices
            u32 indexA = octantBit.y * 2 + octantBit.x + 0;
            u32 indexB = octantBit.z * 2 + octantBit.y + 4;
            u32 indexC = octantBit.z * 2 + octantBit.x + 8;
            
            u32 indexAFlipped = octantBit.z * 2 + octantBitFlipped.x + 8;
            u32 indexBFlipped = octantBitFlipped.y * 2 + octantBit.x + 0;
            u32 indexCFlipped = octantBitFlipped.y * 2 + octantBit.y + 4;
            
            // Selection
            bool vertASelect = dot(abs(direction), vec3(1.0, kT2, -kT)) > 0.0;
            bool vertBSelect = dot(abs(direction), vec3(-kT, 1.0, kT2)) > 0.0;
            bool vertCSelect = dot(abs(direction), vec3(kT2, -kT, 1.0)) > 0.0;
            
            *i0 = vertASelect ? indexA : indexAFlipped;
            *i1 = vertBSelect ? indexB : indexBFlipped;
            *i2 = vertCSelect ? indexC : indexCFlipped;
        }
        
        template <typename T>
        inline static u32 indexIcosahedronTriangle(const glm::tvec3<T, glm::highp>& direction)
        {
            float kT = AmbientDice::kT;
            float kT2 = kT * kT;
            
            ivec3 octantBit = ivec3(direction.x < 0 ? 1 : 0,
                                    direction.y < 0 ? 1 : 0,
                                    direction.z < 0 ? 1 : 0);
            
            u32 t = octantBit.x + octantBit.y * 2 + octantBit.z * 4;
            u32 tRed = 8 + octantBit.y + octantBit.z * 2;
            u32 tGreen = 12 + octantBit.x + octantBit.z * 2;
            u32 tBlue = 16 + octantBit.x + octantBit.y * 2;
            
            // Selection
            bool vertASelect = dot(abs(direction), vec3(1.0, kT2, -kT)) > 0.0;
            bool vertBSelect = dot(abs(direction), vec3(-kT, 1.0, kT2)) > 0.0;
            bool vertCSelect = dot(abs(direction), vec3(kT2, -kT, 1.0)) > 0.0;
            
            t = vertASelect ? t : tRed;
            t = vertBSelect ? t : tGreen;
            t = vertCSelect ? t : tBlue;
            
            return t;
        }
        
        template <typename T>
        inline static void computeBarycentrics(vec3 direction, u32 *triIndexOut, u32 *i0Out, u32 *i1Out, u32 *i2Out, T *b0Out, T *b1Out, T *b2Out)
        {
            u32 triIndex = AmbientDice::indexIcosahedronTriangle(direction);
            
            *triIndexOut = triIndex;
            
            u32 i0 = AmbientDice::triangleIndices[triIndex][0];
            u32 i1 = AmbientDice::triangleIndices[triIndex][1];
            u32 i2 = AmbientDice::triangleIndices[triIndex][2];
            
            *i0Out = i0;
            *i1Out = i1;
            *i2Out = i2;
            
            vec3 n0 = AmbientDice::triangleBarycentricNormals[triIndex][0];
            vec3 n1 = AmbientDice::triangleBarycentricNormals[triIndex][1];
            vec3 n2 = AmbientDice::triangleBarycentricNormals[triIndex][2];
            
            *b0Out = dot(direction, n0);
            *b1Out = dot(direction, n1);
            *b2Out = dot(direction, n2);
        }
        
        static Eigen::MatrixXf computeGramMatrixBezier();
        static Eigen::MatrixXf computeGramMatrixSRBF();
        static Eigen::MatrixXf computeGramMatrixLinear();
        
        template <typename T>
        static void hybridCubicBezierWeights(u32 triIndex, float b0, float b1, float b2, VertexWeights<T> *w0, VertexWeights<T> *w1, VertexWeights<T> *w2);
        
        template <typename T>
        static void hybridCubicBezierWeights(vec3 direction, u32 *i0Out, u32 *i1Out, u32 *i2Out, VertexWeights<T> *w0Out, VertexWeights<T> *w1Out, VertexWeights<T> *w2Out);
        
        template <typename T>
        static void srbfWeights(vec3 direction, T *weightsOut);
        
        inline vec3 evaluateLinear(const vec3& direction) const
        {
            u32 triIndex;
            u32 i0, i1, i2;
            float b0, b1, b2;
            this->computeBarycentrics(direction, &triIndex, &i0, &i1, &i2, &b0, &b1, &b2);
            
            return b0 * this->vertices[i0].value + b1 * this->vertices[i1].value + b2 * this->vertices[i2].value;
        }
        
        vec3 evaluateSRBF(const vec3& direction) const;
        
        inline vec3 evaluateBezier(const vec3& direction) const
        {
            u32 i0, i1, i2;
            AmbientDice::VertexWeights<float> weights[3];
            this->hybridCubicBezierWeights(direction, &i0, &i1, &i2, &weights[0], &weights[1], &weights[2]);
            
            vec3 result =
            weights[0].value * this->vertices[i0].value +
            weights[0].directionalDerivativeU * this->vertices[i0].directionalDerivativeU +
            weights[0].directionalDerivativeV * this->vertices[i0].directionalDerivativeV +
            weights[1].value * this->vertices[i1].value +
            weights[1].directionalDerivativeU * this->vertices[i1].directionalDerivativeU +
            weights[1].directionalDerivativeV * this->vertices[i1].directionalDerivativeV +
            weights[2].value * this->vertices[i2].value +
            weights[2].directionalDerivativeU * this->vertices[i2].directionalDerivativeU +
            weights[2].directionalDerivativeV * this->vertices[i2].directionalDerivativeV;
            
            return  result;
        }
        
        inline vec3 evaluateBezierYCoCg(const vec3& direction) const
        {
            u32 i0, i1, i2;
            AmbientDice::VertexWeights<float> weights[3];
            this->hybridCubicBezierWeights(direction, &i0, &i1, &i2, &weights[0], &weights[1], &weights[2]);
            
            u32 triIndex;
            float b0, b1, b2;
            this->computeBarycentrics(direction, &triIndex, &i0, &i1, &i2, &b0, &b1, &b2);
            
            float Yresult =
            weights[0].value * this->vertices[i0].value.r +
            weights[0].directionalDerivativeU * this->vertices[i0].directionalDerivativeU.r +
            weights[0].directionalDerivativeV * this->vertices[i0].directionalDerivativeV.r +
            weights[1].value * this->vertices[i1].value.r +
            weights[1].directionalDerivativeU * this->vertices[i1].directionalDerivativeU.r +
            weights[1].directionalDerivativeV * this->vertices[i1].directionalDerivativeV.r +
            weights[2].value * this->vertices[i2].value.r +
            weights[2].directionalDerivativeU * this->vertices[i2].directionalDerivativeU.r +
            weights[2].directionalDerivativeV * this->vertices[i2].directionalDerivativeV.r;
            
            float CoResult = b0 * this->vertices[i0].value.g + b1 * this->vertices[i1].value.g + b2 * this->vertices[i2].value.g;
            float CgResult = b0 * this->vertices[i0].value.b + b1 * this->vertices[i1].value.b + b2 * this->vertices[i2].value.b;
            
            return YCoCTo2RGB(vec3(Yresult, CoResult, CgResult));
        }
    };

    enum AmbientDiceType {
        AmbientDiceTypeLinear,
        AmbientDiceTypeBezier,
        AmbientDiceTypeSRBF,
        AmbientDiceTypeBezierYCoCg
    };
    
    class ExperimentAmbientDice : public Experiment
    {
        public:
        
        static AmbientDice solveAmbientDiceLeastSquaresLinear(ImageBase<vec3>& directions, const Image& irradiance);
        static AmbientDice solveAmbientDiceLeastSquaresBezier(ImageBase<vec3>& directions, const Image& irradiance);
        static AmbientDice solveAmbientDiceLeastSquaresBezierYCoCg(ImageBase<vec3>& directions, const Image& irradiance);
        static AmbientDice solveAmbientDiceLeastSquaresSRBF(ImageBase<vec3>& directions, const Image& irradiance);
        
        void run(SharedData& data) override;
        
        ExperimentAmbientDice& setDiceType(AmbientDiceType diceType) {
            this->m_diceType = diceType;
            return *this;
        }
        
    private:
        AmbientDiceType m_diceType = AmbientDiceTypeBezier;
    };
    
}
