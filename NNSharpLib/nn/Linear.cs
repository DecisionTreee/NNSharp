using NNSharp.Extension;
using System;

namespace NNSharp.nn
{
    public class Linear : Layer
    {
        public int NumNodesIn { get; set; }
        public int NumNodesOut { get; set; }
        public float LearningRate { get; set; }
        public bool InitWeight { get; set; }
        public float SqrtNumIn { get; set; }
        public Tensor Weights { get; set; }
        public bool HasBias { get; set; }
        public Tensor Biases { get; set; }
        public string Name { get; set; }

        public Linear(int numNodesIn, int numNodesOut, float learningRate, bool initWeight = true, bool hasBias = true, string name = "")
        {
            NumNodesIn = numNodesIn;
            NumNodesOut = numNodesOut;
            LearningRate = learningRate;
            InitWeight = initWeight;
            Name = name;
            SqrtNumIn = MathF.Sqrt(numNodesIn);
            Weights = new Tensor(new float[NumNodesIn * NumNodesOut], [NumNodesIn, NumNodesOut], true, $"{Name}_weights");
            HasBias = hasBias;
            if (HasBias)
            {
                Biases = new Tensor(new float[NumNodesOut], [NumNodesOut], true, $"{Name}_biases");
            }
            InitWeights();
        }

        public override Tensor Forward(Tensor input)
        {
            return Tensor.MatMul(input, Weights) + Biases;
        }

        public override void InitWeights()
        {
            if (InitWeight)
            {
                Random random = new Random();
                for (int i = 0; i < NumNodesOut; i++)
                {
                    for (int j = 0; j < NumNodesIn; j++)
                    {
                        Weights[j, i] = (random.NextSingle() * 2f - 1) / SqrtNumIn;
                    }
                    Biases[0, i] = (random.NextSingle() * 2f - 1) / SqrtNumIn;
                }
            }
        }

        public override void ZeroGrad()
        {
            Weights.ZeroGrad();
            Biases.ZeroGrad();
        }

        public override void ApplyGrad()
        {
            for (int i = 0; i < NumNodesOut; i++)
            {
                for (int j = 0; j < NumNodesIn; j++)
                {
                    Weights[j, i] -= Weights.Gradient[j, i] * LearningRate;
                }
                Biases[0, i] -= Biases.Gradient[0, i] * LearningRate;
            }
        }
    }
}
