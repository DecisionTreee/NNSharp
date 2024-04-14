using NNSharp.autograd;
using NNSharp.Extension;
using System.Diagnostics.Contracts;

namespace NNSharp
{
    public class Tensor
    {
        public string Name { get; set; }
        public float[] Data { get; set; }
        public int[] Shape { get; set; }
        public bool RequireGrad { get; set; }
        public Tensor Gradient { get; set; }
        public Func<Tensor, Tensor, Tensor, (Tensor, Tensor)> GradFn { get; set; }
        public bool IsLeaf { get; set; }
        public Tensor Father { get; set; }
        public Tensor LeftChild { get; set; }
        public Tensor RightChild { get; set; }

        public static Tensor One { get; set; } = 1f.ToTensor();

        public float this[params int[] indices]
        {
            get
            {
                int index = this.FlattenIndex(indices);
                return Data[index];
            }
            set
            {
                int index = this.FlattenIndex(indices);
                Data[index] = value;
            }
        }

        public Tensor()
        {

        }

        public Tensor(float[] data, int[] shape, bool requireGrad = false, string name = "")
        {
            Data = data;
            Shape = shape;
            RequireGrad = requireGrad;
            Name = name;
            if (RequireGrad)
            {
                ZeroGrad();
            }
            GradFn = null;
            IsLeaf = true;
            Father = null;
            LeftChild = null;
            RightChild = null;
        }

        public void ZeroGrad()
        {
            if (RequireGrad)
            {
                if (Gradient == null)
                {
                    Gradient = new Tensor(new float[Data.Length], Shape, false, $"{Name}_gradient");
                }
                else
                {
                    Array.Clear(Gradient.Data, 0, Gradient.Data.Length);
                }
            }
        }

        public static Tensor operator +(Tensor a, Tensor b)
        {
            Tensor broadA, broadB;
            (broadA, broadB) = TensorOperator.Broadcast(a, b);
            Tensor result = new Tensor(new float[broadA.Data.Length], broadA.Shape, a.RequireGrad || b.RequireGrad);

            result.Data = VectorOperator.Add(broadA.Data, broadB.Data);
            result.GradFn = GradFunc.AddGradFn;
            result.IsLeaf = false;
            result.LeftChild = a;
            result.LeftChild.Father = result;
            result.RightChild = b;
            result.RightChild.Father = result;

            return result;
        }

        public static Tensor operator -(Tensor a)
        {
            Tensor result = new Tensor(new float[a.Data.Length], a.Shape, a.RequireGrad);

            result.Data = VectorOperator.Neg(a.Data);
            result.GradFn = GradFunc.NegGradFn;
            result.IsLeaf = false;
            result.LeftChild = a;
            result.LeftChild.Father = result;
            result.RightChild = null;

            return result;
        }

        public static Tensor operator -(Tensor a, Tensor b)
        {
            Tensor broadA, broadB;
            (broadA, broadB) = TensorOperator.Broadcast(a, b);
            Tensor result = new Tensor(new float[broadA.Data.Length], broadA.Shape, a.RequireGrad || b.RequireGrad);

            result.Data = VectorOperator.Sub(broadA.Data, broadB.Data);
            result.GradFn = GradFunc.SubGradFn;
            result.IsLeaf = false;
            result.LeftChild = a;
            result.LeftChild.Father = result;
            result.RightChild = b;
            result.RightChild.Father = result;

            return result;
        }

        public static Tensor operator *(Tensor a, Tensor b)
        {
            Tensor broadA, broadB;
            (broadA, broadB) = TensorOperator.Broadcast(a, b);
            Tensor result = new Tensor(new float[broadA.Data.Length], broadA.Shape, a.RequireGrad || b.RequireGrad);

            result.Data = VectorOperator.Mul(broadA.Data, broadB.Data);
            result.GradFn = GradFunc.MulGradFn;
            result.IsLeaf = false;
            result.LeftChild = a;
            result.LeftChild.Father = result;
            result.RightChild = b;
            result.RightChild.Father = result;

            return result;
        }

        public static Tensor operator /(Tensor a, Tensor b)
        {
            Tensor broadA, broadB;
            (broadA, broadB) = TensorOperator.Broadcast(a, b);
            Tensor result = new Tensor(new float[broadA.Data.Length], broadA.Shape, a.RequireGrad || b.RequireGrad);

            result.Data = VectorOperator.Div(broadA.Data, broadB.Data);
            result.GradFn = GradFunc.DivGradFn;
            result.IsLeaf = false;
            result.LeftChild = a;
            result.LeftChild.Father = result;
            result.RightChild = b;
            result.RightChild.Father = result;

            return result;
        }

        public static Tensor Pow(Tensor a, Tensor b)
        {
            Tensor broadA, broadB;
            (broadA, broadB) = TensorOperator.Broadcast(a, b);
            Tensor result = new Tensor(new float[broadA.Data.Length], broadA.Shape, a.RequireGrad);

            for (int i = 0; i < result.Data.Length; i++)
            {
                result.Data[i] = MathF.Pow(broadA.Data[i], broadB.Data[i]);
            }

            result.GradFn = GradFunc.PowGradFn;
            result.IsLeaf = false;
            result.LeftChild = a;
            result.LeftChild.Father = result;
            result.RightChild = b;
            result.RightChild.Father = result;

            return result;
        }

        public static Tensor Ln(Tensor a)
        {
            Tensor result = new Tensor(new float[a.Data.Length], a.Shape, a.RequireGrad);

            for (int i = 0; i < result.Data.Length; i++)
            {
                result.Data[i] = MathF.Log(a.Data[i]);
            }

            result.GradFn = GradFunc.LnGradFn;
            result.IsLeaf = false;
            result.LeftChild = a;
            result.LeftChild.Father = result;
            result.RightChild = null;

            return result;
        }

        public static Tensor MatMul(Tensor a, Tensor b)
        {
            if (a.Shape[1] != b.Shape[0])
            {
                throw new ArgumentException("矩阵大小不匹配。");
            }

            int[] resultShape = [a.Shape[0], b.Shape[1]];
            Tensor result = new Tensor(new float[resultShape.Aggregate((acc, next) => acc * next)], resultShape, a.RequireGrad || b.RequireGrad);

            for (int i = 0; i < result.Shape[0]; i++)
            {
                for (int j = 0; j < result.Shape[1]; j++)
                {
                    float sum = 0;
                    for (int k = 0; k < a.Shape[1]; k++)
                    {
                        sum += a[i, k] * b[k, j];
                    }
                    result[i, j] = sum;
                }
            }

            result.GradFn = GradFunc.MatMulGradFn;
            result.IsLeaf = false;
            result.LeftChild = a;
            result.LeftChild.Father = result;
            result.RightChild = b;
            result.RightChild.Father = result;

            return result;
        }

        public void Backward(Tensor grad)
        {
            Gradient += grad;
            if (!IsLeaf)
            {
                Tensor leftGrad, rightGrad;
                (leftGrad, rightGrad) = GradFn(grad, LeftChild, RightChild);
                if (LeftChild != null && LeftChild.RequireGrad)
                {
                    LeftChild.Backward(leftGrad);
                }
                if (RightChild != null && RightChild.RequireGrad)
                {
                    RightChild.Backward(rightGrad);
                }
            }
        }

        public Tensor Copy()
        {
            return new Tensor()
            {
                Name = Name,
                Data = Data,
                Shape = Shape,
                RequireGrad = RequireGrad,
                Gradient = Gradient,
                GradFn = GradFn,
                IsLeaf = IsLeaf,
                Father = Father,
                LeftChild = LeftChild,
                RightChild = RightChild,
            };
        }

        public Tensor CopyOnlyData()
        {
            return new Tensor()
            {
                Data = Data
            };
        }
    }
}
