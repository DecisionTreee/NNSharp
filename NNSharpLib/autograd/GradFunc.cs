namespace NNSharp.autograd
{
    public class GradFunc
    {
        public static (Tensor, Tensor) AddGradFn(Tensor gradient, Tensor leftChild, Tensor rightChild)
        {
            return (gradient, gradient);
        }

        public static (Tensor, Tensor) NegGradFn(Tensor gradient, Tensor leftChild, Tensor rightChild)
        {
            return (-gradient, null);
        }

        public static (Tensor, Tensor) SubGradFn(Tensor gradient, Tensor leftChild, Tensor rightChild)
        {
            return (gradient, -gradient);
        }

        public static (Tensor, Tensor) MulGradFn(Tensor gradient, Tensor leftChild, Tensor rightChild)
        {
            return (gradient * rightChild, gradient * leftChild);
        }

        public static (Tensor, Tensor) DivGradFn(Tensor gradient, Tensor leftChild, Tensor rightChild)
        {
            return (gradient / rightChild, -leftChild * gradient / (rightChild * rightChild));
        }

        public static (Tensor, Tensor) LnGradFn(Tensor gradient, Tensor leftChild, Tensor rightChild)
        {
            return (gradient / leftChild, null);
        }

        public static (Tensor, Tensor) PowGradFn(Tensor gradient, Tensor leftChild, Tensor rightChild)
        {
            return (gradient * rightChild * Tensor.Pow(leftChild, rightChild - new Tensor([1f], [1])), gradient * Tensor.Pow(leftChild, rightChild) * Tensor.Ln(leftChild));
        }

        public static (Tensor, Tensor) MatMulGradFn(Tensor gradient, Tensor leftChild, Tensor rightChild)
        {
            if (leftChild.Shape[1] != rightChild.Shape[0])
            {
                throw new Exception("矩阵大小不匹配。");
            }

            Tensor dALeft = new Tensor(new float[leftChild.Data.Length], leftChild.Shape, leftChild.RequireGrad);
            for (int i = 0; i < leftChild.Shape[0]; i++)
            {
                for (int j = 0; j < leftChild.Shape[1]; j++)
                {
                    for (int k = 0; k < rightChild.Shape[1]; k++)
                    {
                        dALeft[i, j] += gradient[i, k] * rightChild[j, k];
                    }
                }
            }
            Tensor dBRight = new Tensor(new float[rightChild.Data.Length], rightChild.Shape, rightChild.RequireGrad);
            for (int i = 0; i < rightChild.Shape[0]; i++)
            {
                for (int j = 0; j < rightChild.Shape[1]; j++)
                {
                    for (int k = 0; k < leftChild.Shape[0]; k++)
                    {
                        dBRight[i, j] += leftChild[k, i] * gradient[k, j];
                    }
                }
            }

            return (dALeft, dBRight);
        }
    }
}