namespace NNSharp.autograd
{
    public static class TensorOperator
    {
        public static int FlattenIndex(this Tensor source, int[] indices)
        {
            int flatIndex = 0;
            int stride = 1;

            for (int i = source.Shape.Length - 1; i >= 0; i--)
            {
                flatIndex += stride * indices[i];
                stride *= source.Shape[i];
            }

            return flatIndex;
        }

        public static void Repeat(this Tensor source, int dimension, int count)
        {
            if (dimension < 0 || dimension >= source.Shape.Length)
            {
                throw new Exception("维度无效或数组形状不匹配。");
            }

            int[] newShape = (int[])source.Shape.Clone();
            newShape[dimension] *= count;

            int[] sourceStride = new int[source.Shape.Length];
            int[] destStride = new int[newShape.Length];

            CalculateStrides(source.Shape, sourceStride);
            CalculateStrides(newShape, destStride);

            float[] newData = new float[SizeFromShape(newShape)];

            int[] sourceIndices = new int[source.Shape.Length];
            int[] destIndices = new int[newShape.Length];

            for (int i = 0; i < source.Data.Length; i++)
            {
                GetIndices(i, source.Shape, sourceIndices);

                for (int j = 0; j < count; j++)
                {
                    destIndices = sourceIndices.ToArray();
                    destIndices[dimension] = j * source.Shape[dimension] + sourceIndices[dimension];

                    int sourceFlatIndex = SourceToFlatIndex(sourceIndices, source.Shape, sourceStride);
                    int destFlatIndex = DestToFlatIndex(destIndices, newShape, destStride);

                    newData[destFlatIndex] = source.Data[sourceFlatIndex];
                }
            }

            source.Data = newData;
            source.Shape = newShape;

            void GetIndices(int flatIndex, int[] shape, int[] indices)
            {
                int currentDim = indices.Length - 1;
                indices[currentDim] = flatIndex % shape[currentDim];
                for (int i = indices.Length - 2; i >= 0; i--)
                {
                    flatIndex /= shape[i + 1];
                    indices[i] = flatIndex % shape[i];
                }
            }

            void CalculateStrides(int[] shape, int[] strides)
            {
                strides[strides.Length - 1] = 1;
                for (int i = strides.Length - 2; i >= 0; i--)
                {
                    strides[i] = strides[i + 1] * shape[i + 1];
                }
            }

            int SizeFromShape(int[] shape)
            {
                int size = 1;
                foreach (int dim in shape)
                {
                    size *= dim;
                }
                return size;
            }

            int SourceToFlatIndex(int[] indices, int[] shape, int[] strides)
            {
                int flatIndex = 0;
                for (int i = 0; i < indices.Length; i++)
                {
                    flatIndex += strides[i] * indices[i];
                }
                return flatIndex;
            }

            int DestToFlatIndex(int[] indices, int[] shape, int[] strides)
            {
                int flatIndex = 0;
                for (int i = 0; i < indices.Length; i++)
                {
                    flatIndex += strides[i] * indices[i];
                }
                return flatIndex;
            }
        }

        [Obsolete]
        public static (Tensor, Tensor) Broadcast_Old(Tensor a, Tensor b)
        {
            int[] newShape = new int[Math.Max(a.Shape.Length, b.Shape.Length)];
            int[] newShapeA = new int[newShape.Length];
            int[] newShapeB = new int[newShape.Length];

            if (a.Shape.Length == b.Shape.Length - 1)
            {
                List<int> temp = a.Shape.ToList();
                temp.Insert(0, 1);
                newShapeA = temp.ToArray();
            }
            else
            {
                newShapeA = a.Shape;
            }
            if (a.Shape.Length - 1 == b.Shape.Length)
            {
                List<int> temp = b.Shape.ToList();
                temp.Insert(0, 1);
                newShapeB = temp.ToArray();
            }
            else
            {
                newShapeB = b.Shape;
            }

            if (newShapeA.Length == newShapeB.Length)
            {
                for (int i = 0; i < newShape.Length; i++)
                {
                    if (newShapeA[i] == newShapeB[i] || newShapeA[i] == 1 || newShapeB[i] == 1)
                    {
                        newShape[i] = Math.Max(newShapeA[i], newShapeB[i]);
                    }
                    else
                    {
                        throw new Exception("数组形状不匹配。");
                    }
                }
            }
            else
            {
                throw new Exception("数组形状不匹配。");
            }

            Tensor resultA = a.CopyOnlyData();
            Tensor resultB = b.CopyOnlyData();
            resultA.Shape = newShapeA;
            resultB.Shape = newShapeB;

            for (int i = 0; i < newShape.Length; i++)
            {
                if (newShapeA[i] != newShape[i])
                {
                    resultA.Repeat(i, newShape[i]);
                }
                if (newShapeB[i] != newShape[i])
                {
                    resultB.Repeat(i, newShape[i]);
                }
            }

            return (resultA, resultB);
        }

        public static (Tensor, Tensor) Broadcast(Tensor a, Tensor b)
        {
            int maxLength = Math.Max(a.Shape.Length, b.Shape.Length);
            int[] newShape = new int[maxLength];
            int[] newShapeA = new int[maxLength];
            int[] newShapeB = new int[maxLength];

            if (a.Shape.Length < b.Shape.Length)
            {
                Array.Copy(a.Shape, 0, newShapeA, 1, a.Shape.Length);
                for (int i = 0; i < a.Shape.Length - i; i++)
                {
                    newShapeA[i] = 1;
                }
            }
            else
            {
                newShapeA = a.Shape;
            }

            if (b.Shape.Length < a.Shape.Length)
            {
                Array.Copy(b.Shape, 0, newShapeB, 1, b.Shape.Length);
                for (int i = 0; i < b.Shape.Length - i; i++)
                {
                    newShapeB[i] = 1;
                }
            }
            else
            {
                newShapeB = b.Shape;
            }

            if (newShapeA.Length == newShapeB.Length)
            {
                for (int i = 0; i < maxLength; i++)
                {
                    if (newShapeA[i] == newShapeB[i] || newShapeA[i] == 1 || newShapeB[i] == 1)
                    {
                        newShape[i] = Math.Max(newShapeA[i], newShapeB[i]);
                    }
                    else
                    {
                        throw new ArgumentException("数组形状不匹配。");
                    }
                }
            }
            else
            {
                throw new ArgumentException("数组形状不匹配。");
            }

            Tensor resultA = a.CopyOnlyData();
            Tensor resultB = b.CopyOnlyData();
            resultA.Shape = newShapeA;
            resultB.Shape = newShapeB;

            for (int i = 0; i < maxLength; i++)
            {
                if (newShapeA[i] != newShape[i])
                {
                    resultA.Repeat(i, newShape[i]);
                }
                if (newShapeB[i] != newShape[i])
                {
                    resultB.Repeat(i, newShape[i]);
                }
            }

            return (resultA, resultB);
        }
    }
}
