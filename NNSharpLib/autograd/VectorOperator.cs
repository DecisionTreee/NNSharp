using System.Numerics;

namespace NNSharp.autograd
{
    public class VectorOperator
    {
        public static float[] Add(float[] a, float[] b)
        {
            float[] result = new float[a.Length];
            int alignedLength = result.Length - result.Length % Vector<float>.Count;

            for (int i = 0; i < alignedLength; i += Vector<float>.Count)
            {
                Vector<float> tempA = new Vector<float>(a, i);
                Vector<float> tempB = new Vector<float>(b, i);
                Vector<float> res = Vector.Add(tempA, tempB);
                res.CopyTo(result, i);
            }

            for (int i = alignedLength; i < result.Length; i++)
            {
                result[i] = a[i] + b[i];
            }

            return result;
        }

        public static float[] Neg(float[] a)
        {
            float[] result = new float[a.Length];
            int alignedLength = result.Length - result.Length % Vector<float>.Count;

            for (int i = 0; i < alignedLength; i += Vector<float>.Count)
            {
                Vector<float> tempA = new Vector<float>(a, i);
                Vector<float> res = Vector.Negate(tempA);
                res.CopyTo(result, i);
            }

            for (int i = alignedLength; i < result.Length; i++)
            {
                result[i] = -a[i];
            }

            return result;
        }

        public static float[] Sub(float[] a, float[] b)
        {
            float[] result = new float[a.Length];
            int alignedLength = result.Length - result.Length % Vector<float>.Count;

            for (int i = 0; i < alignedLength; i += Vector<float>.Count)
            {
                Vector<float> tempA = new Vector<float>(a, i);
                Vector<float> tempB = new Vector<float>(b, i);
                Vector<float> res = Vector.Subtract(tempA, tempB);
                res.CopyTo(result, i);
            }

            for (int i = alignedLength; i < result.Length; i++)
            {
                result[i] = a[i] - b[i];
            }

            return result;
        }

        public static float[] Mul(float[] a, float[] b)
        {
            float[] result = new float[a.Length];
            int alignedLength = result.Length - result.Length % Vector<float>.Count;

            for (int i = 0; i < alignedLength; i += Vector<float>.Count)
            {
                Vector<float> tempA = new Vector<float>(a, i);
                Vector<float> tempB = new Vector<float>(b, i);
                Vector<float> res = Vector.Multiply(tempA, tempB);
                res.CopyTo(result, i);
            }

            for (int i = alignedLength; i < result.Length; i++)
            {
                result[i] = a[i] * b[i];
            }

            return result;
        }

        public static float[] Div(float[] a, float[] b)
        {
            float[] result = new float[a.Length];
            int alignedLength = result.Length - result.Length % Vector<float>.Count;

            for (int i = 0; i < alignedLength; i += Vector<float>.Count)
            {
                Vector<float> tempA = new Vector<float>(a, i);
                Vector<float> tempB = new Vector<float>(b, i);
                Vector<float> res = Vector.Divide(tempA, tempB);
                res.CopyTo(result, i);
            }

            for (int i = alignedLength; i < result.Length; i++)
            {
                result[i] = a[i] / b[i];
            }

            return result;
        }

        public static float Dot(float[] a, float[] b)
        {
            float result = 0f;
            int alignedLength = a.Length - a.Length % Vector<float>.Count;

            for (int i = 0; i < alignedLength; i += Vector<float>.Count)
            {
                Vector<float> tempA = new Vector<float>(a, i);
                Vector<float> tempB = new Vector<float>(b, i);
                result += Vector.Dot(tempA, tempB);
            }

            for (int i = alignedLength; i < a.Length; i++)
            {
                result += a[i] * b[i];
            }

            return result;
        }
    }
}
