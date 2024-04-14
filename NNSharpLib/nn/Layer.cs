namespace NNSharp.nn
{
    public abstract class Layer
    {
        public abstract Tensor Forward(Tensor input);

        public abstract void InitWeights();

        public abstract void ZeroGrad();

        public abstract void ApplyGrad();
    }
}
