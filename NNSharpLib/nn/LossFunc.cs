using NNSharp.Extension;

namespace NNSharp.nn
{
    public abstract class LossFunc
    {
        public Tensor Loss { get; set; }
        public float LossValue { get; set; }

        public abstract void Forward(Tensor pred, Tensor target);

        public void Backward()
        {
            Loss.Backward(Tensor.One);
        }
    }
}
