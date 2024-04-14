using NNSharp.Extension;

namespace NNSharp.nn
{
    public class MSELoss : LossFunc
    {
        public Tensor PowFactor { get; set; }
        public bool WithBatch { get; set; }

        public MSELoss(float powFactor = 2f, bool withBatch = false)
        {
            PowFactor = powFactor.ToTensor();
            WithBatch = withBatch;
        }

        public override void Forward(Tensor pred, Tensor target)
        {
            Loss = Tensor.Pow(pred - target, PowFactor);
            if (WithBatch)
            {
                Loss /= pred.Shape[0].ToTensor();
            }
        }
    }
}
