namespace NNSharp.Extension
{
    public static class TensorEx
    {
        public static Tensor ToTensor(this float input)
        {
            return new Tensor([input], [1], false);
        }

        public static Tensor ToTensor(this float input, bool requireGrad = false, string name = "")
        {
            return new Tensor([input], [1], requireGrad, name);
        }

        public static Tensor ToTensor(this int input)
        {
            return new Tensor([input], [1], false);
        }

        public static Tensor ToTensor(this int input, bool requireGrad = false, string name = "")
        {
            return new Tensor([input], [1], requireGrad, name);
        }
    }
}
