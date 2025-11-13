import torch
from model import KinNet

"""
During the review stage, we provide a runnable demo script (run_demo.py) for inference on sample data. 
Complete training and evaluation scripts, along with all configuration files, 
will be released once the paper is formally accepted.
"""

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = torch.randn(1, 12, 22, device=device)
    model = KinNet(fusionflag=1,
        in_size1=22, in_dim1=12,
        in_size2=11, in_dim2=24,
        in_size3=6, in_dim3=48,
        nclass=6,
    ).to(device)
    out1, out2, out3, final_out = model(inputs)
    print(f"Input shape     : {tuple(inputs.shape)}")
    print(f"Branch1 output  : {tuple(out1.shape)}")
    print(f"Branch2 output  : {tuple(out2.shape)}")
    print(f"Branch3 output  : {tuple(out3.shape)}")
    print(f"Final prediction: {tuple(final_out.shape)}")