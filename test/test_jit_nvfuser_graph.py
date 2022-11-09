import torch
from torch import nn
import os
# sh build.sh
# PYTORCH_JIT_LOG_LEVEL="graph_fuser" PYTORCH_NVFUSER_DUMP="lower2device" python test/test_jit_nvfuser_graph.py
class InteractionPPBlock(torch.jit.ScriptModule):
    def __init__(
        self,
       lookup_size,
       feat_dim,
       out_feat_dim,
       num_elements
    ):
        super(InteractionPPBlock, self).__init__()
        self.lin_sbf = nn.Linear(feat_dim, out_feat_dim)

    @torch.jit.script_method
    def forward(self, x_kj, idx_kj, sbf):
        sbf_res = torch.index_select(x_kj, 0, idx_kj) * sbf
        sbf_res = sbf_res + 17
        res = self.lin_sbf(sbf_res)
        return sbf_res

def main():
    lookup_size = 68
    feat_dim = 128
    out_feat_dim = 64
    num_elements = 355984
    lookup_tv = torch.rand((lookup_size, feat_dim)).to("cuda:0")
    indies_tv = torch.randint(0, lookup_size, (num_elements,)).to("cuda:0")

    sbf = torch.rand((num_elements, feat_dim)).to("cuda:0")
    model = InteractionPPBlock(lookup_size, feat_dim, out_feat_dim, num_elements).to("cuda:0")
    # pattern_matching(model.graph)
    with torch.jit.fuser("fuser2"):
        for i in range(5):
            res = model(lookup_tv, indies_tv, sbf)
            # loss = torch.mean(res)
            # loss.backward()
    print(model.graph)

if __name__ == "__main__":
    main()