import dgl
import torch
from dgl.nn.pytorch import GraphConv
from sklearnex import patch_sklearn

patch_sklearn()

from sklearn.decomposition import PCA


class PCAPass:
    def __init__(
        self,
        khop: int,
        hidden_feats: int,
        norm: str = 'right',
        standardize: bool = False,
        seed: int = 13,
    ) -> None:
        self._khop = khop
        self._hidden_feats = hidden_feats
        self._standardize = standardize

        self._conv = GraphConv(1, 1, norm=norm, weight=False,
                               bias=False, allow_zero_in_degree=True)

        self._decomposers = [PCA(hidden_feats, random_state=seed)
                             for _ in range(khop)]

    def __call__(self, g: dgl.DGLGraph, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward(g, inputs)

    def _validate_inputs(self, inputs: torch.Tensor) -> None:
        max_dim = 2 * inputs.shape[-1]

        if self._hidden_feats > max_dim:
            raise ValueError(
                f'Number of hidden feats ({self._hidden_feats}) couldn\'t be '
                f'higher than 2 * inputs last dimension ({max_dim}) '
                f'because of PCA dimensionality reduction restrictions.'
            )

    def _standardize_data(self, inputs: torch.Tensor, ) -> torch.Tensor:
        std, mean = torch.std_mean(inputs, dim=-1, keepdim=True)

        # change 0 std to 1 to resolve NaN issue (same approach as in sklearn)
        std[std == 0] = 1

        x = (inputs - mean) / std

        return x

    def forward(self, g: dgl.DGLGraph, inputs: torch.Tensor):
        self._validate_inputs(inputs)

        x = inputs

        for decomposer in self._decomposers:
            x_k = self._conv(g, x)
            x = torch.cat([x, x_k], dim=-1)

            if self._standardize:
                x = self._standardize_data(x)

            x = torch.from_numpy(decomposer.fit_transform(x)).to(inputs.dtype)

        return x

    def inference(self, g: dgl.DGLGraph, inputs: torch.Tensor):
        self._validate_inputs(inputs)

        for decomposer in self._decomposers:
            x_k = self._conv(g, x)
            x = torch.cat([x, x_k], dim=-1)

            if self._standardize:
                x = self._standardize_data(x)

            x = torch.from_numpy(decomposer.transform(x)).to(inputs.dtype)

        return x
