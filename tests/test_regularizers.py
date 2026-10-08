import torch

from src.photonic_training.regularizers import RegularizationConfig, VoltageRegularizer


def test_l1_and_sm_match_the_stated_definitions() -> None:
    voltages = torch.tensor([[[1.0, -2.0]], [[4.0, 2.0]]])
    regularizer = VoltageRegularizer(RegularizationConfig(l1=1.0, sm=1.0))
    penalties = regularizer(voltages)
    assert torch.isclose(penalties["l1"], torch.tensor(9.0))
    assert torch.isclose(penalties["sm"], torch.sqrt(torch.tensor(25.0)))
    assert torch.isclose(penalties["weighted"], torch.tensor(14.0))
