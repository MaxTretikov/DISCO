import torch

from disco.training.bitnet import BitLinear, apply_bitnet_quantization


def test_bitlinear_backward_for_binary_and_ternary_weights():
    for weight_bits in (1, 1.58):
        layer = BitLinear(
            in_features=8,
            out_features=4,
            weight_bits=weight_bits,
            activation_bits=8,
        )
        input_tensor = torch.randn(3, 8, requires_grad=True)

        loss = layer(input_tensor).square().mean()
        loss.backward()

        assert layer.weight.grad is not None
        assert input_tensor.grad is not None


def test_apply_bitnet_quantization_skips_frozen_linears():
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 2),
    )
    model[0].requires_grad_(False)

    report = apply_bitnet_quantization(
        model,
        {
            "weight_bits": 1.58,
            "activation_bits": 8,
            "only_trainable": True,
        },
    )

    assert isinstance(model[0], torch.nn.Linear)
    assert isinstance(model[2], BitLinear)
    assert report.replaced_modules == 1
    assert report.skipped_modules == 1
