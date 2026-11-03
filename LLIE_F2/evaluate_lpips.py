import argparse
from pathlib import Path

import lpips
import torch
from PIL import Image
from torchvision.transforms import ToTensor


def load_image(path: Path, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    tensor = ToTensor()(image).unsqueeze(0)
    return tensor.to(device) * 2.0 - 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate mean LPIPS for output/GT image pairs.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--net", choices=("alex", "vgg", "squeeze"), default="alex")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    if not args.output.is_dir():
        raise FileNotFoundError("Output directory does not exist: {}".format(args.output))
    if not args.gt.is_dir():
        raise FileNotFoundError("GT directory does not exist: {}".format(args.gt))

    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    output_files = {path.name: path for path in args.output.iterdir() if path.is_file()}
    gt_files = {path.name: path for path in args.gt.iterdir() if path.is_file()}
    names = sorted(set(output_files) & set(gt_files))
    missing_output = sorted(set(gt_files) - set(output_files))
    missing_gt = sorted(set(output_files) - set(gt_files))

    if missing_output or missing_gt:
        raise RuntimeError(
            "Image pairs do not match. Missing output: {}; missing GT: {}.".format(
                missing_output[:5], missing_gt[:5]
            )
        )
    if not names:
        raise RuntimeError("No matching image pairs found.")

    metric = lpips.LPIPS(net=args.net).to(device)
    metric.eval()
    total = 0.0

    with torch.no_grad():
        for name in names:
            output = load_image(output_files[name], device)
            gt = load_image(gt_files[name], device)
            if output.shape != gt.shape:
                raise RuntimeError(
                    "Image size mismatch for {}: output {}, GT {}.".format(
                        name, tuple(output.shape), tuple(gt.shape)
                    )
                )
            total += float(metric(output, gt).item())

    print("LPIPS network: {}".format(args.net))
    print("Device: {}".format(device))
    print("Images: {}".format(len(names)))
    print("Mean LPIPS: {:.6f}".format(total / len(names)))


if __name__ == "__main__":
    main()
