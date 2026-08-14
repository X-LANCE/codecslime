"""Command-line interface."""

import argparse

import torch

from .api import CodecSlime
from .audio import load_audio, load_tokens, save_audio, save_tokens


def _frame_rate(value: str) -> float:
    frame_rate = float(value)
    if not 36 <= frame_rate <= 80:
        raise argparse.ArgumentTypeError("frame rate must be between 36 and 80 Hz")
    return frame_rate


def main():
    parser = argparse.ArgumentParser(prog="codecslime")
    parser.add_argument("command", choices=("encode", "decode", "reconstruct"))
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--variant", choices=("vq8k", "fsq18k"), default="vq8k")
    parser.add_argument("--frame-rate", type=_frame_rate, default=40.0, metavar="HZ")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    codec = CodecSlime(args.variant, args.checkpoint, args.device) if args.checkpoint else CodecSlime.from_pretrained(args.variant, args.device)
    if args.command == "encode":
        waveform = load_audio(args.input)
        codes, durations, length = codec.encode(waveform, args.frame_rate)
        save_tokens(args.output, codes, durations, variant=args.variant, frame_rate=args.frame_rate, length=length)
    elif args.command == "decode":
        payload = load_tokens(args.input)
        waveform = codec.decode(payload["codes"], payload["durations"], int(payload["length"]))
        save_audio(args.output, waveform)
    else:
        waveform = codec.reconstruct(load_audio(args.input), args.frame_rate)
        save_audio(args.output, waveform)


if __name__ == "__main__":
    main()
