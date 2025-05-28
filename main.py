from argparse import ArgumentParser
from lmetk.setup import configure, initialize


def create_parser():
    parser = ArgumentParser()

    parser.add_argument("--guild_mode", action="store_true", help="guild mode")

    parser.add_argument(
        "--dataset_dir", type=str, default=None, help="dataset directory"
    )
    parser.add_argument(
        "--working_dir", type=str, default=None, help="working directory"
    )
    parser.add_argument(
        "--exp_name", default="default", type=str, help="experiment name"
    )

    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    print("*" * 80)
    print("[ARGPARSE]:", args)
    print("*" * 80)
    cfg = configure(
        args.exp_name,
        data_dir=args.dataset_dir,
        working_dir=args.working_dir,
        guild_mode=args.guild_mode,
    )

    exp = initialize(cfg)
    exp.setup()
    exp.run()
