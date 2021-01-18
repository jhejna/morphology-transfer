import argparse
import os

from bot_transfer.utils.tester import test, test_composition
from bot_transfer.utils.loader import BASE

parser = argparse.ArgumentParser()
parser.add_argument('--name', "-n", help='name of checkpoint model', required=True)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', "-p", help='name of checkpoint model', required=True)
    parser.add_argument('--low-level', "-l", help="Low Level Policy", default=None)
    parser.add_argument('--episodes', "-e", type=int, default=10)
    parser.add_argument('--gif', "-g", action='store_true', default=False)
    parser.add_argument('--undeterministic', "-u", action='store_false', default=True)
    args = parser.parse_args()

    if not args.low_level is None:
        test_composition(args.low_level, args.path, num_ep=args.episodes, deterministic=args.undeterministic, gif=args.gif, render=True)
    else:
        # Auto forward the path until we reach a model.
        folder = args.path
        if not folder.startswith('/'):
            folder = os.path.join(BASE, folder)
        while 'params.json' not in os.listdir(folder):
            contents = [os.path.join(folder, d) for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
            assert len(contents) == 1, "Traversing down directory with multiple paths."
            folder = os.path.join(folder, contents[0])
        
        test(folder, num_ep=args.episodes, gif=args.gif, deterministic=args.undeterministic)

if __name__ == '__main__':
    main()